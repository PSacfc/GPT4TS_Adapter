from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim

from layers.Embed import DataEmbedding, TokenEmbedding
from models.hugging_gpt2.GPT2 import GPT2Model
from einops import rearrange, repeat, reduce
import copy
import math

class Adapter(nn.Module):
    def __init__(self, in_feat, hid_dim):
        super().__init__()
        self.D_fc1 = nn.Linear(in_feat, hid_dim)
        self.D_fc2 = nn.Linear(hid_dim, in_feat)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(self.D_fc2(self.act(self.D_fc1(x))))
        

class SpectModule(nn.Module):
    def __init__(self, freq_len, adapter_len):
        super().__init__()
        self.adapter_len = adapter_len
        # self.weight = nn.Parameter(torch.rand(freq_len, adapter_len//2, dtype=torch.cfloat))
        self.weight_r = nn.Parameter(torch.rand(freq_len, adapter_len//2))
        self.weight_i = nn.Parameter(torch.rand(freq_len, adapter_len//2))
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        # B M N P
        B, M, N, P = x.shape
        x = rearrange(x, 'b m n p -> b m p n')
        x_ft = torch.fft.rfft(x, dim=-1)

        x_real = x_ft.real
        x_imag = x_ft.imag
        x_real = torch.einsum("bmpn, nd->bmpd", x_real, self.weight_r)
        x_imag = torch.einsum("bmpn, nd->bmpd", x_imag, self.weight_i)
        x_ft = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))

        res = torch.fft.irfft(x_ft, dim=-1, n=self.adapter_len)
        res = rearrange(res, 'b m p n -> b m n p')

        return self.drop(res)


class SpectBlock(nn.Module):
    def __init__(self, in_feat, freq_len, low_rank=8, adapter_len=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_feat)
        self.ln_2 = nn.LayerNorm(in_feat)
        self.attn = SpectModule(freq_len//2+1, adapter_len)
    
    def forward(self, x):
        # B M N P
        x = self.attn(self.ln_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class FFT_adapter(nn.Module):
    def __init__(self, n_layer, in_feat, seq_len):
        super().__init__()
        self.blocks = nn.ModuleList([SpectBlock(in_feat, seq_len) for i in range(n_layer)])

    def forward(self, x):
        # B, M, L
        res_list = []
        for i, block in enumerate(self.blocks):
            res_list.append(block(x))
        
        return res_list


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()

        # Patching Params
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = self.seq_len // self.stride
        
        self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_size, configs.d_model, 
                                           configs.embed, configs.freq, configs.dropout)
        # GPT2Model
        self.gpt_layers = configs.gpt_layers
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model    
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        for i in range(self.gpt_layers):
            self.gpt2.h[i].scale = configs.scale
            self.gpt2.h[i].attn.scale = configs.scale
            
            if configs.T_type:
                self.gpt2.h[i].T_adapter = Adapter(configs.d_model, configs.adapter_dim)
                self.gpt2.h[i].T_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.patch_num, 1))
            
            if configs.C_type:
                self.gpt2.h[i].C_adapter = Adapter(configs.d_model, configs.adapter_dim)
                self.gpt2.h[i].C_num = configs.enc_in
                self.gpt2.h[i].C_adapter_gate = torch.nn.Parameter(torch.zeros(1, configs.enc_in, 1))

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'adapter' in name or 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Frequency Adapter
        self.fft_adapter = FFT_adapter(configs.gpt_layers, configs.enc_in, self.patch_num)
        # self.adapter_in_layer = nn.ModuleList([nn.Linear(self.patch_size, configs.d_model) for i in range(self.gpt_layers)])
        self.adapter_in_layer = nn.ModuleList([nn.Linear(self.patch_size * configs.enc_in, configs.d_model) for i in range(self.gpt_layers)])
        self.in_layer = nn.Linear(self.patch_size, configs.d_model)

        # Reconstruction Projection Layer
        # self.proj_layer = nn.Linear(configs.d_model, configs.d_ff)
        # self.proj_layer = nn.Linear(configs.d_model, 1)

        self.proj_ln = nn.LayerNorm(configs.d_model)
        self.proj_cov_layer = nn.Conv1d(configs.d_model, configs.enc_in, 5, padding=2)
        self.proj_layer = nn.Linear(configs.d_model, configs.enc_in)
        
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask):

        B, L, M = x.shape
        # Normalization from Non-stationary Transformer
        means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x /= stdev

        fft_adapter_list = self.fft_adapter(rearrange(x, 'b (n p) m -> b m n p', p=1)) # B M N P
        adapters = []
        for i in range(len(fft_adapter_list)):
            fft_adapter_list[i] = rearrange(fft_adapter_list[i], 'b m n p -> b n (m p)')
            fft_adapter_list[i] = self.adapter_in_layer[i](fft_adapter_list[i])
            # fft_adapter_list[i] = rearrange(fft_adapter_list[i], 'b m n p -> b m n p')
            # print(fft_adapter_list[i].shape)
            adapters.append(fft_adapter_list[i])

        enc_out = self.enc_embedding(x, x_mark_enc)  # [B,T,C]

        outputs = self.gpt2(inputs_embeds=enc_out, adapters=adapters).last_hidden_state
        
        outputs = self.proj_ln(outputs)
        dec_out = self.proj_layer(outputs)
        dec_out = self.proj_cov_layer(outputs.permute(0, 2, 1)).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out
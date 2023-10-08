from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from models.hugging_gpt2.GPT2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from models.embed import DataEmbedding, DataEmbedding_wo_time

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

class gpt4ts(nn.Module):
    
    def __init__(self, config, data):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']
        self.scale = config['scale']
        self.adapter_dim = config['adapter_dim']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        for i in range(self.gpt_layers):
            self.gpt2.h[i].scale = self.scale
            self.gpt2.h[i].attn.scale = self.scale
            
            self.gpt2.h[i].T_adapter = Adapter(self.d_model, self.adapter_dim)
            self.gpt2.h[i].T_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.patch_num, 1))
            
            self.gpt2.h[i].C_adapter = Adapter(self.d_model, self.adapter_dim)
            self.gpt2.h[i].C_num = self.feat_dim
            self.gpt2.h[i].C_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.feat_dim, 1))
        
        # Frequency Adapter
        self.fft_adapter = FFT_adapter(self.gpt_layers, self.feat_dim, self.patch_num)
        self.adapter_in_layer = nn.ModuleList([nn.Linear(self.patch_size, self.d_model) for i in range(self.gpt_layers)])
        self.in_layer = nn.Linear(self.patch_size, self.d_model)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))
        self.gpt2.to(device=device)

        self.act = F.gelu
        
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)
        
    def forward(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        fft_adapter_list = self.fft_adapter(input_x) # B M N P
        adapters = []
        for i in range(self.gpt_layers - len(fft_adapter_list)):
            adapters.append(None)
        for i in range(len(fft_adapter_list)):
            fft_adapter_list[i] = self.adapter_in_layer[i](fft_adapter_list[i])
            fft_adapter_list[i] = rearrange(fft_adapter_list[i], 'b m n p -> (b m) n p')
            adapters.append(fft_adapter_list[i])
        
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        outputs = self.enc_embedding(input_x, None)
        
        outputs = self.gpt2(inputs_embeds=outputs, adapters=adapters).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs

    
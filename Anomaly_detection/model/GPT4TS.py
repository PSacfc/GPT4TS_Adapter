from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim

from .embed import DataEmbedding, TokenEmbedding
from .GPT2 import GPT2Model
from .attn import AttentionLayer, AnomalyAttention
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


class GPT4TS(nn.Module):
    
    def __init__(self, seq_len, enc_in, c_out, patch_len=3, stride=3, gpt_layers=3, adapter_dim=128, d_model=768, 
                 d_ff=32, dropout=0.0, scale=1000, output_attention=True, anomaly_layer=1):
        super(GPT4TS, self).__init__()

        # Patching Params
        self.seq_len = seq_len
        self.patch_size = patch_len
        self.stride = stride
        self.patch_num = seq_len // stride
        # self.patch_num = (seq_len - patch_len) // stride + 1
        # self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        # if stride > 1 or patch_len > 1:
        #     self.patch_num += 1
        
        # GPT2Model
        self.gpt_layers = gpt_layers
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model    
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        for i in range(gpt_layers):
            self.gpt2.h[i].scale = scale
            self.gpt2.h[i].attn.scale = scale
            
            self.gpt2.h[i].T_adapter = Adapter(d_model, adapter_dim)
            self.gpt2.h[i].T_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.patch_num, 1))
            
            self.gpt2.h[i].C_adapter = Adapter(d_model, adapter_dim)
            self.gpt2.h[i].C_num = enc_in
            self.gpt2.h[i].C_adapter_gate = torch.nn.Parameter(torch.zeros(1, enc_in, 1))

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'adapter' in name or 'ln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Frequency Adapter
        self.fft_adapter = FFT_adapter(gpt_layers, enc_in, self.patch_num)
        self.adapter_in_layer = nn.ModuleList([nn.Linear(patch_len, d_model) for i in range(gpt_layers)])
        self.in_layer = nn.Linear(patch_len, d_model)

        # Reconstruction Projection Layer
        self.proj_layer = nn.Linear(d_model, d_ff)
        self.out_layer = nn.Linear(d_ff * self.patch_num, seq_len)

        # Discrepancy Proj
        self.dis_proj = nn.ModuleList([nn.Linear(self.patch_num * enc_in, seq_len) for i in range(gpt_layers)])
        self.sigma_proj = nn.ModuleList([nn.Linear(enc_in, 12) for i in range(gpt_layers)])
        
        # self.sigma_proj_1 = nn.Linear(self.patch_num, seq_len)
        # self.sigma_proj_2 = nn.Linear(enc_in * d_model, 12)
        self.distances = torch.zeros((seq_len, seq_len)).cuda()
        for i in range(seq_len):
            for j in range(seq_len):
                self.distances[i][j] = abs(i - j)

    def forward(self, x, *args, **kwargs):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev
        
        enc_out = rearrange(x, 'b (n p) m -> b m n p', p=self.patch_size)

        fft_adapter_list = self.fft_adapter(enc_out) # B M N P
        adapters = []
        for i in range(self.gpt_layers - len(fft_adapter_list)):
            adapters.append(None)
        for i in range(len(fft_adapter_list)):
            fft_adapter_list[i] = self.adapter_in_layer[i](fft_adapter_list[i])
            fft_adapter_list[i] = rearrange(fft_adapter_list[i], 'b m n p -> (b m) n p')
            adapters.append(fft_adapter_list[i])

        enc_out = rearrange(enc_out, 'b m n p -> (b m) n p')
        outputs = self.in_layer(enc_out)
        outputs = self.gpt2(inputs_embeds=outputs, adapters=adapters)
        
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states
        outputs = outputs.last_hidden_state
        outputs = self.proj_layer(outputs)
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        attns = []
        for i in range(len(attentions)):
            temp = (attentions[i] + rearrange(attentions[i], 'b l s1 s2 -> b l s2 s1'))
            for tt in range(temp.shape[-1]):
                temp[:, :, tt, tt] /= 2
            temp = reduce(temp, '(b m) l s1 s2-> b l s1 s2', 'mean', b=B)
            temp = repeat(temp / (self.seq_len//self.patch_num) / (self.seq_len//self.patch_num), 
                          'b h s1 s2 -> b h (s1 l_a) (s2 l_b)', 
                          l_a=self.seq_len//self.patch_num, 
                          l_b=self.seq_len//self.patch_num)
            attns.append(temp)
        
        # priors = []
        # for i in range(len(hidden_states) - 1):
        #     temp = self.sigma_proj_1(hidden_states[i].permute(0, 2, 1)).permute(0, 2, 1)
        #     temp = rearrange(temp, '(b m) l d -> b l (m d)', b=B)
        #     sigma = self.sigma_proj_2(temp).permute(0, 2, 1)

        #     sigma = torch.sigmoid(sigma * 5) + 1e-5
        #     sigma = torch.pow(3, sigma) - 1
        #     sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, self.seq_len)
        #     prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        #     prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        #     priors.append(prior)

        priors = []
        for i, (dis_layer, sigma_layer) in enumerate(zip(self.dis_proj, self.sigma_proj)):
            sigma = sigma_layer(x).permute(0, 2, 1)

            sigma = torch.sigmoid(sigma * 5) + 1e-5
            sigma = torch.pow(3, sigma) - 1
            sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, self.seq_len)
            prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
            prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
            
            priors.append(prior)


        # hids = []
        # for hid in hidden_states:
        #     temp = reduce(hid, 'b l (d q)-> b l q', 'mean', q=1)
        #     hids.append(rearrange(temp, '(b m) l d -> b l (m d)', b=B))
        
        return outputs, attns, priors, None

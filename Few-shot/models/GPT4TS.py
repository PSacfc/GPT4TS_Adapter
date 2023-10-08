from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim

from transformers import GPT2ForSequenceClassification
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from models.hugging_gpt2.GPT2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
# from layers.Embed import DataEmbedding, DataEmbedding_wo_time
# from peft import get_peft_model, LoraConfig, TaskType
import copy

class Adapter(nn.Module):
    def __init__(self, in_feat, hid_dim, skip=True):
        super().__init__()
        self.D_fc1 = nn.Linear(in_feat, hid_dim)
        self.D_fc2 = nn.Linear(hid_dim, in_feat)
        self.act = nn.GELU()
        self.skip = skip
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        if self.skip:
            return x + self.drop(self.D_fc2(self.act(self.D_fc1(x))))
        else:
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
        # self.mlp = SpectFFN(in_feat, low_rank)
    
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
        # for i, block in enumerate(self.blocks):
        #     x = block(x)
        #     res_list.append(x)
        for i, block in enumerate(self.blocks):
            res_list.append(block(x))
        
        return res_list


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.pred_len = configs.pred_len
        self.d_ff = configs.d_ff
        self.gpt_layers = configs.gpt_layers
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        if self.stride > 1 or self.patch_size > 1:
            self.patch_num += 1
        
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model    
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        for i in range(configs.gpt_layers):
            # self.gpt2.h[i].scale = configs.scale
            self.gpt2.h[i].scale = configs.scale
            self.gpt2.h[i].attn.scale = configs.scale
            if configs.T_type == 1:
                self.gpt2.h[i].T_adapter = Adapter(configs.d_model, configs.adapter_dim, skip=False)
                self.gpt2.h[i].T_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.patch_num, 1))
            if configs.C_type == 1:
                self.gpt2.h[i].C_adapter = Adapter(configs.d_model, configs.adapter_dim, skip=False)
                self.gpt2.h[i].C_num = configs.enc_in
                self.gpt2.h[i].C_adapter_gate = torch.nn.Parameter(torch.zeros(1, configs.enc_in, 1))

        self.fft_adapter = FFT_adapter(configs.spect_adapter_layer, configs.enc_in, self.patch_num)
        self.adapter_in_layer = nn.ModuleList([nn.Linear(configs.patch_len, configs.d_model) for i in range(configs.adapter_layer)])
        self.in_layer = nn.Linear(configs.patch_len, configs.d_model)

        self.proj_layer = nn.Linear(configs.d_model, self.d_ff)
        self.out_layer = nn.Linear(self.d_ff * self.patch_num, configs.pred_len)
    
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'adapter' in name:
                param.requires_grad = True
            elif 'ln' in name:
                param.requires_grad = True
            elif 'wpe' in name:
                param.requires_grad = False
            else:
                param.requires_grad = False

        params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        print("number of self.gpt2 params: {}".format(params))
        params = sum(p.numel() for p in self.in_layer.parameters() if p.requires_grad)
        print("number of self.in_layer params: {}".format(params))
        params = sum(p.numel() for p in self.out_layer.parameters() if p.requires_grad)
        print("number of self.out_layer params: {}".format(params))
        params = sum(p.numel() for p in self.fft_adapter.parameters() if p.requires_grad)
        print("number of self.fft_adapter params: {}".format(params))


    def forward(self, x, *args, **kwargs):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        fft_adapter_list = self.fft_adapter(x) # B M N P
        adapters = []
        for i in range(self.gpt_layers - len(fft_adapter_list)):
            adapters.append(None)
        for i in range(len(fft_adapter_list)):
            fft_adapter_list[i] = self.adapter_in_layer[i](fft_adapter_list[i])
            fft_adapter_list[i] = rearrange(fft_adapter_list[i], 'b m n p -> (b m) n p')
            adapters.append(fft_adapter_list[i])

        x = rearrange(x, 'b m n p -> (b m) n p')
        outputs = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=outputs, adapters=adapters).last_hidden_state

        outputs = self.proj_layer(outputs)
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs

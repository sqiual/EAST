import math
from math import sqrt
#from train_flowpred import get_k_nearest
import torch.nn.functional as F
import torch
import torch.nn as nn
import pickle
from rtree import index
from collections import defaultdict
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaTokenizer, GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer, VivitModel, TimesformerModel, TimesformerConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers.models.timesformer.modeling_timesformer import TimesformerModel
#from layers.Embed import PatchEmbedding
import transformers
#from layers.StandardNorm import Normalize
import numpy as np
import torch.nn.functional as F
import re
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from torch.nn.functional import unfold
import random


transformers.logging.set_verbosity_error()
# ST encoding: ViViT or TimeSformer


class FusionGate(nn.Module):
    def __init__(self, embed_dim):
        super(FusionGate, self).__init__()
        self.linear_t = nn.Linear(embed_dim, embed_dim)
        self.linear_s = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, x_s):
        """
        x_t: Tensor of shape [B, T, D] (temporal embedding)
        x_s: Tensor of shape [B, T, D] (spatio embedding)
        Returns: fused tensor of shape [B, T, D]
        """
        h_t = self.linear_t(x_t)
        h_s = self.linear_s(x_s)
        gate = self.sigmoid(self.gate(h_t + h_s))
        fused = gate * x_t + (1 - gate) * x_s
        return fused

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
        
def contrastive_loss(z1, z2, temperature=0.07):
    """
    z1, z2: [B, T, d_proj] → contrastive over B*T pairs
    """
    B, T, D = z1.shape
    z1 = z1.reshape(B*T, D)
    z2 = z2.reshape(B*T, D)

    # Normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.matmul(z1, z2.T) / temperature  # [B*T, B*T]
    labels = torch.arange(B*T, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss
    
class Chomp1d_anti(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d_anti, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x): 
        return x[:, :, self.chomp_size:].contiguous()    #anti_tcn
        #return x[:, :, :-self.chomp_size].contiguous()  #org_tcn

class Chomp1d_casual(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d_casual, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x): 
        #return x[:, :, self.chomp_size:].contiguous()    #anti_tcn
        return x[:, :, :-self.chomp_size].contiguous()  #org_tcn

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, padding_casual, dropout=0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv12 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                stride=stride, padding=padding_casual, dilation=dilation)
        self.chomp1 = Chomp1d_anti(padding)
        self.chomp12 = Chomp1d_casual(padding_casual)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv22 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                stride=stride, padding=padding_casual, dilation=dilation)
        self.chomp2 = Chomp1d_anti(padding)
        self.chomp22 = Chomp1d_casual(padding_casual)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net_anti = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                      self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net_casual = nn.Sequential(self.conv12, self.chomp12, self.relu1, self.dropout1,
                                        self.conv22, self.chomp22, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # Fusion gate: learn to fuse anti and causal outputs
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(n_outputs * 2, 1, kernel_size=1),  # Gate: [B, 1, T]
            nn.Sigmoid()
        )

        self.relu = nn.ELU()
        self.init_weights()
        self.n_inputs = n_outputs

    def init_weights(self):
        for m in [self.conv1, self.conv2, self.conv12, self.conv22]:
            m.weight.data.normal_(0, 0.001)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.001)

    def forward(self, x):  # x: [B, C, T]
        out_anti = self.net_anti(x)   # [B, C, T]
        out_casual = self.net_casual(x)

        # Concatenate along channel axis and compute fusion weights
        combined = torch.cat([out_anti, out_casual], dim=1)  # [B, 2C, T]
        gate = self.fusion_gate(combined)                    # [B, 1, T]

        out = gate * out_anti + (1 - gate) * out_casual      # [B, C, T]

        res = x if self.downsample is None else self.downsample(x)
        out = out + res

        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, padding_casual=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.padding_casual=(kernel_size-1) * dilation_size
        self.padding=(kernel_size-1) * dilation_size
        self.dilation = dilation_size
        self.network = nn.Sequential(*layers)
        self.kernel = kernel_size
    def forward(self, x):
        return self.network(x)
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, d_model)  # optional: project to same size

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, input_dim]
        """
        x = self.input_proj(x)                 # [B, T, d_model]
        x = self.pos_encoder(x)                # [B, T, d_model]
        x = self.transformer_encoder(x)        # [B, T, d_model]
        # Option 1: mean pooling
        #out = x.mean(dim=1)                    # [B, d_model]
        # Option 2: last token
        #out = x[:, -1, :]                    # [B, d_model]
        out = x                               # [B, T, d_model]
        return self.output_proj(out)           # [B, T, d_model]


class OverlappingPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=4, stride=1, padding=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)

    def forward(self, x):
        # x: [B*T, C, H, W]
        x = self.proj(x)                         # [B*T, D, H', W']
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)         # [B*T, N_patches, D]
        return x, (H_p, W_p)


class TimeSformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.spatial_attn = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attn = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x, B, T, P):
        # x: [B*T, P, D]
        # Step 1: Spatial Attention (frame-by-frame)
        x_spatial = x.view(B * T, P, -1)                         # [B*T, P, D]
        x_spatial = self.spatial_attn(x_spatial)                # [B*T, P, D]

        # Step 2: Temporal Attention (patch-by-patch across time)
        x_temporal = x_spatial.view(B, T, P, -1).transpose(1, 2)  # [B, P, T, D]
        x_temporal = x_temporal.reshape(B * P, T, -1)             # [B*P, T, D]
        x_temporal = self.temporal_attn(x_temporal)              # [B*P, T, D]

        # Restore shape: [B, P, T, D] → [B, T, P, D] → [B*T, P, D]
        x_temporal = x_temporal.view(B, P, T, -1).transpose(1, 2).reshape(B * T, P, -1)
        return x_temporal


class TimeSformerEncoder(nn.Module):
    def __init__(
        self,
        img_size=[8,8],
        patch_size=4,
        stride=1,
        padding=2,
        in_channels=3,
        embed_dim=768,
        depth=3,
        num_heads=2,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = OverlappingPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride=stride,
            padding=padding
        )

        # Calculate number of patches per frame
        dummy = torch.zeros(1, in_channels, img_size[0], img_size[1])
        patch_tokens, (H_p, W_p) = self.patch_embed(dummy)
        self.num_patches = H_p * W_p
        self.H_p, self.W_p = H_p, W_p

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Shared for all time steps

        self.blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)                             # [B*T, C, H, W]
        patch_tokens, _ = self.patch_embed(x)                 # [B*T, P, D]

        # Add spatial positional encoding
        patch_tokens = patch_tokens + self.pos_embed_spatial[:, :patch_tokens.size(1), :]  # [B*T, P, D]

        # Apply Spatio-Temporal Transformer Blocks
        for blk in self.blocks:
            patch_tokens = blk(patch_tokens, B, T, self.num_patches)  # [B*T, P, D]

        # Final reshape: [B, T, P, D]
        x = patch_tokens.view(B, T, self.num_patches, -1)
        x = self.norm(x)
        return x  # [B, T, P, D]
    
    def preprocess(self, x_enc) -> torch.Tensor:
        """
        Convert raw traffic data to 3-channel input.
        
        Args:
            x_enc (B,224,224,12,12)
            density: Traffic density [0,1], shape [B,T,H,W]
            angle_deg: Direction angle [0,360], shape [B,T,H,W]
        Returns:
            input_tensor: [B,T,3,H,W]
        """
        density = x_enc[:,:,:,:,2].permute(0,3,1,2) # [B,224,224,12] -> [B,12,224,224]
        angle_deg = x_enc[:,:,:,:,3].permute(0,3,1,2)
        
        angle_rad = torch.deg2rad(angle_deg)
        valid_mask = ~torch.isnan(angle_rad)
        
        sin_angle = torch.zeros_like(angle_rad)
        cos_angle = torch.zeros_like(angle_rad)
        
        sin_angle[valid_mask] = torch.sin(angle_rad[valid_mask])
        cos_angle[valid_mask] = torch.cos(angle_rad[valid_mask])
        density = torch.where(valid_mask, density, torch.zeros_like(density))

        out = torch.stack([density, sin_angle, cos_angle], dim=2)
        #out = torch.stack([density, density, density], dim=2)
        return out # [B,T,3,H,W]
        
            
class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.model = configs.model
        self.dataset = configs.data
        self.data_short = configs.data_short
        self.flag = configs.flag
        self.root_path = configs.root_path
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_llm = configs.llm_dim
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.gpt_layers = 1
        self.d_model = configs.d_model
        self.img_size = configs.img_size
        self.btcn = TemporalConvNet(num_inputs = 3, num_channels = [self.d_llm,self.d_llm,self.d_llm,self.d_llm,self.d_llm]) 
        #self.btcn =SequencePatternEncoder(input_dim = 1, hidden_dim = self.d_llm, num_layers = 6)
        
        # self.transformer = TransformerEncoder(
        #     input_dim=1,
        #     d_model=self.d_llm,
        #     nhead=4,
        #     num_layers=2
        # ).to(configs.device)
        
        self.timesformer = TimeSformerEncoder(
            img_size=self.img_size,
            patch_size=3,
            stride=1,
            padding=1,
            in_channels=self.enc_in,
            embed_dim=self.d_llm,
            depth=4,
            num_heads=8
        ).to(configs.device)

        self.reprogramming_layer = ReprogrammingLayer(d_model = self.d_llm, n_heads = 1, d_keys = self.d_llm, d_llm = self.d_llm)
        
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )

                            
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2LMHeadModel.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2LMHeadModel.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )


        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        # self.lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=16,
        #     target_modules=["c_attn", "q_proj", "v_proj"],  # depends on the model
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type=TaskType.CAUSAL_LM
        # )

        #self.llm_model = get_peft_model(self.llm_model, self.lora_config)
        
        with open('/root/Time-LLM_OURS/OURS/dataset/prompt_bank/{0}.txt'.format(self.dataset), 'r') as f:
            self.description = f.read()   
        
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
        for param in self.llm_model.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(configs.dropout)
        self.fusion_gate = FusionGate(embed_dim=self.d_llm)  # or whatever your D is

        # for param in self.llm_model.parameters():
        #     param.requires_grad = False
        
        self.x = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm),
            nn.GELU(),
            #nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, self.d_llm),
            nn.GELU(),
            nn.Dropout(0.1)  # Regularization
        )
                
        self.x2 = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm), 
            nn.GELU(),
            #nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, self.d_llm),
            nn.Dropout(0.1)  # Regularization
        )
        self.x3 = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm),  
            nn.GELU(),
            #nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, self.d_llm),
            nn.Dropout(0.1)  # Regularization
        )     
        
        # self.final_proj = nn.Sequential(
        #     nn.Linear(self.d_llm, self.d_llm//2), 
        #     nn.ReLU(),
        #     nn.Linear(self.d_llm//2, 1), 
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        # )
        self.final_proj = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm//2), 
            nn.ReLU(),
            nn.Linear(self.d_llm//2, self.d_llm//10), 
            nn.ReLU(),
            nn.Linear(self.d_llm//10, 1), 
            nn.ReLU(),
        )
        # self.final_proj = nn.Sequential(
        #     nn.Linear(self.d_llm, 1), 
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        # )
        # self.timestep_embedding = nn.Embedding(48, self.d_llm)
        # self.day_embedding = nn.Embedding(366, self.d_llm)
        # self.month_embedding = nn.Embedding(13, self.d_llm)

        
    def param_num(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

        
    def get_hidden_states(self, embeds):
        """Consistent hidden state extraction"""
        outputs = self.llm_model(
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states[-1]
        
    def forward(self, x_enc, y_enc, scheduled_prob=1.0, training=True):
        output = self.forecast(x_enc, y_enc, scheduled_prob, training)
        return output      

    def forecast(self, x_enc, y_enc, scheduled_prob=1.0, training=True):        
        # x_enc (B,224,224,24,12)
        B,H,W,T,f = x_enc.size()
        x_enc = x_enc.contiguous()
        y_enc = y_enc.contiguous()
        x_full = torch.cat([x_enc, y_enc], dim = -2)
        
        x_t = x_enc[:,:,:,:,2]
        x_t = x_t.reshape(-1, T, 1) #[batch_size, seq_len, input_dim]
        
        x_time = x_enc[:,:,:,:,[5,9,7]] #time 0-48, doy 366, month 13 #[2, 8, 8, 12, 3]
        x_time = x_time.reshape(-1,T,3)
        # x_time = self.timestep_embedding(x_enc[:,:,:,:,5].long())
        # x_doy = self.day_embedding(x_enc[:,:,:,:,9].long())
        # x_month = self.month_embedding(x_enc[:,:,:,:,7].long())
        # x_time = x_time + x_doy + x_month #[2, 8, 8, 12, 256, 3]
        x_time = x_time.reshape(-1,T,3)
        
        x_org = self.timesformer.preprocess(x_enc).to(x_enc.device) #[1, 12, 3, 8, 8]
        x_t_emb = x_org.permute(0,3,4,1,2).reshape(-1,T,3)
        x_t_emb = self.btcn(x_t_emb.permute(0,2,1)).permute(0,2,1) #[batch_size*img_size*img_size, T, d_model]
        x_t_emb = x_t_emb.reshape(-1,T,self.d_llm)
        # x_t_emb = x_t_emb+x_time
        x_st_emb = self.timesformer(x_org)
        x_st_emb = x_st_emb.permute(0,2,1,3).reshape(-1, self.pred_len, self.d_llm)
        c_x_t_emb = self.x2(x_t_emb)
        c_x_st_emb = self.x3(x_st_emb)
        
        x_org_full = self.timesformer.preprocess(x_full).to(x_enc.device) #[1, 12, 3, 8, 8]
        x_t_emb_full = x_org_full.permute(0,3,4,1,2).reshape(-1,T*2,3)
        x_t_emb_full = self.btcn(x_t_emb_full.permute(0,2,1)).permute(0,2,1) #[batch_size*img_size*img_size, T, d_model]
        x_t_emb_full = x_t_emb_full.reshape(-1,T*2,self.d_llm)
        x_st_emb_full = self.timesformer(x_org_full)
        x_st_emb_full = x_st_emb_full.permute(0,2,1,3).reshape(-1, self.pred_len*2, self.d_llm)
        c_x_t_emb_full = self.x2(x_t_emb_full)
        c_x_st_emb_full = self.x3(x_st_emb_full)
        
        # prompts = []
        # for b in range(x_time.shape[0]):
        #     prompt = (
        #         f"The input is a series of traffic flow measurements taken on month {int(x_time[b,0,-1])}, day {int(x_time[b,0,-2])}, starting from {int(x_time[b,0,0]//2):02d}:{(int(x_time[b,0,0])%2)*30:02d}. "
        #         f"Please forecast the traffic flow of the target region for the next {self.pred_len} half-hour intervals, "
        #         f"based on the observed pattern in the past data.\n"
        #     )
        #     prompts.append(prompt)
            
        # prompts = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # prompt_embeddings = self.llm_model.get_input_embeddings()(prompts.to(x_enc.device))   #[128, 52, 768]

        # use_teacher = training and (y_enc is not None) and (torch.rand(1).item() < scheduled_prob)

        # if use_teacher:
        #     c_loss = contrastive_loss(c_x_t_emb_full, c_x_st_emb_full)
        #     hid_out = self.llm_model(inputs_embeds=c_x_t_emb_full, output_hidden_states=True).hidden_states
        #     output = hid_out[-1][:, -self.pred_len:, :]


        # Autoregressive branch with improvements
        # else:

        c_loss = contrastive_loss(c_x_t_emb, c_x_st_emb)
        position_ids = torch.arange(T, dtype=torch.long, device=c_x_t_emb.device).unsqueeze(0).expand(B*H*W, -1)
        position_emb = self.llm_model.transformer.wpe(position_ids)
        fused_emb = self.fusion_gate(c_x_t_emb, c_x_st_emb)  # [B, T, D]
        input_with_pos = fused_emb + position_emb
        output = self.llm_model(inputs_embeds=input_with_pos)
        last_step = output.hidden_states[-1][:, -1:, :]

        generated = []
        context = input_with_pos.clone()
        for _ in range(self.pred_len):
            out = self.llm_model(inputs_embeds=context)
            next_token_emb = self.x(out.hidden_states[-1][:, -1:, :]) 
            generated.append(next_token_emb)
            context = torch.cat([context, next_token_emb], dim=1)
        outputs = torch.cat(generated, dim=1)
            
        # predictions = []
        # for t in range(self.pred_len):
        #     outputs = self.llm_model(inputs_embeds=c_x_t_emb, output_hidden_states=True)
        #     hidden = outputs.hidden_states[-1][:, -1:, :]
            
        #     # Scheduled sampling during training
        #     if training and (y_enc is not None) and (random.random() < scheduled_prob):
        #         hidden = c_x_t_emb_full[:, T+t:T+t+1, :]
            
        #     predictions.append(hidden)
        #     x_cur_emb = torch.cat([c_x_t_emb, hidden], dim=1)
        #     c_x_t_emb = x_cur_emb #.detach() if (t % 2 == 0) else x_cur_emb
        
        # output = torch.cat(predictions, dim=1)[:, -self.pred_len:, :]


        
        # outputs = self.llm_model(inputs_embeds=c_x_t_emb, output_hidden_states=True).hidden_states
        # hidden = outputs[-1][:, -1:, :]
        #hidden = outputs[-1][:, -self.pred_len:, :]
        #hidden = torch.cat((hidden,x_t_emb), dim = -1)

        outputs = self.final_proj(outputs)#.permute(0,2,1)
        outputs = outputs.reshape(B,H,W,self.pred_len, -1)
        return outputs, c_loss

 
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, llm_embedding, source_embedding, value_embedding):
        B, L1, _ = llm_embedding.shape #[1, 12, 50258]
        S, L2, _ = source_embedding.shape #[200, 16]
        H = self.n_heads
        Q = self.query_projection(llm_embedding).view(B, L1, H, -1)
        K = self.key_projection(source_embedding).view(S, L2, H, -1)
        V = self.value_projection(value_embedding).view(S, L2, H, -1)

        out = self.reprogramming(Q,K,V)

        out = out.reshape(B, L1, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,slhe->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,slhe->blhe", A, value_embedding)

        return reprogramming_embedding
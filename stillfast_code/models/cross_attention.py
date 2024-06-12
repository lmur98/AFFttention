import torch
from torch import nn
from detectron2.config import configurable
from torch.nn import functional as F

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn
from torch.nn.init import trunc_normal_
import math
import numpy as np


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)    

class CrossAttention_for_fusion(nn.Module):
    def __init__(self, 
                 channels_2d,
                 channels_3d,
                 inter_channels):
        super().__init__()
        self.channels_2d = channels_2d
        self.channels_3d = channels_3d
        self.inter_channels = channels_2d 
        self.max_height_2d = 128
        
        #self.mlp_2d = nn.Sequential(nn.Linear(channels_2d, channels_2d), nn.ReLU(), nn.Linear(channels_2d, channels_2d))
        #self.mlp_2d = nn.Sequential(nn.Conv2d(channels_2d, channels_2d // 2, kernel_size=3, stride=1, padding=1), nn.ReLU(), 
        #                            nn.Conv2d(channels_2d // 2, channels_2d, kernel_size=3, stride=1, padding=1))
        self.norm_2d = nn.LayerNorm(channels_2d)
        self.num_tokens_2d = 90 * 70 
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, self.num_tokens_2d, channels_2d))
        trunc_normal_(self.pos_embed_2d, std=0.02)
        
        #self.mlp_3d = nn.Sequential(nn.Linear(channels_3d, channels_2d), nn.ReLU(), nn.Linear(channels_2d, channels_2d))
        #self.mlp_3d = nn.Sequential(nn.Conv2d(channels_3d, channels_3d // 2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        #                            nn.Conv2d(channels_3d // 2, channels_2d, kernel_size=3, stride=1, padding=1))
        self.norm_3d = nn.LayerNorm(channels_2d)
        self.num_tokens_3d = 14 * 21 #+ 90 * 70
        self.pos_embed_3d = nn.Parameter(torch.zeros(1, self.num_tokens_3d, channels_2d))
        trunc_normal_(self.pos_embed_3d, std=0.02)
        
            
        # All 1x1 filters
        self.q = nn.Linear(self.channels_2d, self.inter_channels)
        self.k = nn.Linear(self.channels_3d, self.inter_channels)
        self.v = nn.Linear(self.channels_3d, self.inter_channels)
        
        self.post_fusion_sum = nn.Conv2d(self.channels_2d, self.channels_2d, kernel_size=3, stride=1, padding=1)
        
        #self.norm_residual_mlp = nn.LayerNorm(self.channels_2d)
        #self.mlp_block = Mlp_for_attention(self.channels_2d, self.channels_2d * 4, self.channels_2d, nn.GELU, 0.0)
     
    def interpolate_2D_pos_encoding(self, w, h):
        N = self.num_tokens_2d
        max_width = 90
        max_height = 70
        if w * h == N:
            return self.pos_embed_2d
        pos_embed = self.pos_embed_2d #.float()
        pos_embed = pos_embed.reshape(1, max_height, max_width, self.channels_2d).permute(0, 3, 1, 2)
        pos_embed = nn.functional.interpolate(pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        return pos_embed
    
    def interpolate_3D_pos_encoding(self, w, h):
        N = self.num_tokens_3d
        max_width = 90
        max_height = 70
        
        if w * h == N: #and w == h:
            return self.pos_embed_3d
        #We interpolate only the 2D part of the positional encoding, since it is the only one which changes
        pos_embed = self.pos_embed_3d #.float()
        pos_embed_2D = pos_embed[:, :max_width * max_height, :] #We have to interpolate this
        pos_embed_3D = pos_embed[:, max_width * max_height:, :] #This part is constant, it is always 14*21
        
        pos_embed_2D = pos_embed_2D.reshape(1, max_height, max_width, self.channels_2d).permute(0, 3, 1, 2)
        pos_embed_2D = nn.functional.interpolate(pos_embed_2D, size=(h, w), mode="bicubic")
        pos_embed_2D = pos_embed_2D.flatten(2).transpose(1, 2)
        
        return torch.cat((pos_embed_2D, pos_embed_3D), dim=1) #.to(previous_dtype)
     
    def interpolate_3D_pos_encoding_old(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] #- 1
        N = self.num_tokens_3d
        N_w = 77
        N_h = 58
        
        if npatch == N: #and w == h:
            return self.pos_embed_3d
        #We interpolate only the 2D part of the positional encoding, since it is the only one which changes
        pos_embed = self.pos_embed_3d #.float()
        patch_2D_pos_embed = pos_embed[:, 0: N_w * N_h] #We have to interpolate this
        patch_3D_pos_embed = pos_embed[:, N_w * N_h:] #This part is constant, it is always 14*21
        dim = x.shape[-1]
        w0 = w #We remove the 3d width offset
        h0 = h #We remove the 3d height offset
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        #patch_pos_embed = nn.functional.interpolate(
        #    patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        #    scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), mode="bicubic",)
        #patch_2D_pos_embed = nn.functional.interpolate(patch_2D_pos_embed.reshape(1, N_w, N_h, dim).permute(0, 3, 1, 2),
        #                                               scale_factor=(w0 / N_w, h0 / N_h), mode="bicubic",)
        #assert int(w0) == patch_2D_pos_embed.shape[-2] and int(h0) == patch_2D_pos_embed.shape[-1]
        #patch_2D_pos_embed = patch_2D_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        patch_2D_pos_embed = nn.functional.interpolate()
        print(patch_2D_pos_embed.shape, patch_3D_pos_embed.shape)
        return torch.cat((patch_2D_pos_embed, patch_3D_pos_embed), dim=1) #.to(previous_dtype)
        
    
    def forward(self, features_2d, features_3d):
        #2D features: add positional encoding and normalize
        BS, Ch_Dino, H_2D, W_2D = features_2d.shape
        x_pre_norm = features_2d.flatten(2).transpose(1, 2)  # (B, Dino_Ch, H, W) -> (B, H*W, Dino_Ch)
        #x_pre_norm = self.mlp_2d(x_pre_norm) #Adapt the dimensions of the 2D features to the 3D features
        x_pre_norm = x_pre_norm + self.interpolate_2D_pos_encoding(W_2D, H_2D) #2D positional encoding
        x = self.norm_2d(x_pre_norm) #Normalize the 2D features
        
        #2D-3D features: adapt dimensions, add other positional encoding and normalize
        #x_2D = self.mlp_2d(features_2d).flatten(2).transpose(1, 2) #Adapt the dimensions of the 2D features to the 3D features
        y_pre_norm = features_3d.flatten(2).transpose(1, 2)
        #y_pre_norm = self.mlp_3d(features_3d)  # (B, EgoVLP_Ch, H, W) -> (B, H*W, EgoVLP_Ch)
        #y_pre_norm = torch.cat((x_2D, y_3D), dim=1) #Concatenate the 2D and 3D features
        y_pre_norm = y_pre_norm + self.pos_embed_3d #3D positional encoding #2D positional encoding #3D positional encoding
        y = self.norm_3d(y_pre_norm) #Normalize the concatenated features
        
        #Cross-Attention
        Q = self.q(x) 
        K = self.k(y)
        V = self.v(y)
        attn = torch.matmul(Q, K.transpose(-1, -2)) * (self.inter_channels ** (-0.5)) # 1/sqrt(NC) is the scaling factor
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out_2D = out.reshape(BS, H_2D, W_2D, Ch_Dino).permute(0, 3, 1, 2)
        x_fused = self.post_fusion_sum(out_2D + features_2d)
        return x_fused
    
class Dual_CrossAttention(nn.Module):
    def __init__(self, 
                 channels_2d,
                 channels_3d):
        super().__init__()
        self.channels_2d = channels_2d
        self.channels_3d = channels_3d
        self.inter_channels = channels_2d 
        self.num_tokens_2d = 90 * 60
        self.num_tokens_3d = 14 * 21
        
        self.num_heads = 12
        head_dim = self.inter_channels // self.num_heads
        self.scale = head_dim**-0.5

        
        #2D as query
        self.norm_2d_A = nn.LayerNorm(channels_2d)
        self.q_A = nn.Linear(channels_2d, channels_2d, bias = False)
        self.pos_embed_2d_A = nn.Parameter(torch.zeros(1, 1 + 90 * 70, channels_2d)) #MOD for class token
        trunc_normal_(self.pos_embed_2d_A, std=0.02)
        
        #3D are then the keys and values
        #self.adapt_3d_A = nn.Linear(channels_3d, channels_2d)
        self.norm_3d_A = nn.LayerNorm(channels_3d)
        self.k_A = nn.Linear(channels_3d, channels_2d, bias = False)
        self.v_A = nn.Linear(channels_3d, channels_2d, bias = False)
        self.pos_embed_3d_A = nn.Parameter(torch.zeros(1, 1 + 14 * 21, channels_3d)) #MOD for class token
        trunc_normal_(self.pos_embed_3d_A, std=0.02)
        self.mlp_A = nn.Sequential(nn.Linear(channels_2d, int(channels_2d * 4)), nn.GELU(), nn.Linear(int(channels_2d * 4), channels_2d))
        self.norm_A = nn.LayerNorm(channels_2d)
        
        #3D as query        
        self.norm_3d_B = nn.LayerNorm(channels_3d)
        self.q_B = nn.Linear(channels_3d, channels_3d, bias = False)
        self.pos_embed_3d_B = nn.Parameter(torch.zeros(1, 1 + 14 * 21, channels_3d)) #MOD for class token
        trunc_normal_(self.pos_embed_3d_B, std=0.02)
        
        #2D are then the keys and values
        #self.adapt_2d_B = nn.Linear(channels_2d, channels_3d)
        self.norm_2d_B = nn.LayerNorm(channels_2d)
        self.k_B = nn.Linear(channels_2d, channels_3d, bias = False)
        self.v_B = nn.Linear(channels_2d, channels_3d, bias = False)
        self.pos_embed_2d_B = nn.Parameter(torch.zeros(1, 1 + 90 * 70, channels_2d)) #MOD for class token
        trunc_normal_(self.pos_embed_2d_B, std=0.02)
        self.mlp_B = nn.Sequential(nn.Linear(channels_3d, int(channels_3d * 4)), nn.GELU(), nn.Linear(int(channels_3d * 4), channels_3d))
        self.norm_B = nn.LayerNorm(channels_3d)
        
        self.reduce_3d = nn.Conv2d(channels_3d, channels_2d, kernel_size=3, stride=1, padding=1)
        #self.reduce_3d_class = nn.Linear(channels_3d, channels_2d, bias = False)
    
    def interpolate_2D_pos_encoding(self, w, h, pos_embed):
        N = self.num_tokens_2d
        max_width = 90
        max_height = 70
        if w * h == N:
            return pos_embed
        #pos_embed = self.pos_embed_2d #.float()
        patch_pos_embed = pos_embed[:, 1:, :] #We have to interpolate this #MOD
        class_pos_embed = pos_embed[:, 0, :] #This part is constant, it is always 1 #MOD
        patch_pos_embed = patch_pos_embed.reshape(1, max_height, max_width, self.channels_2d).permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(h, w), mode="bicubic")
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        pos_embed = torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1) #MOD
        return pos_embed
             
    def cross_attn(self, Q, K, V):
        #Cross-Attention
        BS, N_tokens, NC = Q.shape
        #print(Q.shape, K.shape, V.shape)
        attn = torch.matmul(Q, K.transpose(-1, -2)) * (NC ** (-0.5)) # 1/sqrt(NC) is the scaling factor
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn
        #return out
        
    def multi_head_cross_attn(self, Q, K, V):
        BS, N_Q_tokens, NC = Q.shape
        B, N_K_tokens, NC = K.shape
        q = Q.reshape(BS, N_Q_tokens, self.num_heads, NC // self.num_heads).permute(0, 2, 1, 3) * self.scale
        k = K.reshape(BS, N_K_tokens, self.num_heads, NC // self.num_heads).permute(0, 2, 1, 3)
        v = V.reshape(BS, N_K_tokens, self.num_heads, NC // self.num_heads).permute(0, 2, 1, 3)
        
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N_Q_tokens, NC)
        return x, attn
    
    def forward(self, features_2d, features_3d, class_2d, class_3d):
        #2D features: add positional encoding and normalize
        BS, Ch_Dino, H_2D, W_2D = features_2d.shape
        features_2d = features_2d.flatten(2).transpose(1, 2)
        features_2d = torch.cat((class_2d, features_2d), dim=1) #MOD
        pos_embed_2d_A = self.interpolate_2D_pos_encoding(W_2D, H_2D, self.pos_embed_2d_A)
        pos_embed_2d_B = self.interpolate_2D_pos_encoding(W_2D, H_2D, self.pos_embed_2d_B)
        
        BS, Ch_EgoVLP, H_3D, W_3D = features_3d.shape
        features_3d = features_3d.flatten(2).transpose(1, 2)
        features_3d = torch.cat((class_3d, features_3d), dim=1) #MOD
        
        #Query: 2D, Keys and Values: 3D
        x_2D_A = self.norm_2d_A(features_2d + pos_embed_2d_A) #Query
        y_3D_A = self.norm_3d_A(features_3d + self.pos_embed_3d_A) #Keys and Values
        
        Q_A = self.q_A(x_2D_A) 
        K_A = self.k_A(y_3D_A)
        V_A = self.v_A(y_3D_A)
        y_A, attn_A = self.multi_head_cross_attn(Q_A, K_A, V_A) #Output shape = queries
        #y_A, attn_A = self.cross_attn(Q_A, K_A, V_A)
        features_2d = features_2d + y_A
        features_2d = features_2d + self.mlp_A(self.norm_A(features_2d))
        
        #Query: 3D, Keys and Values: 2D
        x_2D_B = self.norm_2d_B(features_2d + pos_embed_2d_B) #Keys and Values
        y_3D_B = self.norm_3d_B(features_3d + self.pos_embed_3d_B) #Now it is the queries
        Q_B = self.q_B(y_3D_B)
        K_B = self.k_B(x_2D_B)
        V_B = self.v_B(x_2D_B)
        y_B, attn_B = self.multi_head_cross_attn(Q_B, K_B, V_B)
        #y_B, attn_B = self.cross_attn(Q_B, K_B, V_B)
        features_3d = features_3d + y_B
        features_3d = features_3d + self.mlp_B(self.norm_B(features_3d))
        
        class_2d = features_2d[:, 0, :] #.unsqueeze(1) #MOD
        class_3d = features_3d[:, 0, :] #.unsqueeze(1) #MOD
        features_2d = features_2d[:, 1:, :].reshape(BS, H_2D, W_2D, Ch_Dino).permute(0, 3, 1, 2)
        features_3d = features_3d[:, 1:, :].reshape(BS, H_3D, W_3D, Ch_EgoVLP).permute(0, 3, 1, 2)
        features_3d = self.reduce_3d(features_3d)
        #class_3d = self.reduce_3d_class(class_3d) #MOD
        return features_2d, features_3d, class_2d, class_3d, None, None #attn_A, attn_B
        
        
class Dual_CrossAttention_Conv(nn.Module):
    def __init__(self, 
                 channels_2d,
                 channels_3d):
        super().__init__()
        self.channels_2d = channels_2d
        self.channels_3d = channels_3d
        self.inter_channels_A = channels_2d // 2
        self.inter_channels_B = channels_3d // 2
        
        #2D as query
        self.q_A = nn.Conv2d(self.channels_2d, self.inter_channels_A, 1)
        self.k_A = nn.Conv2d(self.channels_3d, self.inter_channels_A, 1)
        self.v_A = nn.Conv2d(self.channels_3d, self.inter_channels_A, 1)
        self.proj_A = nn.Conv2d(self.inter_channels_A, self.channels_2d, 1) #Initialize to the identity function
        nn.init.constant_(self.proj_A.weight, 0)
        nn.init.constant_(self.proj_A.bias, 0)
        
        #3D as query        
        self.q_B = nn.Conv2d(self.channels_3d, self.inter_channels_B, 1)
        self.k_B = nn.Conv2d(self.channels_2d, self.inter_channels_B, 1)
        self.v_B = nn.Conv2d(self.channels_2d, self.inter_channels_B, 1)
        self.proj_B = nn.Conv2d(self.inter_channels_B, self.channels_3d, 1) #Initialize to the identity function
        nn.init.constant_(self.proj_B.weight, 0)
        nn.init.constant_(self.proj_B.bias, 0)
        
    def cross_attn(self, Q, K, V):
        BS, NC, H, W = Q.shape
        Q = Q.view(BS, NC, -1).permute(0,2,1) #concat H and W and re-arrange to have BS x HW x NC
        K = K.view(BS, NC, -1) 
        V = V.view(BS, NC, -1).permute(0,2,1) 
        
        #Cross-Attention
        attn = torch.matmul(Q, K) * (NC ** (-0.5)) # 1/sqrt(NC) is the scaling factor
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1).view(BS, NC, H, W)
        return out
    
    def forward(self, features_2d, features_3d):
        #Query: 2D, Keys and Values: 3D
        Q_A = self.q_A(features_2d) 
        K_A = self.k_A(features_3d)
        V_A = self.v_A(features_3d)
        
        cross_attn_A = self.cross_attn(Q_A, K_A, V_A) #Output shape = queries
        x_2D = features_2d + self.proj_A(cross_attn_A) #Output shape = queries
        
        #Query: 3D, Keys and Values: 2D
        Q_B = self.q_B(features_3d)
        K_B = self.k_B(features_2d)
        V_B = self.v_B(features_2d)
        cross_attn_B = self.cross_attn(Q_B, K_B, V_B)
        x_3D = features_3d + self.proj_B(cross_attn_B)
        
        return x_2D, x_3D
  
        
class NonLocalTemporalPooling_linear(nn.Module):
    def __init__(self, num_channels, inter_channels, max_height_before_pooling):
        super().__init__()
        self.num_channels = num_channels
        self.inter_channels = inter_channels
        
        self.norm_2D = nn.LayerNorm(num_channels)
        self.norm_3D = nn.LayerNorm(num_channels)
        self.final_norm = nn.LayerNorm(num_channels)
        
        self.pos_enc_2D = nn.Parameter(torch.zeros(1, 1 + 14 * 21, num_channels)) #MOD for class token
        trunc_normal_(self.pos_enc_2D, std=0.02)
        self.pos_enc_3D = nn.Parameter(torch.zeros(1, 1 + 16 * 14 * 21, num_channels)) #MOD for class token
        trunc_normal_(self.pos_enc_3D, std=0.02)
        
        self.q = nn.Linear(num_channels, num_channels, bias=False)
        self.k = nn.Linear(num_channels, num_channels, bias=False)
        self.v = nn.Linear(num_channels, num_channels, bias=False)
        
        self.mlp = nn.Sequential(nn.Linear(num_channels, int(num_channels * 4)), nn.GELU(), nn.Linear(int(num_channels * 4), num_channels))
 
        
    def cross_attn(self, Q, K, V):
        #Cross-Attention
        BS, N_tokens, NC = Q.shape
        attn = torch.matmul(Q, K.transpose(-1, -2)) * (NC ** (-0.5)) # 1/sqrt(NC) is the scaling factor
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn
        
    def forward(self, class_t, x_t):
        BS, _, T, H, W = x_t.shape
        
        last_frame = x_t[:,:,-1,:,:]
        x_last_t = last_frame.flatten(2).transpose(1, 2)
        x_last_t = torch.cat((class_t, x_last_t), dim=1) #MOD
        x_last_t_norm = self.norm_2D(x_last_t + self.pos_enc_2D)
        
        x_all_t = x_t.flatten(2).transpose(1, 2)
        x_all_t = torch.cat((class_t, x_all_t), dim=1) #MOD
        x_all_t_norm = self.norm_3D(x_all_t + self.pos_enc_3D)  
        
        #Cross-Attention
        Q = self.q(x_last_t_norm)
        K = self.k(x_all_t_norm)
        V = self.v(x_all_t_norm)
        y, attn = self.cross_attn(Q, K, V)
        x_last_t = x_last_t + y
        x_last_t = x_last_t + self.mlp(self.final_norm(x_last_t))
        class_t = x_last_t[:,0,:].unsqueeze(1)
        patch_t = x_last_t[:,1:,:].reshape(BS, H, W, self.num_channels).permute(0, 3, 1, 2)
        return class_t, patch_t, attn
    

class NonLocalTemporalPooling(nn.Module):
    def __init__(self, num_channels, inter_channels, max_height_before_pooling):
        super().__init__()
        self.inter_channels = inter_channels
        self.num_channels = num_channels
        self.max_height = max_height_before_pooling
            
        # All 1x1 filters
        self.q = nn.Conv2d(self.num_channels, self.inter_channels, 1)
        self.k = nn.Conv3d(self.num_channels, self.inter_channels, 1)
        self.v = nn.Conv3d(self.num_channels, self.inter_channels, 1)
        
        # Initialize to zero so that the module implements an identity function at initialization
        self.out_conv = nn.Conv2d(self.inter_channels, num_channels, 1)
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)
        
        self.num_heads = 1
        head_dim = 768 // self.num_heads
        self.scale = head_dim**-0.5
        
    def forward(self, x):
        BS, _, T, H, W = x.shape
        NC = self.inter_channels
        last_frame = x[:,:,-1,:,:]

        Q = self.q(last_frame) 
        if H>self.max_height:
            k = math.ceil(H/self.max_height)
            x_pool = F.max_pool3d(x, kernel_size=(1, k, k))
        else:
            x_pool = x
        
        K = self.k(x_pool)
        V = self.v(x_pool)
        
        
        """
        #concat H and W and re-arrange to have BS x HW x NC
        Q = Q.view(BS, NC, -1).permute(0,2,1)
        #concat H and W and T 
        K = K.view(BS, NC, -1)
        #concat H and W and T and re-arrange to have BS x HWT x NC
        V = V.view(BS, NC, -1).permute(0,2,1)
        print(Q.shape, K.shape, V.shape, 'for the multiplication of the atteniton')
        att = torch.matmul(Q, K) * (NC ** (-0.5))
        att = F.softmax(att, dim=-1)
        
        out = torch.matmul(att, V) #BS x HW x NC
        out = out.permute(0, 2, 1).view(BS, NC, H, W) #rearrange and reshape to obtain BS x NC x H x W
        """
        Q = Q.view(BS, NC, -1)
        N_Q_tokens = Q.shape[2]
        q = Q.reshape(BS, N_Q_tokens, self.num_heads, NC // self.num_heads).permute(0, 2, 1, 3) * self.scale
        
        K = K.view(BS, NC, -1)
        N_K_tokens = K.shape[2]
        k = K.reshape(BS, N_K_tokens, self.num_heads, NC // self.num_heads).permute(0, 2, 1, 3)
        
        V = V.view(BS, NC, -1)
        N_V_tokens = V.shape[2]
        v = V.reshape(BS, N_V_tokens, self.num_heads, NC // self.num_heads).permute(0, 2, 1, 3)
        
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(BS, N_Q_tokens, NC)
        out = out.permute(0, 2, 1).view(BS, NC, H, W)
        
        # residual connection and normalization
        x = last_frame + self.out_conv(out)
        #x = x.view(BS, self.num_channels, -1)
        #x = self.norm(x)
        #x = x.view(BS, self.num_channels, H, W)
        return x
               
        

if __name__ == '__main__':
    #pth = '/home/lmur/stillfast_baseline/stillfast/output/sta/StillFast_Dino2D_EgoVLP_temp_and_fusion_MULTIRES/version_5/checkpoints/epoch=21-step=0101926-map_box_noun_verb_ttc=2.9492.ckpt'
    #checkpoint = torch.load(pth, map_location='cpu')
    #for name, param in checkpoint['state_dict'].items():
    #    print(name, param.shape)
    #model = Dual_CrossAttention(channels_2d = 384, channels_3d = 768)    
    model = NonLocalTemporalPooling(num_channels = 768, inter_channels = 768, max_height_before_pooling = 128)
    # Cálculo del número de parámetros por cada subcapa
    for name, module in model.named_modules():
        total_params = sum(p.numel() for p in module.parameters())
        print(f"Capa: {name}, Número de parámetros: {total_params / 1e6}M")
    x_2D = torch.rand(2, 384, 62, 94)
    x_3D = torch.rand(2, 768, 14, 21)
    class_2d = torch.rand(2, 1, 384)
    class_3d = torch.rand(2, 1, 768)
    
    x_t = torch.rand(2, 768, 16, 14, 21)
    class_t = torch.rand(2, 1, 768)

    #x_2D, x_3D, class_2d, class_3d, _, _ = model(x_2D, x_3D, class_2d, class_3d)
    class_t, x_t = model(x_t)
    print(class_t.shape, x_t.shape, '***************')
    
import functools
from imaplib import Int2AP
import torch
from torch import nn
from detectron2.config import configurable
from torch.nn import functional as F
from typing import List
#from stillfast.models.backbone_utils_2d import build_clean_2d_backbone, build_still_backbone
#from stillfast.models.backbone_utils_3d import build_clean_3d_backbone
from functools import partial
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork
from stillfast.models.backbone_utils_3d import FeaturePyramidNetwork3D, LastLevelMaxPool3D
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from stillfast.ops.misc import Conv2dNormActivation
from typing import Union
import math
import time
from stillfast.models.x3d import create_x3d_res_stage
from stillfast.models.timesformer_block import Space_Time_Attention
from stillfast.models.locate_model import Locate_Aff, MLP_Dino_single
from stillfast.models.cross_attention import CrossAttention_for_fusion, Dual_CrossAttention_Conv, Dual_CrossAttention
import time

import argparse
import sys 
sys.path.append('/home/lmur/EgoVLPv2/EgoVLPv2/')
from parse_config import ConfigParser
import pdb
#import model.model as module_arch #From the EgoVLPv2 repo 
from model.video_transformer import SpaceTimeTransformer
from torch.nn.init import trunc_normal_

from stillfast.models.swim_transformer import build_swin_transformer

def inflate_positional_embeds(state_dict, load_temporal_fix = 'bilinear'):
    # allow loading of timesformer with fewer num_frames
    curr_keys = list(state_dict.keys())

    if 'temporal_embed' in state_dict:
        load_temporal_embed = state_dict['temporal_embed']
        load_num_frames = load_temporal_embed.shape[1]
        curr_num_frames = 16 #self.video_params['num_frames']
        embed_dim = load_temporal_embed.shape[2]

        if load_num_frames != curr_num_frames:
            if load_num_frames > curr_num_frames:
                print('### loaded  model has MORE frames than current loading weights, filling in the extras via bilinear') #{self.video_params["model"]}
                new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
            else:
                print('### We fill the temporal embeding, with bilinear interpolation') #{self.video_params["model"]}
                if load_temporal_fix == 'zeros':
                    new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                    new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                elif load_temporal_fix in ['interp', 'bilinear']:
                    # interpolate
                    # unsqueeze so pytorch thinks its an image
                    mode = 'nearest'
                    if load_temporal_fix == 'bilinear':
                        mode = 'bilinear'
                    print('The shape of the loaded temporal embed is', load_temporal_embed.shape)
                    load_temporal_embed = load_temporal_embed.unsqueeze(0)
                    new_temporal_embed = F.interpolate(load_temporal_embed,
                                                       (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    print('The new temporal embed shape is', new_temporal_embed.shape)
                else:
                    raise NotImplementedError
            state_dict['temporal_embed'] = new_temporal_embed


        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
    if 'pos_embed' in state_dict:
        load_spatial_embed = state_dict['pos_embed']
        load_num_patches = load_spatial_embed.shape[1]
        curr_num_patches = (224 // 16) * (336 // 16) #self.video_params['num_patches'] + 1, the class token
        embed_dim = load_spatial_embed.shape[2]

        if load_num_patches != curr_num_patches:
            if load_num_patches > curr_num_patches:
                print('### loaded  model has MORE patches than current loading weights, filling in the extras via bilinear')
                new_spatial_embed = load_spatial_embed[:, :curr_num_patches, :]
            else:
                print('### We fill the spatial embeddings, with bilinear interpolation')
                class_embed = load_spatial_embed[:, 0, :].unsqueeze(1)
                patch_embed = load_spatial_embed[:, 1:, :]
                load_spatial_embed = patch_embed.unsqueeze(0)
                new_spatial_embed = F.interpolate(load_spatial_embed,
                                                 (curr_num_patches, embed_dim), mode='bilinear', align_corners=True).squeeze(0)
                new_spatial_embed = torch.cat((class_embed, new_spatial_embed), dim=1)

            state_dict['pos_embed'] = new_spatial_embed
    return state_dict
            

     
class NonLocalTemporalPooling_linear(nn.Module):
    def __init__(self, num_channels, inter_channels, max_height_before_pooling):
        super().__init__()
        self.num_channels = num_channels
        self.inter_channels = inter_channels
        
        self.norm_2D = nn.LayerNorm(num_channels)
        self.norm_3D = nn.LayerNorm(num_channels)
        self.final_norm = nn.LayerNorm(num_channels)
        
        self.pos_enc_2D = nn.Parameter(torch.zeros(1, 14 * 21, num_channels)) #MOD for class token
        trunc_normal_(self.pos_enc_2D, std=0.02)
        self.pos_enc_3D = nn.Parameter(torch.zeros(1, 16 * 14 * 21, num_channels)) #MOD for class token
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
        return out
        
    def forward(self, x_t):
        BS, _, T, H, W = x_t.shape
        
        last_frame = x_t[:,:,-1,:,:]
        x_last_t = last_frame.flatten(2).transpose(1, 2)
        x_last_t_norm = self.norm_2D(x_last_t + self.pos_enc_2D)
        
        x_all_t = x_t.flatten(2).transpose(1, 2)
        x_all_t_norm = self.norm_3D(x_all_t + self.pos_enc_3D)  
        
        #Cross-Attention
        Q = self.q(x_last_t_norm)
        K = self.k(x_all_t_norm)
        V = self.v(x_all_t_norm)
        x_last_t = x_last_t + self.cross_attn(Q, K, V)
        x_last_t = x_last_t + self.mlp(self.final_norm(x_last_t))
        patch_t = x_last_t.reshape(BS, H, W, self.num_channels).permute(0, 3, 1, 2)
        return patch_t       

class NonLocalTemporalPooling(nn.Module):
    def __init__(self, num_channels, inter_channels, max_height_before_pooling):
        super().__init__()
        if inter_channels is None or inter_channels == 'half':
            self.inter_channels = num_channels // 2
            
        if inter_channels==0:
            self.inter_channels = 1
            
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
        self.num_heads = 12 #It was 12 before
        head_dim = self.inter_channels // self.num_heads
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

        att = torch.matmul(Q, K) * (NC ** (-0.5))
        att = F.softmax(att, dim=-1)
        
        out = torch.matmul(att, V) #BS x HW x NC
        out = out.permute(0, 2, 1).view(BS, NC, H, W) #rearrange and reshape to obtain BS x NC x H x W
        """
        
        #Multi-head attention
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

class DINO_2D_feature_extractor(nn.Module):
    def __init__(self, embedding_size, patch_size, last_layers):
        super().__init__()
        print('We are loading DINO-v2, the 2D encoder!!, with an embedding_size', embedding_size)
        self.patch_size = patch_size
        self.resize_factor =  {"0": 14/4, "1": 14/8, "2": 14/16, "3": 14/32, "pool": 14/64}
        self.out_channels = embedding_size
        self.last_layers = last_layers

        if embedding_size == 384:
            self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif embedding_size == 768:
            self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif embedding_size == 1024:
            self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        else:
            raise ValueError(f"Embedding size {self.embedding_size} not supported")
        for name, parameter in self.dino_extractor.named_parameters():
            parameter.requires_grad_(False)

    def forward(self, x, return_attn = False):
        h_still, h_fast = x 
        with torch.no_grad():
            # ------------ Dino extraction ------------ #
            B, _, w_i, h_i = h_still.shape
            dino_output = self.dino_extractor.forward_features(h_still)
            x_patch = dino_output["x_norm_patchtokens"].detach()
            x_clss = dino_output["x_norm_clstoken"].detach().unsqueeze(1)
            #attn = dino_output["attn"].detach()
            
            # ------------ Reshape and create the pyramid ------------ #    
            x_2D = x_patch.reshape(B, w_i // self.patch_size, h_i // self.patch_size, self.out_channels).permute(0, 3, 1, 2)
            features_2d = OrderedDict([("0", x_2D), ("1", x_2D), ("2", x_2D), ("3", x_2D), ("pool", x_2D)])
            for key, value in features_2d.items():
                features_2d[key] = nn.functional.interpolate(value, scale_factor=self.resize_factor[key], mode="bilinear") # (bs, ch, w_featmap, h_featmap) With different resizings  
        if return_attn:
            return x_2D, x_patch
        else:
            return features_2d

class DINO_2D_with_FAST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print('****************We are loading DINO-v2, the 2D encoder!!')
        self.patch_size = 14
        self.resize_factor =  {"0": 14/4, "1": 14/8, "2": 14/16, "3": 14/32, "pool": 14/64}
        self.out_channels = 384
        self.last_layers = [11]
        self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for name, parameter in self.dino_extractor.named_parameters():
            parameter.requires_grad_(False)

        print('****************We are loading the FAST encoder!!')
        fast_backbone = torch.hub.load("facebookresearch/pytorchvideo", model="x3d_m", pretrained=True)
        channels_list = [24, 48, 96, 192]
        del fast_backbone.blocks[5]
        self.fast_backbone = fast_backbone
        self.fast_layers = range(1, 5) # List[int]
        self.fast_fpn = FeaturePyramidNetwork3D([24, 48, 96, 192], 192, extra_blocks=LastLevelMaxPool3D())
        
        print('****************We are loading the fusion blocks!!')
        fast_backbone_channels = [192, 192, 192, 192]
        still_backbone_channels = [384, 384, 384, 384]
        fusion_block = 'convolutional'
        fusion_block = partial(ConvolutionalFusionBlock_old, cfg.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK)
        for layer, c2d, c3d in zip(self.fast_layers, still_backbone_channels, fast_backbone_channels):
            setattr(self, f"pre_pyramid_fusion_block_{layer}", fusion_block(channels_2d=c2d, channels_3d=c3d))


    def forward(self, x):
        h_still, h_fast = x 
        
        with torch.no_grad():
            # ------------ Dino extraction ------------ #
            B, _, w_i, h_i = h_still.shape
            dino_output = self.dino_extractor.forward_features(h_still)
            x_patch = dino_output["x_norm_patchtokens"].detach()
            x_clss = dino_output["x_norm_clstoken"].detach().unsqueeze(1)

            # ------------ Reshape and create the pyramid ------------ #    
            x_2D = x_patch.reshape(B, w_i // self.patch_size, h_i // self.patch_size, self.out_channels).permute(0, 3, 1, 2)
            still_features = OrderedDict([("0", x_2D), ("1", x_2D), ("2", x_2D), ("3", x_2D), ("pool", x_2D)])
            for key, value in still_features.items():
                still_features[key] = nn.functional.interpolate(value, scale_factor=self.resize_factor[key], mode="bilinear") # (bs, ch, w_featmap, h_featmap) With different resizings  
        
        # Forward through the backbones, layer by layer
        h_fast = self.fast_backbone.blocks[0](h_fast)
        fast_features = OrderedDict()
        for layer in self.fast_layers:
            layer_fast = self.fast_backbone.blocks[layer]
            h_fast = layer_fast(h_fast)
            fast_features[f"{layer-1}"] = h_fast
        #FPN to fast features
        fast_features = self.fast_fpn(fast_features)
            
        #Prepyramid fusion
        for layer in self.fast_layers:
            still_features[f"{layer-1}"] = getattr(self, f"pre_pyramid_fusion_block_{layer}") (still_features[f"{layer-1}"], fast_features[f"{layer-1}"])
            
        return still_features

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name= 'dinov2_vitb14', #'dinov2_vits14',#'dinov2_vitb14',
            num_trainable_blocks=1,
            norm_layer=True,
            return_token=True
        ):
        super().__init__()
        print('**************** We are loading DINO-v2, the 2D encoder !!')
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = 768 #DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        for name, parameter in self.model.named_parameters():
            if name.startswith('blocks') and int(name.split('.')[1]) < len(self.model.blocks) - num_trainable_blocks:
                parameter.requires_grad_(False)

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape
        new_H, new_W = H // 1.5, W // 1.5
        new_H = int((new_H // 14) * 14 if new_H % 14 != 0 else new_H)
        new_W = int((new_W // 14) * 14 if new_W % 14 != 0 else new_W)
        x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        x, pos_encoding = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
            
        """
        with torch.no_grad():
            for blk in self.model.blocks:
                x = blk(x)
        x = self.model.norm(x)
        x = x.detach()
        """
        t = x[:, 0].unsqueeze(1)
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, new_H // 14, new_W // 14, self.num_channels)).permute(0, 3, 1, 2)
        
        if self.return_token:
            return f, t
        return f

import torch 
from torch import nn
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, resnet_fpn_backbone, BackboneWithFPN
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from collections import OrderedDict

class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        still_backbone_channels = [256, 512, 1024, 2048]
        pyramid_channels = 256
        pretrained = True
        trainable_layers = 3
        self.layers = range(1, 5)
        self.resnet = resnet_fpn_backbone('resnet50', pretrained, trainable_layers = trainable_layers)
        
    def forward(self, x):
        x = self.resnet(x)
        x_class = torch.nn.functional.adaptive_avg_pool2d(x["pool"], (1, 1)).squeeze(2).squeeze(2)
        print('RESNET OUTPUT')
        return x["0"], x_class.unsqueeze(1)
    
class X3D_FPN(nn.Module):
    def __init__(self):
        super().__init__()
        fast_backbone = torch.hub.load("facebookresearch/pytorchvideo", model="x3d_m", pretrained=True)
        channels_list = [24, 48, 96, 192]
        del fast_backbone.blocks[5]
        self.fast_backbone = fast_backbone
        self.fast_layers = range(1, 5) # List[int]
        self.fast_fpn = FeaturePyramidNetwork3D([24, 48, 96, 192], 192, extra_blocks=LastLevelMaxPool3D())
    
    def forward(self, h_fast_resize):
        # Forward through the backbones, layer by layer
        h_fast = self.fast_backbone.blocks[0](h_fast_resize)
        fast_features = OrderedDict()
        for layer in self.fast_layers:
            layer_fast = self.fast_backbone.blocks[layer]
            h_fast = layer_fast(h_fast)
            fast_features[f"{layer-1}"] = h_fast
        #FPN to fast features
        fast_features = self.fast_fpn(fast_features)
        fast_features_highres = fast_features["2"]
        x_3D_class = fast_features["pool"].mean(dim=(2, 3, 4)).unsqueeze(1) # (bs, ch)
        return fast_features_highres, x_3D_class

class DINO_2D_with_EgoVLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print('**************** We are loading DINO-v2, the 2D encoder !!')
        self.patch_size = 14
        #self.resize_factor =  {"0": 14/4, "1": 14/8, "2": 14/16, "3": 14/32, "pool": 14/64}
        self.resize_factor = {"0": 1/4, "1": 1/8, "2": 1/16, "3": 1/32, "pool": 1/64}
        self.out_channels = 256
        self.last_layers = [9, 10, 11]
        self.dino_extractor = DINOv2()
        #self.dino_extractor = build_swin_transformer('swin_L_384_22k', 384, dilation=False)
        #self.resnet_fpn = Resnet50()
        
        print('**************** We are loading the EgoVLP encoder !!')
        self.egovlp = SpaceTimeTransformer(num_frames = 16, time_init = 'zeros', attention_style = 'frozen-in-time')
        official_egovlp_weights = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/EgoVLPv2.pth', map_location='cpu')
        prefix = "module.video_model."
        selected_TimeSformer_weights = {k[len(prefix):]: v for k, v in official_egovlp_weights['state_dict'].items() if k.startswith(prefix)}
        selected_TimeSformer_weights = inflate_positional_embeds(selected_TimeSformer_weights) # The weights are for only 4 frames, but we adapt for 16
        self.egovlp.load_state_dict(selected_TimeSformer_weights, strict=True)
        
        trainable_blocks = 4
        for name, parameter in self.egovlp.named_parameters(): #We modify also the EgoVLP function
            if name.startswith('blocks') and int(name.split('.')[1]) < len(self.egovlp.blocks) - trainable_blocks:
                parameter.requires_grad_(False)
            print(name, parameter.requires_grad)
        #self.x3d_fpn = X3D_FPN()
            
        print('**************** We are loading the fusion blocks !!')
        dino_backbone_channels =  [768, 768, 768, 768, 768] #[384, 384, 384, 384, 384]
        egovlp_backbone_channels = [768, 768, 768, 768, 768] #[192, 192, 192, 192, 192]
        self.egovlp_embed_dim = 768
        self.dino_embed_dim = 768
        self.layers = [0, 1, 2, 3, 'pool'] #'pool'] #range(0, 4) # List[int]

        self.temporal_pooling = NonLocalTemporalPooling(self.egovlp_embed_dim, 'half', 16) #Temporal pooling with attention mechanism
        self.dual_cross_attn = Dual_CrossAttention(channels_2d = self.dino_embed_dim, channels_3d = self.egovlp_embed_dim)
        for layer, c2d, c3d in zip(self.layers, dino_backbone_channels, egovlp_backbone_channels):
            setattr(self, f"fusion_block_{layer}", ConvolutionalFusionBlock(pooling = None, 
                                                                            conv_block_architecture = 'simple_convolution', 
                                                                            post_up_conv_block = True, post_sum_conv_block = True, gating_block = None,
                                                                            pooling_frames = 16, 
                                                                            channels_2d = c2d, channels_3d = self.dino_embed_dim, 
                                                                            temporal_nonlocal_pooling_inter_channels= 128,
                                                                            temporal_nonlocal_pooling_max_height_before_max_pooling = 16))
        self.fuse_clase_token = nn.Sequential(nn.Linear(self.dino_embed_dim + self.egovlp_embed_dim, 768), nn.ReLU(), nn.Linear(768, 384))
        
        self.intermediate_outputs = None

       

    def forward(self, x):
        h_still, h_fast = x 
        out = {}
        B, C, H_still, W_still = x.shape
        x_2D, x_2D_class = self.dino_extractor(h_still)
        
        # ------------ EgoVLP extraction ------------ #
        bs, ch, frames, h_f, w_f = h_fast.shape
        for t in range(frames):
            h_t_resize = F.interpolate(h_fast[:, :, t, :, :], size=(224, 336), mode='bilinear') #If we reshape to (224, 336), then N_patches = 14*21 = 294
            h_fast_resize = h_t_resize.unsqueeze(2) if t == 0 else torch.cat((h_fast_resize, h_t_resize.unsqueeze(2)), dim=2)
        h_fast_resize = h_fast_resize.permute(0,2,1,3,4).contiguous() #EgoVLP requires bs, T, ch, H, W

        x_3D_class, fast_features = self.egovlp(h_fast_resize) #Normalized!!
        fast_features = fast_features.reshape(frames * bs, 294, self.egovlp_embed_dim)
        fast_features = fast_features.transpose(2, 1).view(frames * bs, self.egovlp_embed_dim, 14, 21)
        fast_features = fast_features.view(bs, frames, self.egovlp_embed_dim, 14, 21).permute(0, 2, 1, 3, 4) #-> (bs, ch, T, H, W)  
        #fast_features, x_3D_class = self.x3d_fpn(h_fast_resize)
        x_3D = self.temporal_pooling(fast_features) # (bs, ch, H = 14, W = 21)
        
        patch_2D_fused, patch_3D_fused, class_2D_fused, class_3D_fused, attn_2D, attn_3D = self.dual_cross_attn(x_2D, x_3D, x_2D_class, x_3D_class)
        still_features = OrderedDict([("0", patch_2D_fused), ("1", patch_2D_fused), ("2", patch_2D_fused), ("3", patch_2D_fused), ("pool", patch_2D_fused)])
        for key in still_features.keys():
            still_features[key] = getattr(self, f"fusion_block_{key}") (still_features[f"{key}"], patch_3D_fused, self.resize_factor[key], H_still, W_still)
        still_features['global'] = self.fuse_clase_token(torch.cat((class_2D_fused, class_3D_fused), dim=1))
       
        self.intermediate_outputs = out
        return still_features


class Convolutional_Pooling_T(nn.Module):
    #@configurable
    def __init__(
        self, 
        channels_2d: int,
        channels_3d: int,
        pooling: str,
        conv_block_architecture: str,
        post_up_conv_block: bool,
        post_sum_conv_block: bool,
        gating_block: str,
        pooling_frames: int,
        temporal_nonlocal_pooling_inter_channels: Union[str,int],
        temporal_nonlocal_pooling_max_height_before_max_pooling: int
    ):
        super().__init__()

        self.pooling = nn.Conv3d(channels_3d, channels_3d, kernel_size=(16,1,1), padding=(0,0,0))
        self.adapt_pooling = nn.Conv2d(channels_3d, channels_2d, kernel_size=3, padding = 1)
        self.conv_block_architecture = conv_block_architecture
        self.dino_embed_dim = channels_2d
        self.egovlp_embed_dim = channels_3d

        #if post_up_conv_block:
        #    self.post_up_conv_block = self._build_conv_block(channels_3d, channels_2d)
        #else:
        #    self.post_up_conv_block = nn.Identity()
        #self.post_up_conv_2d = self._build_conv_block(channels_2d, channels_2d)
        #self.post_up_conv_3d = self._build_conv_block(channels_3d, channels_2d)

        if post_sum_conv_block:
            self.post_sum_conv_block = self._build_conv_block(384, 384)
        else:
            self.post_sum_conv_block = nn.Identity()


    def _build_conv_block(self, in_channels, out_channels):
        if self.conv_block_architecture == 'simple_convolution':
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        elif self.conv_block_architecture == 'Conv2dNormActivation':
            return Conv2dNormActivation(in_channels, out_channels, kernel_size=3)
        else:
            raise ValueError(f'Unknown convolution block architecture: {self.conv_block_architecture}')

    def forward(self, x_2D, x_3D, resize_value, H_still, W_still):
        up_2d = F.interpolate(x_2D, (int(resize_value * H_still), int(resize_value * W_still)), mode="nearest")
        pooled_3d = self.pooling(x_3D).squeeze(2)
        pooled_3d = self.adapt_pooling(pooled_3d)
        up_3d = F.interpolate(pooled_3d, up_2d.shape[-2:], mode="nearest")
        out_patch = self.post_sum_conv_block(up_2d + up_3d)
        return out_patch

class ConvolutionalFusionBlock(nn.Module):
    #@configurable
    def __init__(
        self, 
        channels_2d: int,
        channels_3d: int,
        pooling: str,
        conv_block_architecture: str,
        post_up_conv_block: bool,
        post_sum_conv_block: bool,
        gating_block: str,
        pooling_frames: int,
        temporal_nonlocal_pooling_inter_channels: Union[str,int],
        temporal_nonlocal_pooling_max_height_before_max_pooling: int
    ):
        super().__init__()

        self.pooling = pooling
        #self.temporal_pooling = NonLocalTemporalPooling(768, 'half', 16) #Temporal pooling with attention mechanism
        #self.dual_cross_attn = Dual_CrossAttention(channels_2d = 384, channels_3d = 768)
        self.conv_block_architecture = conv_block_architecture
        self.dino_embed_dim = channels_2d
        self.egovlp_embed_dim = channels_3d

        #self.dual_cross_attn = Dual_CrossAttention(channels_2d, channels_3d = self.egovlp_embed_dim)
        
        #if post_up_conv_block:
        #    self.post_up_conv_block = self._build_conv_block(channels_3d, channels_2d)
        #else:
        #    self.post_up_conv_block = nn.Identity()
        #self.post_up_conv_2d = self._build_conv_block(channels_2d, 768)
        #self.post_up_conv_3d = self._build_conv_block(768, 256)

        if post_sum_conv_block:
            self.post_sum_conv_block = self._build_conv_block(channels_2d, 768)
            #self.post_sum_conv_block = nn.Sequential(nn.Conv2d(768 + 384, 768, kernel_size=(3,3), padding=(1,1)), nn.ReLU(), 
            #                                         nn.Conv2d(768, 384, kernel_size=(3,3), padding=(1,1)))
            
        else:
            self.post_sum_conv_block = nn.Identity()


    def _build_conv_block(self, in_channels, out_channels):
        if self.conv_block_architecture == 'simple_convolution':
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        elif self.conv_block_architecture == 'Conv2dNormActivation':
            return Conv2dNormActivation(in_channels, out_channels, kernel_size=3)
        else:
            raise ValueError(f'Unknown convolution block architecture: {self.conv_block_architecture}')

    def forward(self, x_2D, x_3D, resize_value, H_still, W_still):
        #fast_features_pooled = self.temporal_pooling(in_3d) # (bs, ch, H = 14, W = 21)
        #fused = self.cross_attention(in_2d, fast_features_pooled) #Which are the original feature maps
        #up_fused = F.interpolate(fused, scale_factor=resize_value, mode="bilinear")
        #x = self.post_sum_conv_block(up_fused)
        
        
        #Version of 18 Enero, works the best so far
        #x_2D, x_3D = self.dual_cross_attention(x_2D, x_3D)
        up_2d = F.interpolate(x_2D, (int(resize_value * H_still), int(resize_value * W_still)), mode="bilinear")
        #up_2d = self.post_up_conv_2d(up_2d)
        up_3d = F.interpolate(x_3D, up_2d.shape[-2:], mode="bilinear")
        #up_3d = self.post_up_conv_3d(up_3d)
        out_patch = self.post_sum_conv_block(up_2d + up_3d)
        #out_patch = self.post_sum_conv_block(torch.cat((up_2d, up_3d), dim=1))
        return out_patch
            
       
class DINO_2D_3D_feature_extractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.resize_factor =  {"0": 14/4, "1": 14/8, "2": 14/16, "3": 14/32, "pool": 14/64}
  
        self.embedding_size = cfg.MODEL.STILL.BACKBONE.EMBEDDING_SIZE
        self.last_layers = [11]
        self.out_channels = self.embedding_size

        if self.embedding_size == 384:
            self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif self.embedding_size == 768:
            self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif self.embedding_size == 1024:
            self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        else:
            raise ValueError(f"Embedding size {self.embedding_size} not supported")
        for name, parameter in self.dino_extractor.named_parameters():
            parameter.requires_grad_(False)
        
        self.still_backbone_channels = [384, 384, 384, 384]
        self.fast_backbone_channels = [384, 384, 384, 384]

        self.pre_pyramid_fusion = cfg.MODEL.STILLFAST.FUSION.PRE_PYRAMID_FUSION
        self.post_pyramid_fusion = cfg.MODEL.STILLFAST.FUSION.POST_PYRAMID_FUSION
        self.lateral_connections = cfg.MODEL.STILLFAST.FUSION.LATERAL_CONNECTIONS

        self.layers = range(1, 5) # List[int] 
        
        fusion_block = cfg.MODEL.STILLFAST.FUSION.FUSION_BLOCK
        if cfg.MODEL.STILLFAST.FUSION.FUSION_BLOCK == 'convolutional':
            fusion_block = partial(ConvolutionalFusionBlock_old, cfg.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK)
        else:
            raise ValueError(f'Unknown fusion block: {cfg.MODEL.STILLFAST.FUSION.FUSION_BLOCK}')

        if self.pre_pyramid_fusion:
            for layer, c2d, c3d in zip(self.layers, self.still_backbone_channels, self.fast_backbone_channels):
                setattr(self, f"pre_pyramid_fusion_block_{layer}", fusion_block(channels_2d=c2d, channels_3d=c3d))

        if self.post_pyramid_fusion:
            for layer, c2d, c3d in zip(self.layers, self.still_backbone_channels, self.fast_backbone_channels):
                setattr(self, f"post_pyramid_fusion_block_{layer}", fusion_block(channels_2d=pyramid_channels, channels_3d=pyramid_channels))

        if self.lateral_connections:
            for layer, c2d, c3d in zip(self.layers, self.still_backbone_channels, self.fast_backbone_channels):
                setattr(self, f"lateral_connection_fusion_block_{layer}", fusion_block(channels_2d=c2d, channels_3d=c3d))

        #self.still_fpn = FeaturePyramidNetwork(self.still_backbone_channels, pyramid_channels, extra_blocks=LastLevelMaxPool())
        
        if self.post_pyramid_fusion:
            self.fast_fpn = FeaturePyramidNetwork3D(self.fast_backbone_channels, pyramid_channels, extra_blocks=LastLevelMaxPool3D())

        self.temporal_convolutions = True
        if self.temporal_convolutions:
            self.temporal_layers = create_x3d_res_stage(depth = 3, 
                             dim_in = 384, 
                             dim_inner = int(2.25 * 384), 
                             dim_out = 384, 
                             conv_stride = (1, 1, 1), #We don't want to reduce the spatial dimensions
                             conv_kernel_size = (5, 3, 3),) #Modified from (3, 3, 3) to enlarge the temporal receptive field
        else:
            #self.temporal_attention = NonLocalTemporalPooling(num_channels = 384, 
            #                                                  inter_channels = 'half', 
            #                                                  max_height_before_pooling = 32)
            self.temporal_attention = Space_Time_Attention(num_frames=16, attention_type='divided_space_time', 
                                                           depth = 3, patch_size = 14, embed_dim = 768, num_heads = 12)
            attn_imagenet_pth = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth'
            self.temporal_attention.load_state_dict(torch.hub.load_state_dict_from_url(attn_imagenet_pth, progress=True), strict=False)

            self.adapt_clss_channels = nn.Linear(384, 768, bias=False)
            self.adapt_patch_channels = nn.Linear(384, 768, bias=False)
            self.adapt_back_channels = nn.Conv2d(768, 384, kernel_size=1, bias=False)
            
                
            """
            temporal_kernel_sizes = [7, 7, 5]  # Por ejemplo, tres capas con kernel size de 6 para lograr un receptive field de 16 en tiempo
            temporal_channels = [self.embedding_size, self.embedding_size // 2, self.embedding_size]  # Número de canales de salida para cada capa

            self.temporal_layers = nn.Sequential()
            for i, (k, out_ch) in enumerate(zip(temporal_kernel_sizes, temporal_channels)):
                in_ch = self.embedding_size if i == 0 else temporal_channels[i-1]
                padding = (2, 1, 1) if i == 2 else (k//2, 1, 1) 
                conv3d = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(k, 3, 3), stride = (1, 1, 1), padding = padding)
                bn = nn.BatchNorm3d(out_ch)
                relu = nn.ReLU()
                self.temporal_layers.append(nn.Sequential(conv3d, bn, relu))
            """
        

    def extract_2D_feats(self, x):
        with torch.no_grad():
            dino_output = self.dino_extractor.get_intermediate_layers(x, n = self.last_layers, reshape = True, return_class_token = True)
        return dino_output

    def extract_3D_feats(self, h_fast):
        with torch.no_grad():
            for b_t in range(0, h_fast.shape[0]):
                dino_3D = self.dino_extractor.get_intermediate_layers(h_fast[b_t, :, :, :].unsqueeze(0), n = self.last_layers, reshape = False, return_class_token = True)[0]
                patch_b_t = dino_3D[0]
                clss_b_t = dino_3D[1].unsqueeze(1)
                if b_t == 0:
                    dino_3D_patch = patch_b_t
                    dino_3D_clss = clss_b_t
                else:
                    dino_3D_patch = torch.cat((dino_3D_patch, patch_b_t), dim=0)
                    dino_3D_clss = torch.cat((dino_3D_clss, clss_b_t), dim=0)
        return dino_3D_patch, dino_3D_clss


    def rescale_Dino_features(self, dino_output):
        features_2d = OrderedDict([("0", dino_output), ("1", dino_output), ("2", dino_output), ("3", dino_output), ("pool", dino_output)])
        for key, value in features_2d.items():
            features_2d[key] = nn.functional.interpolate(value, scale_factor=self.resize_factor[key], mode="bilinear") # (bs, ch, w_featmap, h_featmap) With different resizings  
        return features_2d


    def forward(self, x):
        h_still, h_fast = x

        #---------------------Still features---------------------#
        with torch.no_grad():
            dino_2D = self.dino_extractor.get_intermediate_layers(h_still, n = self.last_layers, reshape = True, return_class_token = True)[0][0]
        still_features = self.rescale_Dino_features(dino_2D) # Dict of {0: 1/4 scale, 1: 1/8 scale, 2: 1/16 scale, 3: 1/32 scale, pool: 1/64 scale}

        #---------------------Fast features---------------------#
        B, C, T, H_low_res, W_low_res = h_fast.shape
        h_fast = h_fast.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H_low_res, W_low_res) #h_fast.view(bs * N_frames, rgb_ch, h_low_res, w_low_res)  #(BS * N_frames, 384, h, w)
        dino_3D_patch, dino_3D_clss = self.extract_3D_feats(h_fast) # (BS, 384, h, w)
        #dino_3D_no_T, _ = self.extract_3D_feats(h_fast) # (BS, 384, h, w)
        #print(dino_3D_no_T.shape, 'dino_3D_patch')
                
        if self.temporal_convolutions:
            _, dino_C, H_dino, W_dino = dino_3D_no_T.shape
            dino_3D_no_T = dino_3D_no_T.view(B, T, dino_C, H_dino, W_dino).permute(0, 2, 1, 3, 4) #.view(bs, dino_ch, N_frames, h_dino, w_dino) # (BS, 384, N_frames, h, w)
            dino_3D_T = self.temporal_layers(dino_3D_no_T) # (BS, 384, T, h, w)
            dino_3D_T_mean = torch.mean(dino_3D_T, dim=2) # (BS, 384, h, w)
            fast_features = self.rescale_Dino_features(dino_3D_T_mean) # Dict of {0: 1/4 scale, 1: 1/8 scale, 2: 1/16 scale, 3: 1/32 scale, pool: 1/64 scale}
        else:
            dino_3D_clss = self.adapt_clss_channels(dino_3D_clss)
            dino_3D_patch = self.adapt_patch_channels(dino_3D_patch)
            dino_3D_T = self.temporal_attention(dino_3D_clss, dino_3D_patch, W_low_res // 14) # (BS, 384, T, h, w)
            dino_3D_T = dino_3D_T.reshape(B, T, 768, H_low_res // 14, W_low_res // 14).permute(0, 2, 1, 3, 4)
            dino_3D_T_mean = torch.mean(dino_3D_T, dim=2) # (BS, 384, h, w)
            dino_3D_T_mean = self.adapt_back_channels(dino_3D_T_mean)
            fast_features = self.rescale_Dino_features(dino_3D_T_mean) # Dict of {0: 1/4 scale, 1: 1/8 scale, 2: 1/16 scale, 3: 1/32 scale, pool: 1/64 scale}
        
        #---------------------Fusion---------------------#
        if self.pre_pyramid_fusion:
            for layer in self.layers:
                still_features[f"{layer-1}"] = getattr(self, f"pre_pyramid_fusion_block_{layer}") (still_features[f"{layer-1}"], fast_features[f"{layer-1}"])
        

        return still_features

if __name__ == '__main__':
    model = Dinov2()
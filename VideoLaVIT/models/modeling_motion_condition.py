import sys
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import partial, reduce
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


try:
    from apex.normalization import FusedLayerNorm
except:
    FusedLayerNorm = LayerNorm
    print("Please 'pip install apex'")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn
            
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class MotionEncoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
            attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-6)):
        super().__init__()

        self.norm0 = norm_layer(dim)

        self.spatial_attention = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
        )

        self.norm1 = norm_layer(dim)
        self.temporal_attention = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        bs, t, seq_len, c = x.shape
        x = x.reshape(bs * t, seq_len, c)
        x = x + self.spatial_attention(self.norm0(x))
        x = x.reshape(bs, t, seq_len, c).permute(0, 2, 1, 3)
        x = x.reshape(bs * seq_len, t, c)
        x = x + self.temporal_attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(bs, seq_len, t, c).permute(0, 2, 1, 3)
        return x


class MotionConditionEncoder(nn.Module):

    def __init__(self, in_channel=2, out_channel=1024, dim=512, num_heads=8, img_size=(16, 16), depth=12, mlp_ratio=4., qkv_bias=True, 
            qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-5), add_temp_pos=False, num_frames=24):
        super().__init__()
        self.blocks = nn.ModuleList([
            MotionEncoderBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(dim)

        self.in_proj = nn.Linear(in_channel, dim)
        self.out_proj = nn.Linear(dim, out_channel)

        num_patches = img_size[0] * img_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , dim))
        trunc_normal_(self.pos_embed, std=.02)

        if add_temp_pos:
            self.temp_pos = nn.Parameter(torch.zeros(1, num_frames, dim))
            trunc_normal_(self.temp_pos, std=.02)
        
        self.add_temp_pos = add_temp_pos

    def forward(self, x):
        # x: bs, 2, t, h, w
        bs, in_channel, t, h, w = x.shape
        assert in_channel == 2
        x = x.permute(0, 2, 3, 4, 1)  # bs, t, h, w, c
        x = x.reshape(bs, t, h * w, in_channel)

        x = self.in_proj(x)
        pos_embed = self.pos_embed.unsqueeze(0)
        x = x + pos_embed

        if self.add_temp_pos:
            temp_pos = self.temp_pos.unsqueeze(2)
            x = x + temp_pos

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)    # the post norm, for next stage use
        x = self.out_proj(x)  # bs, t, h * w, c

        return x


def build_condition_encoder(dim=512, num_heads=8, depth=12, add_temp_pos=False, **kwargs):
    # Building Model
    model = MotionConditionEncoder(in_channel=2, out_channel=1024, dim=dim, 
            num_heads=num_heads, img_size=(16, 16), depth=depth, add_temp_pos=add_temp_pos,)

    return model
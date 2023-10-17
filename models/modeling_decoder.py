import math
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

# The vqkd decoder to reconstruct the image semantics

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
    print("Please install apex")


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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, codebook_embeds, codebook_mask):
        B, N, C = codebook_embeds.shape
        _, N_x, _ = x.shape

        q = self.query(x).reshape(B, N_x, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(codebook_embeds).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(codebook_embeds).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale

        extended_mask = codebook_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        attn = attn + extended_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_x, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
            attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm0 = norm_layer(dim)
        self.self_attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, codebook_embeds, codebook_mask):
        x = x + self.self_attn(self.norm0(x))
        x = x + self.cross_attn(self.norm1(x), codebook_embeds, codebook_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VQDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=32, embed_dim=1408, 
            depth=12, num_heads=16, mlp_ratio=4.3637, qkv_bias=True, qk_scale=None, drop_rate=0., 
            attn_drop_rate=0., norm_layer=partial(FusedLayerNorm, eps=1e-5), **kwargs):
        super().__init__()

        self.in_proj = nn.Linear(in_chans, embed_dim)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # The postion embedding for the latent code

        self.query_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # The query embedding for reconstruction
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # The decoder task layer
        self.decoder_out_dim = 1408
        self.decode_task_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, self.decoder_out_dim),
        )

        self.unet_proj = nn.Linear(self.decoder_out_dim, 768)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'query_embed'}

    def decode(self, quantize, decisions, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        decoder_features = self.decoder(quantize, decisions)
        # print(decoder_features)
        rec = self.decode_task_layer(decoder_features)
        return rec

    def forward(self, x, token_num):
        # codebook_fea
        # B, nc, w, h = codebook_fea.shape
        x = self.in_proj(x)
        B = len(token_num)
        num_tokens, C = x.shape
        device = x.device

        x_list = torch.split(x, token_num.tolist(), dim=0)
        max_token_num = token_num.max().item()
        x_pad = torch.zeros(B, max_token_num, C, dtype=x.dtype).to(device)
        mask = torch.zeros(B, max_token_num, dtype=x.dtype).to(device)
        
        for i, x_tensor in enumerate(x_list):
            x_pad[i][:len(x_tensor)] = x_tensor
            mask[i][:len(x_tensor)] = 1
        
        x_pad = x_pad + self.pos_embed[:,:max_token_num]
        x_pad = self.pos_drop(x_pad)

        query_embeds = self.query_embed.expand(B, -1, -1)

        for blk in self.blocks:
            query_embeds = blk(query_embeds, codebook_embeds=x_pad, 
                    codebook_mask=mask)

        query_embeds = self.norm(query_embeds)  # To align with the raw vit features

        visual_rec = self.decode_task_layer(query_embeds)

        visual_rec = self.unet_proj(visual_rec)

        return visual_rec


def build_tokenizer_decoder(model_path=''):
    model = VQDecoder(depth=12)
    weight_path = os.path.join(model_path, 'visual_tokenizer', 'tokenizer_decoder.bin')
    print(f"Load visual tokenizer decoder weight from {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu") 
    model.load_state_dict(state_dict)
    return model
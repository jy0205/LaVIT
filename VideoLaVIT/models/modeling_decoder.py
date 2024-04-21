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


class AttentionPool2d(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(seq_len + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, return_all_tokens=False):
        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = x.permute(1, 0, 2)   # (N(HW)C) => (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        if return_all_tokens:
            return x
        else:
            return x[0]


class HighresVQDecoder(nn.Module):
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

        # Convert the decoded features to Unet Condition
        self.unet_proj_1 = nn.Linear(self.decoder_out_dim, 768)
        self.unet_proj_2 = nn.Linear(self.decoder_out_dim, 1280)
        self.unet_attnpool = AttentionPool2d(num_patches, self.decoder_out_dim, num_heads, 1280)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'query_embed'}

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

        encoder_hidden_1 = self.unet_proj_1(visual_rec)   # [bs, 256, 768]
        encoder_hidden_2 = self.unet_proj_2(visual_rec)   # [bs, 256, 1280]
        prompt_embeds = torch.cat([encoder_hidden_1, encoder_hidden_2], dim=-1)   # [bs, 256, 2048]
        pooled_prompt_embeds = self.unet_attnpool(visual_rec)   # [bs, 1280]

        return prompt_embeds, pooled_prompt_embeds


def build_tokenizer_decoder(model_path=''):
    model = HighresVQDecoder(depth=12)
    weight_path = os.path.join(model_path, 'visual_tokenizer', 'highres_tokenizer_decoder.bin')
    print(f"Load visual tokenizer decoder weight from {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu") 
    model.load_state_dict(state_dict)
    return model
import torch
import os
import numpy as np
from torch import nn
import torch.nn.functional as F
import math

from collections import OrderedDict
from functools import partial, reduce
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from models.modeling_visual_encoder import build_eva_clip


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


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)


class CodebookEmbedding(nn.Module):
    def __init__(self, num_tokens, codebook_dim):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        weight = torch.randn(num_tokens, codebook_dim)
        weight = l2norm(weight)
        self.weight = nn.Parameter(weight)
        
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.embedding = CodebookEmbedding(self.num_tokens, self.codebook_dim)

    def tokenize(self, z):
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)

        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'
        
        encoding_indices = torch.argmin(d, dim=1)
        
        z_q = self.embedding(encoding_indices)  # [np, d]
        
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)   # [np, 16384]

        return z_q, encoding_indices
    
    def get_quantize_from_id(self, encoding_indices):
        z_q = self.embedding(encoding_indices)  # [np, d]
        return z_q


class TokenCrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.size()
        B, H, N, N = attn.size()
        fuse_policy = 1 - policy    # Each token only attend to the dropped tokens
        attn_policy = fuse_policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        attn_policy = attn_policy.expand(B, 1, N, N)
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)

        return attn.type_as(max_att)

    def forward(self, x, x_origin, decisions):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x_origin).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(x_origin).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax_with_policy(attn, decisions)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TokenCausalAttention(nn.Module):

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

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.size()
        device = attn.device
        assert attn.shape[-1] == attn.shape[-2]
        assert attn.shape[-2] == N
        B, H, N, N  = attn.size()
        
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye

        # Use the causal attention
        seq_ids = torch.arange(N, device=device)
        causal_mask = (
            seq_ids[None, None, :].repeat(B, N, 1)
            <= seq_ids[None, :, None]
        )
        causal_mask = causal_mask[:,None,:,:].to(attn_policy.dtype)
        attn_policy = attn_policy * causal_mask

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, decisions):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if decisions is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, decisions)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalFuserBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
            attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-5)):
        super().__init__()

        self.norm0 = norm_layer(dim)
        self.token_causal_attn = TokenCausalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )

        self.norm1 = norm_layer(dim)
        self.token_cross_attn = TokenCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_origin, decisions):
        x = x + self.token_causal_attn(self.norm0(x), decisions)
        x = x + self.token_cross_attn(self.norm1(x), x_origin, decisions)
        x = x + self.mlp(self.norm2(x))
        return x


class TokenMerger(nn.Module):

    def __init__(self, dim, num_heads, depth=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
            attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-5)):
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalFuserBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])

        self.ln_vision = norm_layer(dim)

        self.norm = norm_layer(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, decisions):
        x_origin = self.ln_vision(x)   # the  raw vit features needs layer normalization

        for blk in self.blocks:
            x = blk(x, x_origin, decisions)

        x = self.norm(x)    # the post norm, for next stage use

        return x


class TokenPredictor(nn.Module):

    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            FusedLayerNorm(embed_dim, eps=1e-5),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


class DynamicVisualTokenizer(nn.Module):
    def __init__(self, img_size=224, patch_size=14, width=1408, layers=12, 
        heads=16, n_code=16384, code_dim=32, model_path='', use_xformers=False):
        """
        The dynamic visual tokenizer in LaVIT, it has 12 transformer blocks
        """
        super().__init__()

        self.encoder = build_eva_clip(model_path=model_path, use_xformers=use_xformers)
        self.encoder.eval()
        # Freeze the vit encoder
        for param in self.encoder.parameters():
            param.requires_grad = False # fix encoder model

        encoder_config = dict(img_size=224, patch_size=14, in_chans=32, embed_dim=1408, depth=12, num_heads=16,  
                mlp_ratio=4.3637, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., norm_layer=partial(FusedLayerNorm, eps=1e-5))

        encoder_config['img_size'] = img_size
        encoder_config['patch_size'] = patch_size
        encoder_config['embed_dim'] = width
        encoder_config['depth'] = layers
        encoder_config['num_heads'] = heads

        # The token predictor
        self.token_predictor = TokenPredictor(encoder_config['embed_dim'])

        # The token merger
        self.causal_encoder = TokenMerger(
            encoder_config['embed_dim'],
            num_heads=encoder_config['num_heads'],
            depth=encoder_config['depth'],
            mlp_ratio=encoder_config['mlp_ratio'],
            qkv_bias=encoder_config['qkv_bias'],
            qk_scale=encoder_config['qk_scale'],
            drop=encoder_config['drop_rate'],
            attn_drop=encoder_config['attn_drop_rate'],
        )

        # The code book embeddings
        self.quantize = VectorQuantizer(n_embed=n_code, embedding_dim=code_dim)

        # encoder task layer, map the feature to the codebook's dimension
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim) # for quantize
        )

        # The vit projection, map the visual feature to LLM's input space
        llm_embed_dim = 4096   # LLaMA 7B's embedding dimension: 4096
        self.vit_proj = nn.Linear(width, llm_embed_dim)

    def encode_features(self, x):
        """
        x: B, 3, H, W
        Usage: Given the input image, encode the visual features for the LLM, without quantization,
            Used for Understanding
        """
        device = x.device
        with torch.no_grad():
            encoder_features = self.encoder(x, return_all_features=True)   # N, 257, D
            encoder_features = encoder_features[:,1:,:]

            B, num_patches, _ = encoder_features.shape
            mask = torch.ones(B, num_patches, 1, dtype=encoder_features.dtype, device=encoder_features.device)

            # To evalaute the score
            pred_score = self.token_predictor(encoder_features, mask).reshape(B, -1, 2)
            # Sample from the score distribution
            hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0]   # [N, num_patches]
        
        # Update the existed features from dropped tokens (To remain the information flow)
        updated_features = self.causal_encoder(encoder_features, hard_keep_decision)
        updated_features = self.vit_proj(updated_features)  # [bs, 256, 4096]

        B, N, C = updated_features.shape
        index_select = hard_keep_decision.long()

        token_num = index_select.sum(dim=-1)
        index_select = index_select.bool()

        remained_token = torch.masked_select(updated_features, index_select[:,:,None])
        remained_token = remained_token.reshape(-1, C)  # [Num Patch]
        remained_token_list = torch.split(remained_token, token_num.tolist())  # [bs]
        remained_token_list = list(remained_token_list)

        return remained_token_list

    def tokenize_image(self, x_tensor, add_special=False, used_for_llm=True):
        # x_tensor: [bs, 3, h, w]
        feature_targets = self.encoder(x_tensor, return_all_features=True)   # N, 257, D
        encoder_features = feature_targets[:,1:,:]

        B, num_patches, _ = encoder_features.shape
        mask = torch.ones(B, num_patches, 1, dtype=encoder_features.dtype, device=encoder_features.device)

        pred_score = self.token_predictor(encoder_features.to(torch.float32), mask).reshape(B, -1, 2)
        # Sample from the score distribution
        hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0]   # [N, num_patches]

        # Update the existed features from dropped tokens (To remain the information flow)
        updated_features = self.causal_encoder(encoder_features, hard_keep_decision)

        B, N, C = updated_features.shape
        index_select = hard_keep_decision.long()
        token_nums = index_select.sum(dim=-1)
        index_select = index_select.bool()
        remained_token = torch.masked_select(updated_features, index_select[:,:,None]).reshape(-1, C)  # [Num Patch]
        
        to_quantizer_features = self.encode_task_layer(remained_token.type_as(self.encode_task_layer[-1].weight))  
        quantize, embed_ind = self.quantize.tokenize(to_quantizer_features)
        
        if not used_for_llm:
            return quantize, token_nums

        embed_ind = embed_ind + 32002
        embed_ind_list = torch.split(embed_ind, token_nums.tolist(), dim=0)

        if add_special:
            # If pad the special image start and end tokens, default is False
            output_embed_ind = []
            image_special = torch.as_tensor([32000, 32001], dtype=torch.long).to(x_tensor.device)
            for ele in embed_ind_list:
                output_embed_ind.append(torch.cat([image_special[:1], ele, image_special[1:]]))
            return output_embed_ind

        return embed_ind_list
        

def build_dynamic_tokenizer(model_path='', use_xformers=False, for_understanding=False, model_sub_dir='language_model'):
    model = DynamicVisualTokenizer(model_path=model_path, use_xformers=use_xformers)
    weight_path = os.path.join(model_path, 'visual_tokenizer', 'tokenizer_encoder.bin')
    print(f"Load visual tokenizer encoder weight from {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu") 
    model.load_state_dict(state_dict, strict=False)

    if for_understanding:
        # For Understanding, the LaVIT use the continuous visual features, 
        # so needs to load the token merger weight trained with LLM
        visual_weight_path = os.path.join(model_path, model_sub_dir, 'visual_weight.bin')
        print(f"For multi-modal understanding, Load visual tokenizer weight from {visual_weight_path}")
        state_dict = torch.load(visual_weight_path, map_location="cpu") 
        model.load_state_dict(state_dict, strict=False)

    return model

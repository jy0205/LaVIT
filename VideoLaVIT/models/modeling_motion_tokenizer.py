import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import partial, reduce
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.modeling_visual_tokenzier import VectorQuantizer


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


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv, kernel_size=None, stride=None, pad=None, permute=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=0)

        self.pad = pad
        self.permute = permute

    def forward(self, x):
        if self.with_conv:
            if self.permute:
                x = x.permute(0, 4, 1, 2, 3)
            x = torch.nn.functional.pad(x, self.pad, mode="constant", value=0)
            x = self.conv(x)
            if self.permute:
                x = x.permute(0, 2, 3, 4, 1)
        else:
            raise NotImplementedError("Not implemented")

        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True, scale_factor=None):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        ori_dtype = x.dtype
        x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=self.scale_factor, mode="nearest")
        x = x.to(ori_dtype)
        if self.with_conv:
            x = self.conv(x)
        return x


class SpatioTemporalBlock(nn.Module):

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


class TokenizerEncoder(nn.Module):

    def __init__(self, in_channel=2, dim=512, num_heads=8, img_size=(36, 20), depth=12, mlp_ratio=4., qkv_bias=True, 
            qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-5), num_frames=24):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(dim)
        self.dim = dim

        num_patches = img_size[0] * img_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.temp_pos = nn.Parameter(torch.zeros(1, num_frames, dim))
        trunc_normal_(self.temp_pos, std=.02)

        self.stage = [3, 3, 3, 3]
        self.kernel_size_list = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        self.stride_list = [(1, 2, 2), (2, 1, 1), (2, 1, 1), (2, 2, 2)]
        self.pad_list = [(0,1,0,1,1,1), (1,1,1,1,0,1), (1,1,1,1,0,1), (0,1,0,1,0,1)]
        self.down_strides = [(1, 2, 2), (2, 1, 1), (2, 1, 1), (2, 2, 2)]

        self.down_sample_layers = nn.ModuleList([
                Downsample(in_channels=dim, out_channels=dim, with_conv=True, kernel_size=self.kernel_size_list[i],
                    stride=self.stride_list[i], pad=self.pad_list[i])
            for i in range(len(self.stage))])

        self.in_proj = nn.Linear(in_channel, dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: bs, 2, t, h, w
        bs, in_channel, t, h, w = x.shape
        assert in_channel == 2

        x = x.permute(0, 2, 3, 4, 1)  # bs, t, h, w, c
        x = self.in_proj(x)
        dim = x.shape[-1]
        x = x.reshape(bs, t, h * w, dim)

        # add the spatio-temporal position embedding
        pos_embed = self.pos_embed.unsqueeze(0)
        x = x + pos_embed

        temp_pos = self.temp_pos.unsqueeze(2)
        x = x + temp_pos

        i_block = 0

        for i_s in range(len(self.stage)):
            layer_num = self.stage[i_s]
            
            for i_b in range(i_block, i_block + layer_num):
                x = self.blocks[i_b](x)

            # downsample
            x = x.reshape(bs, t, h, w, self.dim)
            x = self.down_sample_layers[i_s](x)
            down_stride = self.down_strides[i_s]
            t = t // down_stride[0]
            h = h // down_stride[1]
            w = w // down_stride[2]
            x = x.reshape(bs, t, h * w, self.dim)

            i_block = i_block + layer_num

        x = self.norm(x)    # the post norm, for next stage use,    # bs, t, h * w, c
        x = x.reshape(bs, t, h, w, self.dim)

        return x


class TokenizerDecoder(nn.Module):

    def __init__(self, in_channel=32, dim=512, num_heads=8, img_size=(36, 20), depth=12, mlp_ratio=4., qkv_bias=True, 
            qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-5), num_frames=24):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(dim)
        self.in_proj = nn.Linear(in_channel, dim)
        self.dim = dim

        self.stage = [3, 3, 3, 3]
        self.scale_factor_list = [(2.0, 2.0, 2.0), (2.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 2.0)]

        self.up_sample_layers = nn.ModuleList([
                Upsample(in_channels=dim, out_channels=dim, with_conv=True, scale_factor=self.scale_factor_list[i],)
            for i in range(len(self.stage))])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: bs, h, w, dim
        bs, t, h, w, dim = x.shape
        x = self.in_proj(x)

        i_block = 0

        for i_s in range(len(self.stage)):
            layer_num = self.stage[i_s]
            
            # upsample
            x = x.permute(0, 4, 1, 2, 3)
            x = self.up_sample_layers[i_s](x)   # x: bs, dim, t, h, w
            bs, dim, t, h, w = x.shape
            x = x.permute(0, 2, 3, 4, 1)  # bs, t, h, w, c
            x = x.reshape(bs, t, h * w, dim)

            for i_b in range(i_block, i_block + layer_num):
                x = self.blocks[i_b](x)

            x = x.reshape(bs, t, h, w, dim)

            i_block = i_block + layer_num

        x = self.norm(x)    # the post norm, for next stage use,    # bs, t, h, w, c

        return x


class MotionTransformerTokenizer(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 decoder_out_dim,
                 n_embed=1024, 
                 embed_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 rec_loss_type='l2',
                 **kwargs
                 ):
        """
        The motion tokenizer
        """
        super().__init__()
        print("Not used", kwargs)

        self.encoder = TokenizerEncoder(**encoder_config)
        self.decoder = TokenizerDecoder(**decoder_config)

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['dim'], encoder_config['dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['dim'], embed_dim) # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['dim'], decoder_config['dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['dim'], self.decoder_out_dim),
        )

        self.quantize = VectorQuantizer(n_embed=n_embed, embedding_dim=embed_dim)
        self.rec_loss_type = rec_loss_type
        self.kwargs = kwargs
        self.motion_scale_factor = 10.0

    @property
    def device(self):
        return self.decoder.pos_embed.device

    @torch.no_grad()
    def get_tokens(self, data, **kwargs):
        N, dim, T, H, W = data.shape
        assert dim == 2

        quantize, embed_ind = self.encode(data)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)   # [bs, 256]
        output['quantize'] = quantize

        return output

    @torch.no_grad()
    def reconstruct(self, x, height, width):
        N, dim, T, H, W = x.shape
        assert dim == 2
        quantize, embed_ind = self.encode(x)
        xrec = self.decode(quantize)

        assert x.shape == xrec.shape

        dtype = xrec.dtype
        motion_input = xrec.permute(0, 2, 1, 3, 4)
        motion_input = motion_input.reshape(N * T, dim, H, W)
        motion_input = torch.nn.functional.interpolate(motion_input.float(), (height, width), mode='bicubic') 
        motion_input = motion_input.reshape(N, T, dim, height, width)
        motion_input = motion_input.permute(0, 2, 1, 3, 4)
        motion_input = motion_input / self.motion_scale_factor
        motion_input = motion_input.to(dtype)
        
        return motion_input

    @torch.no_grad()
    def reconstruct_from_token(self, x, height, width):
        N, seq_len = x.shape    # seq_len = 135

        assert seq_len == 135

        quantize_embed = self.quantize.get_quantize_from_id(x.flatten(0,1))
        quantize_embed = quantize_embed.reshape(N, seq_len, quantize_embed.shape[-1])  # [N, seq_len, 32]

        t, h, w = 3, 5, 9
        quantize_embed = rearrange(quantize_embed, 'b (t h w) c -> b t h w c', t=t, h=h, w=w)
        xrec = self.decode(quantize_embed)

        dtype = xrec.dtype
        N, dim, T, H, W = xrec.shape
        motion_input = xrec.permute(0, 2, 1, 3, 4)
        motion_input = motion_input.reshape(N * T, dim, H, W)
        motion_input = torch.nn.functional.interpolate(motion_input.float(), (height, width), mode='bicubic') 
        motion_input = motion_input.reshape(N, T, dim, height, width)
        motion_input = motion_input.permute(0, 2, 1, 3, 4)
        motion_input = motion_input / self.motion_scale_factor
        motion_input = motion_input.to(dtype)

        return motion_input

    def encode(self, x):
        encoder_features = self.encoder(x)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))   # [bs, t, h, w, 32]

        quantize, embed_ind = self.quantize.tokenize(to_quantizer_features)

        quantize = quantize.view(to_quantizer_features.shape)

        return quantize, embed_ind

    def decode(self, quantize, **kwargs):
        # input quantize: [bs, t, h, w, 32]
        decoder_features = self.decoder(quantize)
        rec = self.decode_task_layer(decoder_features) # [bs, t, h, w, 2]
        rec = rec.permute(0, 4, 1, 2, 3)
        return rec

    def get_codebook_indices(self, x, **kwargs):
        return self.get_tokens(x, **kwargs)['token']


def get_motion_trans_model_params():
    return dict(in_channel=2, dim=512, num_heads=8, img_size=(36, 20), depth=12, mlp_ratio=4., qkv_bias=True, 
            qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-5), num_frames=24)


def build_motion_tokenizer(pretrained_weight=None, n_code=1024, code_dim=32, as_tokenizer=True, **kwargs):

    encoder_config, decoder_config = get_motion_trans_model_params(), get_motion_trans_model_params()

    # decoder settings
    decoder_config['in_channel'] = code_dim
    decoder_out_dim = 2
    
    model = MotionTransformerTokenizer(encoder_config, decoder_config, decoder_out_dim, n_code, code_dim, **kwargs)

    if as_tokenizer:
        # Load pretrained weight
        assert pretrained_weight is not None
        print(f"Load checkpoint of motion tokenizer from {pretrained_weight}")
        weights = torch.load(pretrained_weight, map_location='cpu')
        load_stat = model.load_state_dict(weights)

    return model
import os
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import partial, reduce
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module

from .modeling_3d_unet import UNetSpatioTemporalConditionModel
from .modeling_motion_condition import build_condition_encoder
from .modeling_motion_tokenizer import build_motion_tokenizer

import utils
import inspect
from utils import _resize_with_antialiasing
from PIL import Image
from tqdm import tqdm
from IPython import embed


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


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


class VideoDetokenizer(nn.Module):
    def __init__(self, model_path, model_dtype, use_xformer=True, encoder_dim=512, encoder_head=8,):
        super().__init__()
        
        tokenizer_weight = os.path.join(model_path, 'motion_tokenizer.bin')
        self.motion_tokenizer = build_motion_tokenizer(tokenizer_weight)

        # motion condtion
        self.motion_encoder = build_condition_encoder(dim=encoder_dim, num_heads=encoder_head)

        # The diffusion related parameters
        if model_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif model_dtype == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print(f"Using model dtype : {torch_dtype}")
        detokenizer_model_dir = os.path.join(model_path, 'video_detokenizer')
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(detokenizer_model_dir, subfolder="vae", torch_dtype=torch_dtype)  
        self.scheduler = EulerDiscreteScheduler.from_pretrained(detokenizer_model_dir, subfolder="scheduler",)
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(detokenizer_model_dir, subfolder="unet", torch_dtype=torch_dtype,)

        self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if use_xformer:
            print("using xformers in video unet")
            self.unet.enable_xformers_memory_efficient_attention()

        self.add_time_ids = None

    @property
    def device(self):
        return self.unet.device

    @torch.no_grad()
    def get_image_latents(self, x):
        latents = self.vae.encode(x.to(self.vae.dtype)).latent_dist.mode()
        return latents

    def encode_motion(self, motion):
        # motion : [bs, 2, T, 16, 16]
        with torch.no_grad():
            # Resize motion to [16, 16] to extract motion features
            bs, dim, t, h, w = motion.shape
            dtype = motion.dtype
            assert dim == 2
            motion_input = motion.permute(0, 2, 1, 3, 4)
            motion_input = motion_input.reshape(bs * t, dim, h, w)
            motion_input = torch.nn.functional.interpolate(motion_input.float(), (16, 16), mode='bicubic') 
            motion_input = motion_input.reshape(bs, t, dim, 16, 16)
            motion_input = motion_input.permute(0, 2, 1, 3, 4)
            motion_input = motion_input.to(dtype)

        motion_hidden_state = self.motion_encoder(motion_input)  # [bs, t, h*w, c]
        return motion_hidden_state

    @torch.no_grad()
    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        latents = latents.to(self.vae.dtype)

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)

        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    @torch.no_grad()
    def reconstruct_from_token(self, image, motion, width=256, height=256, num_frames=24, num_inference_steps=25, min_guidance_scale=1.0, 
            max_guidance_scale=4.0, fps=6, motion_bucket_id=127, cond_on_ref_frame=True, decode_from_token=False, noise_aug_strength=0.02,
            decode_chunk_size=8, num_videos_per_prompt=1, use_linear_guidance=True, quantize_motion=False):
        
        assert num_videos_per_prompt == 1, "Now only support one prompt one video"
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        batch_size = len(image)
        device = image.device

        # 1. Get motion conditions
        if quantize_motion:
            # Use the reconstructed motion as input
            motion = self.motion_tokenizer.reconstruct(motion, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if decode_from_token:
            motion = self.motion_tokenizer.reconstruct_from_token(motion, height // self.vae_scale_factor, width // self.vae_scale_factor)

        motion_conditions = self.encode_motion(motion)  # [bsz, num_frames, h*w, 1024]
        negative_motion_conditions = torch.zeros_like(motion_conditions)
        motion_conditions = torch.cat([negative_motion_conditions, motion_conditions])

        # 2. Encode ref_frame_condition using VAE
        noise = randn_tensor(image.shape, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise        

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self.get_image_latents(image)
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        if cond_on_ref_frame:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])
        else:
            image_latents = torch.cat([image_latents, image_latents])

        motion_latents = motion.permute(0, 2, 1, 3, 4)
        negative_motion_latents = torch.zeros_like(motion_latents)
        motion_latents = torch.cat([negative_motion_latents, motion_latents])

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 3. Get Added Time IDs
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        add_time_ids = torch.tensor([add_time_ids], dtype=image.dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids])
        added_time_ids = add_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = 8
        shape = (
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, device=device, dtype=image_latents.dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # self.guidance_scale = guidance_scale
        # 6. Prepare guidance scale
        if use_linear_guidance:
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
            guidance_scale = guidance_scale.to(device, latents.dtype)
            guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
            guidance_scale = _append_dims(guidance_scale, latents.ndim)
        else:
            guidance_scale = max_guidance_scale

        if isinstance(guidance_scale, torch.Tensor):
            self.guidance_scale = guidance_scale.to(self.unet.dtype)

        unet_dtype = self.unet.dtype
        motion_conditions = motion_conditions.to(unet_dtype)
        added_time_ids = added_time_ids.to(unet_dtype)

        # 6. Denoising loop
        for t in tqdm(timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimention
            latent_model_input = torch.cat([latent_model_input, image_latents, motion_latents], dim=2)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input.to(unet_dtype),
                t.to(unet_dtype),
                encoder_hidden_states=motion_conditions,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = tensor2vid(frames, self.image_processor, output_type="pil")

        return frames


def build_video_detokenizer(model_path, model_dtype='fp16', use_xformer=True, encoder_dim=512, encoder_head=8, pretrained_weight=None, **kwargs):
    model = VideoDetokenizer(model_path, model_dtype, use_xformer=use_xformer, encoder_dim=encoder_dim, encoder_head=encoder_head, **kwargs)
    if pretrained_weight is None:
        pretrained_weight = os.path.join(model_path, 'video_detokenizer/unet/diffusion_pytorch_model.bin')
    print(f"Load the weight of video detokenizer from {pretrained_weight}")
    weights = torch.load(pretrained_weight, map_location='cpu')
    load_res = model.load_state_dict(weights, strict=False)
    print(load_res)
    return model
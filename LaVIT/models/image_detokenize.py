import torch
import numpy as np
import contextlib
from torch import nn, einsum
import torch.nn.functional as F
import math
import os
import random

from packaging import version
from collections import OrderedDict
from functools import partial
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from models.modeling_decoder import build_tokenizer_decoder
from models.modeling_visual_tokenzier import build_dynamic_tokenizer, VectorQuantizer

import PIL
from PIL import Image
from tqdm import tqdm
from IPython import embed


class LaVITDetokenizer(nn.Module):
    def __init__(
        self,
        model_path="",
        model_dtype="bf16",
        use_xformers=True,
        pixel_decoding='highres',
        **kwargs
    ):
        """
        Usage:
            This aims to show the detokenize result of LaVIT (from discrete token to Original Image)
            It is used to present the reconstruction fidelity.
        params:
            model_path: The pre-trained model checkpoint path, the local path for downloaded LaVIT weight
            model_dtype: The precision of model weight during inference, should be set bf16 or fp16, default is bf16.
            pixel_decoding: can be set to `highres` or `lowres`, default is `highres`: using the high resolution decoding 
                for generating high-quality images (1024 x 1024), if set to `lowres`, using the origin decoder to generate 512 x 512 image
        """
        super().__init__()

        visual_vocab_size = 16384   # The visual vocab size of LaVIT is 16384

        self.visual_tokenizer = build_dynamic_tokenizer(model_path, use_xformers=use_xformers, for_understanding=False)
        for name, param in self.visual_tokenizer.named_parameters():
            param.requires_grad = False

        self.tokenizer_decoder = build_tokenizer_decoder(model_path, pixel_decoding=pixel_decoding)
        for name, param in self.tokenizer_decoder.named_parameters():
            param.requires_grad = False
        
        img_size = 224

        # The diffusion related parameters
        self.pixel_decoding = pixel_decoding

        if pixel_decoding == 'lowres':
            diff_model_dir = os.path.join(model_path, 'pixel_decoding')
            self.register_buffer('uncond_embeddings', torch.load(os.path.join(diff_model_dir, 'uncond_embeddings.bin'), map_location='cpu'))
        else:
            diff_model_dir = os.path.join(model_path, 'highres_pixel_decoding')

        self.vae = AutoencoderKL.from_pretrained(diff_model_dir, subfolder="vae",)
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.scheduler = DDIMScheduler.from_pretrained(diff_model_dir, subfolder="scheduler")     # For evaluation

        self.unet = UNet2DConditionModel.from_pretrained(diff_model_dir, subfolder="unet", use_safetensors=False, 
                torch_dtype=torch.bfloat16 if model_dtype=="bf16" else torch.float16)

        if use_xformers:
            print("You are using XFormers ops, please make sure your device install and support xformers")
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    print(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly or set `use_xformers=False`")
        
        self.kwargs = kwargs

    @property
    def device(self):
        return self.tokenizer_decoder.pos_embed.device

    @property
    def dtype(self):
        if self.model_dtype == 'fp16':
            dtype = torch.float16
        elif self.model_dtype == 'bf16':
            dtype = torch.bfloat16
        else:
            # The default dtype is fp16
            dtype = torch.float16
        return dtype

    @torch.no_grad()
    def pre_process(self, data, process_type):
        if process_type == 'vae':
            mean = torch.as_tensor([0.5, 0.5, 0.5]).to(self.device)[None, :, None, None]
            std = torch.as_tensor([0.5, 0.5, 0.5]).to(self.device)[None, :, None, None]
        elif process_type == 'clip':
            data = F.interpolate(data, (224, 224), mode='bicubic')
            mean = torch.as_tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)[None, :, None, None]
            std = torch.as_tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)[None, :, None, None]
        else:
            raise NotImplementedError

        normed_data = (data - mean) / std

        return normed_data

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    def _get_add_time_ids(self, original_size, target_size, dtype):
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        self.add_time_ids = add_time_ids
        return add_time_ids

    @torch.no_grad()
    def reconstruct_from_token(self, x, width=1024, height=1024, original_size=None, num_inference_steps=50, guidance_scale=5.0):
        # Original_size is 
        batch_size = len(x)
        torch_device = x.device

        original_size = original_size or (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        with torch.no_grad():
            x_tensor = self.pre_process(x, process_type='clip')
            quantize, token_nums = self.visual_tokenizer.tokenize_image(x_tensor, used_for_llm=False)
            prompt_embeds, pooled_prompt_embeds = self.tokenizer_decoder(quantize, token_nums)

        # The negative prompt embeddings shall be forced to always be set to 0
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=torch_device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8), dtype=prompt_embeds.dtype,
        )
        latents = latents.to(torch_device)
        latents = latents * self.scheduler.init_noise_sigma
    
        # Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(original_size, target_size, prompt_embeds.dtype)
        negative_add_time_ids = add_time_ids

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(torch_device)
        add_text_embeds = add_text_embeds.to(torch_device)
        add_time_ids = add_time_ids.to(torch_device).repeat(batch_size, 1)

        for t in tqdm(timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()

        with torch.cuda.amp.autocast(enabled=False):
            latents = latents.to(torch.float32)
            output_image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.detach().cpu().permute(0, 2, 3, 1).numpy()
        output_images = self.numpy_to_pil(output_image)

        return output_images
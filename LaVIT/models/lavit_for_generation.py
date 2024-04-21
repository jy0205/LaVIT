import torch
import numpy as np
import contextlib
from torch import nn, einsum
import torch.nn.functional as F
import math
import os

from packaging import version
from collections import OrderedDict
from functools import partial
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from models.modeling_decoder import build_tokenizer_decoder
from models.modeling_visual_tokenzier import build_dynamic_tokenizer, VectorQuantizer
from models.transform import LaVITImageProcessor, LaVITQuestionProcessor

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import CLIPImageProcessor
from utils import get_rank
import PIL
from PIL import Image
from tqdm import tqdm


class LaVITforGeneration(nn.Module):
    def __init__(
        self,
        model_path="",
        model_dtype="bf16",
        device_id=None,
        use_xformers=False,
        check_safety=False,
        load_tokenizer=False,
        pixel_decoding='highres',
        model_sub_dir='language_model',
        **kwargs
    ):
        """
        model_path: The pre-trained model checkpoint path, the local path for downloaded LaVIT weight
        model_dtype: The precision of model weight during inference, should be set bf16 or fp16, default is bf16.
        load_tokenizer: Whether to load the tokenizer encoder during the image generation. For text-to-image generation,
        The visual tokenizer is not needed, so the default is False for saving the GPU memory. When using for the 
        multi-modal synthesis (the input image needs to be tokenizd to dircrete ids), the load_tokenizer must be set to True.
        check_safety: load the stable diffusion safety checker to check the safety of generated image, if not safe, output a black image
        pixel_decoding: can be set to `highres` or `lowres`, default is `highres`: using the high resolution decoding 
            for generating high-quality images, if set to `lowres`, using the origin decoder to generate 512 x 512 image
        """
        super().__init__()

        visual_vocab_size = 16384   # The visual vocab size of LaVIT is 16384

        print(f"Loading LaVIT Model Weight from {model_path}, model precision: {model_dtype}")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(model_path, subfolder=model_sub_dir, use_fast=False)
        self.llama_tokenizer.padding_side = "left"
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if device_id is None:
            device_map={"": get_rank() % 8}
        else:
            device_map={"": device_id}

        self.llama_model = LlamaForCausalLM.from_pretrained(
            model_path, subfolder=model_sub_dir, torch_dtype=torch.bfloat16 if model_dtype=="bf16" else torch.float16, 
            device_map=device_map,
        )
        self.model_dtype = model_dtype
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        if load_tokenizer:
            self.visual_tokenizer = build_dynamic_tokenizer(model_path, use_xformers=use_xformers, for_understanding=False)
        else:
            # For text-to-image generation, we does not load the tokenizer for saving GPU memory
            # For using multi-modal synthesis, please set load_tokenizer to True
            self.visual_tokenizer = None
            self.quantize = VectorQuantizer(n_embed=16384, embedding_dim=32)
            # Load the quantize embedding weight
            weight_path = os.path.join(model_path, 'visual_tokenizer', 'tokenizer_encoder.bin')
            state_dict = torch.load(weight_path, map_location="cpu")
            quantize_dict = OrderedDict({'embedding.weight' : state_dict['quantize.embedding.weight']})
            self.quantize.load_state_dict(quantize_dict)

        self.tokenizer_decoder = build_tokenizer_decoder(model_path, pixel_decoding=pixel_decoding)
        img_size = 224
        self.processer = LaVITImageProcessor(image_size=img_size)
        self.check_safety = check_safety

        # The diffusion related parameters
        self.pixel_decoding = pixel_decoding

        if pixel_decoding == 'lowres':
            diff_model_dir = os.path.join(model_path, 'pixel_decoding')
            self.register_buffer('uncond_embeddings', torch.load(os.path.join(diff_model_dir, 'uncond_embeddings.bin'), map_location='cpu'))
        else:
            diff_model_dir = os.path.join(model_path, 'highres_pixel_decoding')

        self.vae = AutoencoderKL.from_pretrained(diff_model_dir, subfolder="vae", 
                torch_dtype=torch.bfloat16 if model_dtype=="bf16" else torch.float16)
        self.scheduler = DDIMScheduler.from_pretrained(diff_model_dir, subfolder="scheduler")
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
        
        if check_safety:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                os.path.join(model_path, 'pixel_decoding'), subfolder="feature_extractor",
            )
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                os.path.join(model_path, 'pixel_decoding'), subfolder="safety_checker",
            )
        
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

    def maybe_autocast(self, dtype=None):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        dtype = self.dtype

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @torch.no_grad()
    def generate_image_tokenids(self, prompts, 
        use_nucleus_sampling=True, 
        top_p=0.9, 
        top_k=50,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=200,
        min_length=20,
        guidance_scale=3.0,
        uncond_input_ids=None,
        is_token_prompt=False,
    ):
        # Input is the multi_modal prompts, generate the image tokenids
        # If is_token_prompt, the input prompts are consiting of token ids
        device = self.llama_model.device
        self.llama_tokenizer.padding_side = "left"
        batch_size = len(prompts)

        if not is_token_prompt:
            prompt_tokens = self.llama_tokenizer(
                prompts, padding="longest", return_tensors="pt", add_special_tokens=False
            ).to(device)
            prompt_token_ids = prompt_tokens.input_ids
            prompt_attn_mask = prompt_tokens.attention_mask
        else:
            # The input prompts is already tokenized to IDs
            max_length = max([len(x) for x in prompts])
            prompt_token_ids = torch.ones((batch_size, max_length), dtype=torch.long).to(device) * self.llama_tokenizer.pad_token_id
            prompt_attn_mask = torch.zeros((batch_size, max_length), dtype=torch.long).to(device)
            for i in range(batch_size):
                prompt_token_ids[i, -len(prompts[i]):] = prompts[i]
                prompt_attn_mask[i, -len(prompts[i]):] = 1

        image_start_token = torch.tensor([32000], dtype=torch.long).to(device)
        image_start_token = image_start_token.expand(batch_size, -1)
        image_start_attn = torch.ones((batch_size, 1), dtype=torch.long).to(device)   # [batch_size, 1]

        prompt_token_ids = torch.cat([prompt_token_ids, image_start_token], dim=-1)
        prompt_attn_mask = torch.cat([prompt_attn_mask, image_start_attn], dim=-1)

        prompt_embeds = self.llama_model.get_input_embeddings()(prompt_token_ids)

        # Supress the text tokens
        supress_range = range(3, 32000)
        suppress_tokens = [x for x in supress_range]

        with self.maybe_autocast():
            if uncond_input_ids is None:
                outputs = self.llama_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_attn_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_new_tokens=min_length,
                    suppress_tokens=suppress_tokens,
                    bos_token_id=32000,
                    eos_token_id=32001,
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_images,
                    guidance_scale=guidance_scale,
                )
            else:
                outputs = self.llama_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_attn_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_new_tokens=min_length,
                    suppress_tokens=suppress_tokens,
                    bos_token_id=32000,
                    eos_token_id=32001,
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_images,
                    guidance_scale=guidance_scale,
                    negative_prompt_ids=uncond_input_ids,
                )
        
        return outputs

    @torch.no_grad()
    def generate_image_embeds(self, image_tokens):
        # Generate the image embeddings, that can be input to decoder to rendering pixel
        batch_size = len(image_tokens)
        tokens_prune = []; token_nums = []
        device = self.device
        
        for i_b in range(batch_size):
            image_token = image_tokens[i_b]
            image_token = image_token - 32002
            image_token = image_token[image_token >= 0]
            token_nums.append(len(image_token))
            tokens_prune.append(image_token)

        tokens_prune = torch.cat(tokens_prune, dim=0)
        token_nums = torch.as_tensor(token_nums, dtype=torch.long).to(device)
        torch_dtype = self.dtype

        if self.visual_tokenizer is None:
            token_quantize = self.quantize.embedding(tokens_prune)  # [np, d]
        else:
            token_quantize = self.visual_tokenizer.quantize.embedding(tokens_prune)  # [np, d]

        token_quantize = token_quantize.to(torch_dtype)

        return self.tokenizer_decoder(token_quantize, token_nums)

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

    def run_safety_checker(self, image_array):
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(image_array), return_tensors="pt").to(self.device)
        image_array, has_nsfw_concept = self.safety_checker(
            images=image_array, clip_input=safety_checker_input.pixel_values.to(self.dtype)
        )
        return image_array, has_nsfw_concept

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

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

    @torch.no_grad()
    def pixel_decoding_origin(self, image_tokens, 
        width=512,
        height=512,
        num_inference_steps=50, 
        guidance_scale_for_decoder=5.0,
    ):
        """
        For the origin LaVIT pixel decoder (based on SD v1.5), we have updated the pixel decoder. The new
        decoder supports to generate high resolution and aesthetics images. We strongly recommond you to use
        our new decoder for image synthesis.
        """
        with self.maybe_autocast():
            # Take the token id as input, generate the decoded embeddings
            xrec = self.generate_image_embeds(image_tokens)
            batch_size = len(xrec)
            torch_device = self.device

            # To prepare the neative condition
            _, num_tokens, C = xrec.shape
            encoder_hidden_uncond = torch.zeros(batch_size, num_tokens, C, dtype=xrec.dtype).to(torch_device)
            uncond_embeddings = self.uncond_embeddings[0].to(xrec.dtype)
            encoder_hidden_uncond[:,:len(uncond_embeddings)] = uncond_embeddings
            
            # To set the mask
            encoder_mask = torch.ones(batch_size, num_tokens, dtype=torch.long).to(torch_device)
            uncond_encoder_mask = torch.zeros(batch_size, num_tokens, dtype=torch.long).to(torch_device)
            uncond_encoder_mask[:, :len(uncond_embeddings)] = 1
            encoder_mask = encoder_mask.bool()
            uncond_encoder_mask = uncond_encoder_mask.bool()
            
            # text_embeddings, uncond_embeddings, encoder_mask, uncond_encoder_mask = self.generate_prompt_embeds(xrec)
            text_embeddings = torch.cat([encoder_hidden_uncond, xrec])
            text_embeddings_mask = torch.cat([uncond_encoder_mask, encoder_mask])
    
            latents = torch.randn(
                (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            )
            latents = latents * self.scheduler.init_noise_sigma
            latents = latents.to(torch_device)

            self.scheduler.set_timesteps(num_inference_steps, device=torch_device)

            for t in tqdm(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, 
                        encoder_attention_mask=text_embeddings_mask).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale_for_decoder * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            latents = latents / self.vae.config.scaling_factor
            output_image = self.vae.decode(latents).sample

        output_image = output_image.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.detach().cpu().permute(0, 2, 3, 1).numpy()

        if self.check_safety:
            output_image, _ = self.run_safety_checker(output_image)
        
        output_images = self.numpy_to_pil(output_image)

        return output_images

    @torch.no_grad()
    def generate_image(self, prompts, 
        width=512,
        height=512,
        original_size=None,
        crops_coords_top_left=None,
        use_nucleus_sampling=True, 
        top_p=1.0, 
        top_k=50,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=200,
        min_length=20,
        num_inference_steps=25, 
        guidance_scale_for_llm=4.0,
        guidance_scale_for_decoder=7.0,
        uncond_input_ids=None,
        is_token_prompt=False,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        original_size = original_size or (height, width)
        target_size = (height, width)
        crops_coords_top_left = crops_coords_top_left or (0, 0)

        image_tokens = self.generate_image_tokenids(
            prompts, use_nucleus_sampling, top_p, top_k, num_beams, temperature, num_return_images,
            length_penalty, max_length, min_length, guidance_scale_for_llm, uncond_input_ids, is_token_prompt,
        )

        if self.pixel_decoding == 'lowres':
            return self.pixel_decoding_origin(image_tokens, width, height, num_inference_steps, guidance_scale_for_decoder)

        # Perform pixel decoding from tokenids to RGB pixel values
        with self.maybe_autocast():
            # Take the token id as input, generate the decoded embeddings
            # The negative prompt embeddings shall be forced to always be set to 0
            prompt_embeds, pooled_prompt_embeds = self.generate_image_embeds(image_tokens)
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

            batch_size = len(prompt_embeds)
            torch_device = self.device

            latents = torch.randn(
                (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            )
            latents = latents.to(torch_device)
            latents = latents * self.scheduler.init_noise_sigma

            self.scheduler.set_timesteps(num_inference_steps, device=torch_device)

            # Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, prompt_embeds.dtype)
            add_time_ids = add_time_ids.repeat(batch_size, 1)
            negative_add_time_ids = add_time_ids

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(torch_device)
            add_text_embeds = add_text_embeds.to(torch_device)
            add_time_ids = add_time_ids.to(torch_device)

            for t in tqdm(self.scheduler.timesteps):
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
                noise_pred = noise_pred_uncond + guidance_scale_for_decoder * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
        
        with torch.cuda.amp.autocast(enabled=False):
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            output_image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        output_image = output_image.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.detach().cpu().permute(0, 2, 3, 1).numpy()

        if self.check_safety:
            output_image, _ = self.run_safety_checker(output_image)
        
        output_images = self.numpy_to_pil(output_image)

        return output_images

    @torch.no_grad()
    def multimodal_synthesis(self, prompts, 
        width=512,
        height=512,
        original_size=None,
        crops_coords_top_left=None,
        use_nucleus_sampling=True, 
        top_p=1.0, 
        top_k=50,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=200,
        min_length=20,
        num_inference_steps=25, 
        guidance_scale_for_llm=5.0,
        guidance_scale_for_decoder=7.0,
        uncond_input_ids=None,
    ):
        # The multi-modal propmts with format:
        # Image+Text and Image+Image
        # prompts: [(img_path, 'image') or (text, 'text')]
        # Now the code only supports: batchsize=1
        assert self.visual_tokenizer is not None, \
            "For multi-modal image synthesis, please set the `load_tokenizer` to True in the `build_model` method"
        input_prompts = []
        device = self.device

        for prompt_str, prompt_type in prompts:
            assert prompt_type in ['image', 'text'], "The prompt type should be image or text"
            if prompt_type == 'text':
                text_tokens = self.llama_tokenizer(
                    [prompt_str], padding="longest", return_tensors="pt", add_special_tokens=False
                ).to(device).input_ids[0]
                input_prompts.append(text_tokens)
            if prompt_type == 'image':
                image_input = Image.open(prompt_str).convert("RGB")
                image_tensor = self.processer(image_input).unsqueeze(0).to(device)
                with self.maybe_autocast():
                    image_tokens = self.visual_tokenizer.tokenize_image(image_tensor, add_special=False)[0]
                input_prompts.append(image_tokens)

        input_prompts = [torch.cat(input_prompts, dim=0)]

        output_images = self.generate_image(
            input_prompts, 
            width=width,
            height=height,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            use_nucleus_sampling=use_nucleus_sampling, 
            top_p=top_p, 
            top_k=top_k,
            num_beams=num_beams,
            temperature=temperature,
            num_return_images=num_return_images,
            length_penalty=length_penalty,
            max_length=max_length,
            min_length=min_length,
            num_inference_steps=num_inference_steps, 
            guidance_scale_for_llm=guidance_scale_for_llm,
            guidance_scale_for_decoder=guidance_scale_for_decoder,
            uncond_input_ids=uncond_input_ids,
            is_token_prompt=True,
        )

        return output_images

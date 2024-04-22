import torch
import numpy as np
import contextlib
from torch import nn, einsum
import torch.nn.functional as F
import math
import os
import json

from packaging import version
from collections import OrderedDict
from functools import partial
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from .modeling_decoder import build_tokenizer_decoder
from .modeling_visual_tokenzier import build_dynamic_tokenizer, VectorQuantizer
from .transform import LaVITImageProcessor
from .video_detokenizer import build_video_detokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import CLIPImageProcessor
from utils import get_rank
from torchvision.transforms.functional import to_tensor
import PIL
from PIL import Image
from tqdm import tqdm


class VideoLaVITforGeneration(nn.Module):
    def __init__(
        self,
        model_path="",
        model_dtype="bf16",
        device_id=None,
        use_xformers=False,
        visual_vocab_size=16384,
        motion_vocab_size=1026,
        model_sub_dir='language_model',
        **kwargs
    ):
        """
        model_path: The pre-trained model checkpoint path, the local path for downloaded LaVIT weight
        model_dtype: The precision of model weight during inference, should be set bf16 or fp16, default is bf16.
        """
        super().__init__()
        self.visual_vocab_size = visual_vocab_size
        self.motion_vocab_size = motion_vocab_size
        self.motion_token_length = 135  # Each video clip has 135 token

        # logging.info(f'Loading LLAMA Model from {llama_model}')
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

        self.visual_tokenizer = build_dynamic_tokenizer(model_path, use_xformers=use_xformers, for_understanding=False)
        self.tokenizer_decoder = build_tokenizer_decoder(model_path)
        img_size = 224
        self.processer = LaVITImageProcessor(image_size=img_size)

        # The keyframe / image detokenizer
        image_detokenizer_dir = os.path.join(model_path, 'image_detokenizer')
        # The keyframe U-net
        self.vae = AutoencoderKL.from_pretrained(image_detokenizer_dir, subfolder="vae", 
                torch_dtype=torch.bfloat16 if model_dtype=="bf16" else torch.float16)
    
        scheduler_config = json.load(open(os.path.join(image_detokenizer_dir, "scheduler", 'scheduler_config.json'), 'r'))
        if scheduler_config['_class_name'] == 'EulerAncestralDiscreteScheduler':
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(image_detokenizer_dir, subfolder="scheduler")
        else:
            self.scheduler = DDIMScheduler.from_pretrained(image_detokenizer_dir, subfolder="scheduler")

        self.unet = UNet2DConditionModel.from_pretrained(image_detokenizer_dir, subfolder="unet", use_safetensors=False, 
                torch_dtype=torch.bfloat16 if model_dtype=="bf16" else torch.float16)

        # The video detokenizer
        self.video_decoder = build_video_detokenizer(model_path, model_dtype=model_dtype, use_xformer=use_xformers,)
        for param in self.video_decoder.parameters():
            param.requires_grad = False 

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
        max_new_tokens=200,
        min_new_tokens=20,
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
            # The text input
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

        # Supress the text tokens
        supress_range_text = range(3, 32000)
        supress_range_motion = range(48386, 48386 + self.motion_vocab_size)
        suppress_tokens = [x for x in supress_range_text] + [x for x in supress_range_motion]

        with self.maybe_autocast():
            if uncond_input_ids is None:
                outputs = self.llama_model.generate(
                    prompt_token_ids,
                    attention_mask=prompt_attn_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
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
                    prompt_token_ids,
                    attention_mask=prompt_attn_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    suppress_tokens=suppress_tokens,
                    bos_token_id=32000,
                    eos_token_id=32001,
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_images,
                    guidance_scale=guidance_scale,
                    negative_prompt_ids=uncond_input_ids,
                )
        
        # Repad the outputs ids to left pad
        output_batch_size = len(outputs)
        input_token_len = prompt_token_ids.shape[1]
        n_diff_input_output = (prompt_token_ids != outputs[:, :input_token_len]).sum().item()

        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )

        output_list = []; pure_image_list = []
        pure_image_tokens = outputs[:, input_token_len:]

        for i_b in range(output_batch_size):
            output_ids = outputs[i_b]
            pure_image_ids = pure_image_tokens[i_b]
            output_ids = output_ids[output_ids != self.llama_tokenizer.pad_token_id]
            pure_image_ids = pure_image_ids[pure_image_ids != self.llama_tokenizer.pad_token_id]
            output_list.append(output_ids)
            pure_image_list.append(pure_image_ids)

        return output_list, pure_image_list

    @torch.no_grad()
    def generate_motion_tokenids(self, prompts, 
        use_nucleus_sampling=True, 
        top_p=0.9, 
        top_k=50,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_new_tokens=300,
        min_new_tokens=20,
        guidance_scale=3.0,
        uncond_input_ids=None,
        is_token_prompt=False,
    ):
        device = self.llama_model.device
        batch_size = len(prompts)

        # The input prompts is already tokenized to IDs
        max_length = max([len(x) for x in prompts])
        prompt_token_ids = torch.ones((batch_size, max_length), dtype=torch.long).to(device) * self.llama_tokenizer.pad_token_id
        prompt_attn_mask = torch.zeros((batch_size, max_length), dtype=torch.long).to(device)
        for i in range(batch_size):
            prompt_token_ids[i, -len(prompts[i]):] = prompts[i]
            prompt_attn_mask[i, -len(prompts[i]):] = 1

        motion_start_token = torch.tensor([48386], dtype=torch.long).to(device)
        motion_start_token = motion_start_token.expand(batch_size, -1)
        motion_start_attn = torch.ones((batch_size, 1), dtype=torch.long).to(device)   # [batch_size, 1]

        prompt_token_ids = torch.cat([prompt_token_ids, motion_start_token], dim=-1)
        prompt_attn_mask = torch.cat([prompt_attn_mask, motion_start_attn], dim=-1)

        # Supress the text tokens
        supress_range = range(3, 48386)
        suppress_tokens = [x for x in supress_range]

        with self.maybe_autocast():
            if uncond_input_ids is None:
                outputs = self.llama_model.generate(
                    prompt_token_ids,
                    attention_mask=prompt_attn_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    suppress_tokens=suppress_tokens,
                    bos_token_id=48386,
                    eos_token_id=48387,
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_images,
                    guidance_scale=guidance_scale,
                )
            else:
                outputs = self.llama_model.generate(
                    prompt_token_ids,
                    attention_mask=prompt_attn_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    suppress_tokens=suppress_tokens,
                    bos_token_id=48386,
                    eos_token_id=48387,
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_images,
                    guidance_scale=guidance_scale,
                    negative_prompt_ids=uncond_input_ids,
                )

        output_batch_size = len(outputs)
        input_token_len = prompt_token_ids.shape[1]
        n_diff_input_output = (prompt_token_ids != outputs[:, :input_token_len]).sum().item()

        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )

        output_list = []; pure_motion_list = []
        pure_motion_tokens = outputs[:, input_token_len:]

        for i_b in range(output_batch_size):
            output_ids = outputs[i_b]
            pure_motion_ids = pure_motion_tokens[i_b]
            output_ids = output_ids[output_ids != self.llama_tokenizer.pad_token_id]
            pure_motion_ids = pure_motion_ids[pure_motion_ids != self.llama_tokenizer.pad_token_id]
            output_list.append(output_ids)
            pure_motion_list.append(pure_motion_ids)

        return output_list, pure_motion_list

    @torch.no_grad()
    def generate_image_embeds(self, image_tokens):
        # Transfer the discrete token to continous image embeddings, that can be input to decoder to rendering pixel
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


    def pre_process(self, key_frames, process_type):
        key_frame_tensors = torch.stack([to_tensor(key_frame) for key_frame in key_frames], dim=0)
        key_frame_tensors = key_frame_tensors.to(self.device)
        
        if process_type == 'default':
            mean = torch.as_tensor([0.5, 0.5, 0.5]).to(self.device)[None, :, None, None]
            std = torch.as_tensor([0.5, 0.5, 0.5]).to(self.device)[None, :, None, None]
        else:
            raise NotImplementedError

        normed_data = (key_frame_tensors - mean) / std

        return normed_data

    def get_image_latents(self, x):
        # Use fp32 to avoid overflow
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()

        if needs_upcasting:
            with torch.cuda.amp.autocast(enabled=False):
                x_hidden = self.pre_process(x, process_type='default')
                latents = self.vae.encode(x_hidden).latent_dist.mode()
                latents = latents * self.vae.config.scaling_factor
        else:
            x_hidden = self.pre_process(x, process_type='default')
            latents = self.vae.encode(x_hidden).latent_dist.mode()
            latents = latents * self.vae.config.scaling_factor

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        return latents

    @torch.no_grad()
    def regenerate_noise(self, last_frame, num_inference_steps, width, height, noise_level):
        # Remap the reframe to noise
        device = self.device
        batch_size = len(last_frame)
        # Prepare latent variables
        last_frame = [frame.resize((width, height), PIL.Image.BICUBIC) for frame in last_frame]
        latents = self.get_image_latents(last_frame)
        noise = torch.randn_like(latents)    # the adding noise to latents
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        noise_step = self.scheduler.timesteps[noise_level].item()
        add_time_steps = torch.tensor([noise_step] * batch_size, dtype=torch.long).to(device)
        latents = self.scheduler.add_noise(latents, noise, add_time_steps)
        latents = latents * self.scheduler.init_noise_sigma
        latents = latents.to(device)
        return latents

    @torch.no_grad()
    def multimodal_video_generate(self,
        prompts, 
        width=1024,
        height=1024,
        video_width=None,
        video_height=None,
        original_size=None,
        crops_coords_top_left=None,
        use_nucleus_sampling=True, 
        top_p=1.0, 
        top_k=200,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=500,
        min_length=20,
        num_inference_steps=50, 
        guidance_scale_for_llm=4.0,
        guidance_scale_for_motion=7.0,
        guidance_scale_for_decoder=3.0,
        uncond_input_ids=None,
        clip_num=1,
        cond_on_ref_frame=True, 
        use_linear_guidance=True, 
        noise_aug_strength=0.02, 
        decode_chunk_size=8,
        inverse_rate=0.85,
        is_token_prompt=False,
    ):
        # The multi-modal propmts with format, (image+text-> video; image->video)
        # Image+Text and Image
        # prompts: [(img_path, 'image') or (text, 'text')]
        # Now the code only supports: batchsize=1
        input_prompts = []
        ref_images = []
        device = self.device

        for prompt_str, prompt_type in prompts:
            assert prompt_type in ['image', 'video', 'text'], "The prompt type should be image or video"
            if prompt_type == 'image':
                if isinstance(prompt_str, str):
                    image_input = Image.open(prompt_str).convert("RGB")
                else:
                    image_input = prompt_str
                ref_images.append(image_input)
                image_tensor = self.processer(image_input).unsqueeze(0).to(device)
                with self.maybe_autocast():
                    image_tokens = self.visual_tokenizer.tokenize_image(image_tensor, add_special=True)[0]
                input_prompts.append(image_tokens)

            if prompt_type == 'text':
                text_tokens = self.llama_tokenizer(
                    [prompt_str], padding="longest", return_tensors="pt", add_special_tokens=False
                ).to(device).input_ids[0]
                input_prompts.append(text_tokens)

        input_prompts = [torch.cat(input_prompts, dim=0)]

        return self.generate_video(input_prompts, width, height, video_width, video_height, original_size, crops_coords_top_left, 
            use_nucleus_sampling, top_p, top_k, num_beams, temperature, num_return_images, length_penalty, max_length, min_length, num_inference_steps, 
            guidance_scale_for_llm, guidance_scale_for_motion, guidance_scale_for_decoder, uncond_input_ids, clip_num, cond_on_ref_frame, use_linear_guidance, 
            noise_aug_strength, decode_chunk_size, inverse_rate, is_token_prompt=True, ref_image=ref_images if len(ref_images) != 0 else None,
        )

    @torch.no_grad()
    def generate_video(self, prompts, 
        width=1024,
        height=1024,
        video_width=None,
        video_height=None,
        original_size=None,
        crops_coords_top_left=None,
        use_nucleus_sampling=True, 
        top_p=1.0, 
        top_k=200,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=500,
        min_length=20,
        num_inference_steps=50, 
        guidance_scale_for_llm=4.0,
        guidance_scale_for_motion=7.0,
        guidance_scale_for_decoder=3.0,
        uncond_input_ids=None,
        clip_num=1,
        cond_on_ref_frame=True, 
        use_linear_guidance=True, 
        noise_aug_strength=0.02, 
        decode_chunk_size=8,
        inverse_rate=0.85,
        is_token_prompt=False,
        ref_image=None,
    ):
        # Now only supoort the text-to-video
        if isinstance(prompts, str):
            prompts = [prompts]

        # assert clip_num == 1, "Now only support generate one clip"
        original_size = original_size or (height, width)
        target_size = (height, width)
        crops_coords_top_left = crops_coords_top_left or (0, 0)

        video_width = video_width or width
        video_height = video_height or height

        whole_videos = [[] * (len(prompts) * num_return_images)]
        whole_key_frames = [[] * (len(prompts) * num_return_images)]
        last_frames = None
        noise_level = int(inverse_rate * num_inference_steps)

        # Generate tokens clip by clip
        for turn in range(clip_num):
            is_tokens = True
            if turn == 0:
                is_tokens = is_token_prompt
            else:
                is_tokens = True

            if ref_image is None:
                # text to video
                output_tokens, pure_image_tokens = self.generate_image_tokenids(
                    prompts, use_nucleus_sampling, top_p, top_k, num_beams, temperature, num_return_images,
                    length_penalty, max_length, min_length, guidance_scale_for_llm, uncond_input_ids, is_token_prompt=is_tokens,
                )
                output_tokens, pure_motion_tokens = self.generate_motion_tokenids(
                    output_tokens, False, top_p, top_k, num_beams, temperature, num_return_images,
                    length_penalty, max_length, min_length, guidance_scale_for_motion, uncond_input_ids, is_token_prompt=True,
                )
            else:
                # image to video, only generate motion tokens
                output_tokens, pure_motion_tokens = self.generate_motion_tokenids(
                    prompts, use_nucleus_sampling, top_p, top_k, num_beams, temperature, num_return_images,
                    length_penalty, max_length, min_length, guidance_scale_for_motion, uncond_input_ids, is_token_prompt=True,
                )

            prompts = output_tokens    # The motion tokens already have image tokens + motion tokens
            # Decoding the images and videos
            if ref_image is None:
                if turn == 0:
                    noise_used = None
                else:
                    noise_used = self.regenerate_noise(last_frames, num_inference_steps, width, height, noise_level)
                
                images = self.image_decoding(pure_image_tokens, width, height, num_inference_steps, guidance_scale_for_decoder, 
                        original_size, target_size, crops_coords_top_left, noise_used, 0 if turn == 0 else noise_level)
            else:
                images = ref_image

            if video_width != width or video_height != height:
                key_frames = [image.resize((video_width, video_height), PIL.Image.BICUBIC) for image in images]
            else:
                key_frames = images

            # Since the ref image is used, set is to None
            ref_image = None

            videos = self.video_decoding(key_frames, pure_motion_tokens, video_width, video_height, cond_on_ref_frame, 
                        use_linear_guidance, noise_aug_strength, decode_chunk_size)

            last_frames = [video[-1] for video in videos]

            for i_b in range(len(videos)):
                whole_videos[i_b].extend(videos[i_b])
                whole_key_frames[i_b].append(key_frames[i_b])

        return whole_videos, whole_key_frames

    @torch.no_grad()
    def video_decoding(self, key_frames, motion_tokens, width, height, cond_on_ref_frame=True, use_linear_guidance=True, 
                noise_aug_strength=0.02, decode_chunk_size=None,):
        # Generate the image embeddings, that can be input to decoder to rendering pixel
        batch_size = len(key_frames)
        tokens_prune = []
        device = self.device
        
        for i_b in range(batch_size):
            motion_token = motion_tokens[i_b]
            motion_token = motion_token[motion_token > 48387]
            
            if len(motion_token) != self.motion_token_length:
                print("MOTION TOKEN NOT EQUAL SETTING:", len(motion_token))
                if len(motion_token) >= self.motion_token_length:
                    motion_token = motion_token[:self.motion_token_length]
                else:
                    pad_length = self.motion_token_length - len(motion_token)
                    pad_token = torch.tensor([48388] * pad_length, dtype=torch.long).to(motion_token.device)
                    motion_token = torch.cat([motion_token, pad_token], dim=0)

            assert len(motion_token) == self.motion_token_length
            motion_token = motion_token - 48388
            tokens_prune.append(motion_token)

        motion_tokens_prune = torch.stack(tokens_prune, dim=0)
        
        # Deal with key_frames
        key_frame_tensors = torch.stack([to_tensor(key_frame) for key_frame in key_frames], dim=0)
        key_frame_tensors = key_frame_tensors.to(device)
        key_frame_tensors = 2.0 * key_frame_tensors - 1.0

        frames = self.video_decoder.reconstruct_from_token(key_frame_tensors, motion_tokens_prune, decode_chunk_size=8, 
                        width=width, height=height, num_frames=24, decode_from_token=True, noise_aug_strength=noise_aug_strength, 
                        cond_on_ref_frame=cond_on_ref_frame, use_linear_guidance=use_linear_guidance)

        return frames

    @torch.no_grad()
    def image_decoding(self, image_tokens,
        width=512,
        height=512,
        num_inference_steps=50, 
        guidance_scale_for_decoder=3.0,
        original_size=None,
        target_size=None,
        crops_coords_top_left=None,
        noise_used=None,
        noise_level=0,
    ):
        # Perform pixel decoding from tokenids to RGB pixel values
        with self.maybe_autocast():
            # Take the token id as input, generate the decoded embeddings
            # The negative prompt embeddings shall be forced to always be set to 0
            prompt_embeds, pooled_prompt_embeds = self.generate_image_embeds(image_tokens)
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

            batch_size = len(prompt_embeds)
            torch_device = self.device

            if noise_used is None:
                latents = torch.randn(
                    (batch_size, self.unet.config.in_channels, height // 8, width // 8),
                )
                latents = latents.to(torch_device)
                latents = latents * self.scheduler.init_noise_sigma
            else:
                latents = noise_used

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

            for t in tqdm(self.scheduler.timesteps[noise_level:]):
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
        top_k=200,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=200,
        min_length=20,
        num_inference_steps=50, 
        guidance_scale_for_llm=4.0,
        guidance_scale_for_decoder=3.0,
        uncond_input_ids=None,
        is_token_prompt=False,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        original_size = original_size or (height, width)
        target_size = (height, width)
        crops_coords_top_left = crops_coords_top_left or (0, 0)

        _, image_tokens = self.generate_image_tokenids(
            prompts, use_nucleus_sampling, top_p, top_k, num_beams, temperature, num_return_images,
            length_penalty, max_length, min_length, guidance_scale_for_llm, uncond_input_ids, is_token_prompt,
        )

        images = self.image_decoding(image_tokens, width, height, num_inference_steps, 
                guidance_scale_for_decoder, original_size, target_size, crops_coords_top_left)
        return images


    @torch.no_grad()
    def multimodal_synthesis(self, prompts, 
        width=512,
        height=512,
        original_size=None,
        crops_coords_top_left=None,
        use_nucleus_sampling=True, 
        top_p=1.0, 
        top_k=200,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=200,
        min_length=20,
        num_inference_steps=50, 
        guidance_scale_for_llm=5.0,
        guidance_scale_for_decoder=3.0,
        uncond_input_ids=None,
    ):
        # The multi-modal propmts with format:
        # Image+Text and Image+Image
        # prompts: [(img_path, 'image') or (text, 'text')]
        # Now the code only supports: batchsize=1
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

        output_images = self.generate_image(input_prompts, width, height, original_size, crops_coords_top_left, use_nucleus_sampling, 
            top_p, top_k, num_beams, temperature, num_return_images, length_penalty, max_length, min_length, num_inference_steps, 
            guidance_scale_for_llm, guidance_scale_for_decoder, uncond_input_ids, is_token_prompt=True,
        )
        return output_images

import logging
import contextlib
import os
import re
import random
import math
import numpy as np

import transformers
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils import get_rank, KeywordsStoppingCriteria
from conversation import default_conversation, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, VIDEO_TOKEN_INDEX, IMAGE_TOKEN_INDEX
from models.modeling_video_lavit_hf import VideoLaVITLlamaForCausalLM
from models.transform import LaVITImageProcessor, LaVITEvalVideoProcessor
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, image_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(image_token)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


class VideoLaVITUnderstandingRunner:
    """
    The LaVIT Model for Multi-modal Understanding, 
    this file is used for reading image contents and answering the questions.
    """
    def __init__(
        self,
        img_size=224,
        model_path="",
        model_dtype="bf16",
        device_id=None,
        apply_lemmatizer=True,
        use_xformers=False,
        max_frames=24,
        max_clips=8,
        motion_vocab_size=1026,
    ):
        """
        img_size: The input image size, should be 224 * 224
        model_path: The pre-trained model checkpoint path, the local path for downloaded LaVIT weight
        model_dtype: The precision of model weight during inference, should be set bf16 or fp16, default is bf16.
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas
        """
        super().__init__()
        assert img_size == 224, "Input Image Size should be set to 224"
        visual_vocab_size = 16384   # The visual vocab size of LaVIT is 16384
        assert max_frames == 24, "One video clip should have 24 frames"

        self.motion_vocab_size = motion_vocab_size
        print(f"Loading Video LaVIT Model Weight from {model_path}, model precision: {model_dtype}")

        if device_id is None:
            device_map={"": get_rank() % 8}
        else:
            device_map={"": device_id}

        config = transformers.AutoConfig.from_pretrained(model_path)
        config.use_xformers = use_xformers
        # For inference, we should use the left padding
        config.tokenizer_padding_side = 'left'
        config.use_cache = True

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path, use_fast=False, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.model = VideoLaVITLlamaForCausalLM.from_pretrained(model_path, config=config, device_map=device_map,
                torch_dtype=torch.bfloat16 if model_dtype=="bf16" else torch.float16,)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.model.eval()

        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.visual_vocab_size = visual_vocab_size
        print(f"The Visual Vocab Size is {self.visual_vocab_size}")
        print(f"The llama tokenizer vocab size is {len(self.tokenizer)}")
        print(f"The maximal clip number is {max_clips}")

        self.model_dtype = model_dtype
        self.apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.image_processer = LaVITImageProcessor(image_size=img_size)
        self.video_processor = LaVITEvalVideoProcessor(image_size=img_size, num_frames=max_frames, fps=6, max_clips=max_clips,)

    @property
    def device(self):
        return self.model.device

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

    def process_image(self, image_inputs, num_beams=1):
        if isinstance(image_inputs, torch.Tensor):
            assert len(image_inputs.shape) == 4, "Image Tensors should have shape (batch_size, 3, H, W)"
            
            image_inputs_list = []
            for i in range(len(image_inputs)):
                for _ in range(num_beams):
                    image_inputs_list.append(((image_inputs[i:i+1],), 'image'))
        
            return image_inputs_list

        if not isinstance(image_inputs, list):
            assert isinstance(image_inputs, str)
            image_inputs = [image_inputs]
        
        image_inputs_list = []
        for image_path in image_inputs:
            image = Image.open(image_path).convert('RGB') 
            image = self.image_processer(image)
            for _ in range(num_beams):
                image_inputs_list.append(((image.unsqueeze(0),), 'image'))

        return image_inputs_list

    def process_video(self, video_inputs, num_beams=1):
        if not isinstance(video_inputs, list):
            if not isinstance(video_inputs, str):
                raise ValueError("The video_inputs should be Tensors or Str (video path)")
            video_inputs = [video_inputs]

        if isinstance(video_inputs[0], list):
            assert isinstance(video_inputs[0][0], torch.Tensor)
            video_inputs_list = []
            for video_input in video_inputs:
                for _ in range(num_beams):
                    video_inputs_list.append(((video_input[0], video_input[1]), 'video'))
            return video_inputs_list
        
        if isinstance(video_inputs[0], str):
            video_inputs_list = []
            for video_path in video_inputs:
                visual_inputs, motion_inputs = self.video_processor(video_path)
                for _ in range(num_beams):
                    video_inputs_list.append(((visual_inputs, motion_inputs), 'video'))

            return video_inputs_list

        raise ValueError("Unsupported data input format")

    @torch.no_grad()
    def __call__(
        self,
        samples,
        use_nucleus_sampling=True,
        num_beams=1,
        max_length=512,
        min_length=8,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1,
        length_penalty=1,
        num_return_sequences=1,
        temperature=1,
        truct_vqa=False,
        stop_str='\\n',
        lemmatize=False,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]        

        if 'image' in samples or 'video' in samples:
            if 'image' in samples:
                visual_inputs = self.process_image(samples['image'], num_beams)
                DEFAULT_VISUAL_TOKEN = DEFAULT_IMAGE_TOKEN
                VISUAL_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            else:
                visual_inputs = self.process_video(samples['video'], num_beams)
                DEFAULT_VISUAL_TOKEN = DEFAULT_VIDEO_TOKEN
                VISUAL_TOKEN_INDEX = VIDEO_TOKEN_INDEX

            text_inputs = [DEFAULT_VISUAL_TOKEN + "\n" + text_input for text_input in samples["text_input"]]

        else:
            DEFAULT_VISUAL_TOKEN = DEFAULT_IMAGE_TOKEN
            VISUAL_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            visual_inputs = None
            text_inputs = [text_input for text_input in samples["text_input"]]
        
        # To get the input prompts
        prompts = []
        for text_input in text_inputs:
            conv = default_conversation.copy()
            conv.append_message(conv.roles[0], text_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)

        # print(prompts)
        input_ids = [tokenizer_image_token(prompt, self.tokenizer, VISUAL_TOKEN_INDEX, \
                DEFAULT_VISUAL_TOKEN, return_tensors='pt') for prompt in prompts]

        # print("input_ids:", input_ids)
        # To pad the token
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.device)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # For understanding, supress the token ids > 32000 (Visual Tokens)
        supress_range = 32000 + self.visual_vocab_size + self.motion_vocab_size + 2
        suppress_tokens = [x for x in range(32000, supress_range)]

        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=visual_inputs,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                temperature=temperature,
                num_beams=num_beams,
                min_new_tokens=min_length,
                max_new_tokens=max_length,
                top_p=top_p,
                use_cache=True,
                suppress_tokens=suppress_tokens,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_return_sequences,
                stopping_criteria=[stopping_criteria],
            )

        # print("output_ids:", output_ids)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )
        outputs = [output.split('\\n')[0] for output in outputs]
        outputs = [output.strip() for output in outputs]

        if truct_vqa:
            # For VQA, lower the output answer and lemmatize for evaluation
            outputs = [text.split('\n')[0] for text in outputs]
            outputs = [text.split('.')[0] for text in outputs]
            outputs = [text.split('!')[0] for text in outputs]
            outputs = [text.split(',')[0] for text in outputs]
            outputs = [text.split('?')[0] for text in outputs]
            outputs = [output.strip() for output in outputs]

        if lemmatize:
            # To lemmatize the answers
            outputs = self._lemmatize(outputs)
        
        return outputs

    def _lemmatize(self, answers):
        def apply(answer):
            answer = answer.lower()
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy
                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
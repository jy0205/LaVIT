import logging
import contextlib
import os
import re
import random
import math
import numpy as np

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from PIL import Image
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from models.modeling_visual_tokenzier import DynamicVisualTokenizer
from models.modeling_motion_tokenizer import build_motion_tokenizer
from conversation import IGNORE_INDEX, IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX


class VideoLaVITLlamaModel(LlamaModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super(VideoLaVITLlamaModel, self).__init__(config) 
        use_xformers = config.use_xformers
        self.visual_tokenizer = DynamicVisualTokenizer(use_xformers=use_xformers) # The weight are not loaded
        self.motion_tokenizer = build_motion_tokenizer(as_tokenizer=False)


class VideoLaVITLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlamaConfig

    def __init__(self, config):
        super(VideoLaVITLlamaForCausalLM, self).__init__(config)
        self.model = VideoLaVITLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.model.visual_tokenizer

    def get_motion_tower(self):
        return self.model.motion_tokenizer

    def compute_motion_embeds(self, motions):
        # motion: [n_clip, 2, T, H, W]
        with torch.no_grad():
            motion_tokens = self.get_motion_tower().get_tokens(motions)['token']  # [n_clip, 256]

        batch_size = len(motion_tokens)
        motion_tokens = motion_tokens + 32000 + 16384 + 4
        motion_pad_token = torch.tensor([[48386, 48387]], dtype=torch.long).to(motions.device).repeat(batch_size, 1)
        motion_tokens = torch.cat([motion_pad_token[:,:1], motion_tokens, motion_pad_token[:, 1:]], dim=-1)
        motion_embeds = self.get_model().embed_tokens(motion_tokens)
        return motion_embeds

    def encode_visual_features(self, images):
        batch_size = len(images)
        visual_input_tensor, motion_input_tensor, return_pad_list, visual_index_list, split_sizes, input_types = [], [], [], [], [],[]
    
        for idx, ele in enumerate(images):
            if ele[1] == 'image':
                assert len(ele[0]) == 1, "Image only has visual tensor"
                input_tensor = ele[0][0]
                split_sizes.append(input_tensor.shape[0])
                visual_index_list.append(idx)
                return_pad_list.append(None)
                visual_input_tensor.append(input_tensor)
                input_types.append('image')
            
            elif ele[1] == 'video':
                assert len(ele[0]) == 2, "Video should has visual tensor and motion vector input"
                visual_input = ele[0][0]
                motion_input = ele[0][1]
                assert visual_input.shape[0] == motion_input.shape[0]
                split_sizes.append(visual_input.shape[0])
                visual_index_list.append(idx)
                return_pad_list.append(None)
                visual_input_tensor.append(visual_input)
                motion_input_tensor.append(motion_input)
                input_types.append('video')

            else:
                # For pure text sequence, add a dummy tensor
                return_pad_list.append(torch.zeros(2, 4096, dtype=self.dtype).to(self.device))

        # The whole batchsize are text
        if len(visual_input_tensor) == 0:
            return return_pad_list

        visual_input_tensor = torch.cat(visual_input_tensor, dim=0)
        visual_input_tensor = visual_input_tensor.to(self.device, self.dtype)

        # calculate the visual embeds list
        visual_embeds_list = self.get_model().visual_tokenizer.encode_features(visual_input_tensor)  # [num_images]
        
        # Pad them with image-start and image-end token
        image_pad_token = torch.tensor([32000, 32001], dtype=torch.long).to(self.device)
        image_pad_embeds = self.get_model().embed_tokens(image_pad_token) # [2, embed_dim]
        for i_b in range(len(visual_embeds_list)):
            visual_embeds_list[i_b] = torch.cat([image_pad_embeds[:1], visual_embeds_list[i_b], image_pad_embeds[1:]], dim=0)
        
        motion_embeds = None
        # calculate the motion embeds
        if len(motion_input_tensor) != 0:
            motion_input_tensor = torch.cat(motion_input_tensor, dim=0)
            motion_input_tensor = motion_input_tensor.to(self.device, self.dtype)
            motion_embeds = self.compute_motion_embeds(motion_input_tensor)

        # Concat the video features in one sequence
        visual_embeds_list_merge = []
        cur_sum = 0
        motion_cur_sum = 0
        for idx in range(len(split_sizes)):
            num_images = split_sizes[idx]
            input_type = input_types[idx]
            feature_seq = visual_embeds_list[cur_sum : cur_sum + num_images]
            assert return_pad_list[visual_index_list[idx]] is None
            if input_type == 'video':
                motion_seq = motion_embeds[motion_cur_sum: motion_cur_sum + num_images]
                input_seq = []
                for i_clip in range(len(feature_seq)):
                    input_seq.extend([feature_seq[i_clip], motion_seq[i_clip]])
                    # input_seq.append(feature_seq[i_clip])
                feature_seq = torch.cat(input_seq, dim=0)
                motion_cur_sum += num_images
            else:
                feature_seq = torch.cat(feature_seq, dim=0)
            cur_sum += num_images
            return_pad_list[visual_index_list[idx]] = feature_seq

        return return_pad_list

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        # Why?? This code is only used in generate
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_visual_features(images)   # [List of features: Tensor:[Seq_len, embed_dim (4096)]]
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            seq_type = images[batch_idx][1]
            VISUAL_TOKEN_INDEX = IMAGE_TOKEN_INDEX if seq_type == 'image' else VIDEO_TOKEN_INDEX
            num_images = (cur_input_ids == VISUAL_TOKEN_INDEX).sum()
            if num_images == 0:
                assert seq_type == 'text'
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == VISUAL_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    # The visual part does not calculate loss
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[List[Tuple]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # The images is the visual content that: [(image_tensor, 'image'), (video_tensor, 'video'), (dummy_tensor, 'text') ...]
        # TODO Now only support one image or video per sequence.
        # TODO Now only support the batchsize =1

        # To substitute the video and text with input features
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images
        )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
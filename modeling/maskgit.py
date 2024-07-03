"""This file contains implementation for MaskGIT model.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/huggingface/open-muse
    https://github.com/baaivision/MUSE-Pytorch
"""

import torch
from torch import nn
import numpy as np
import math
import torch.utils.checkpoint
from transformers import BertConfig, BertModel


class ImageBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.target_codebook_size = config.model.vq_model.codebook_size
        self.condition_num_classes = config.model.generator.condition_num_classes
        self.image_seq_len = config.model.generator.image_seq_len
        self.mask_token_id = self.target_codebook_size

        self.model = BertModel(BertConfig(
            vocab_size=self.target_codebook_size + self.condition_num_classes + 2,
            hidden_size=768,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=3072,
            hidden_act='gelu',
            hidden_dropout_prob=config.model.generator.dropout,
            attention_probs_dropout_prob=config.model.generator.attn_drop,
            max_position_embeddings=config.model.generator.image_seq_len + 1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=None,
            position_embedding_type="absolute",
            use_cache=True
        ), add_pooling_layer=False)
        self.model.lm_head = nn.Linear(768, self.target_codebook_size, bias=True)
        
        self.model.post_init()

    def forward(self, input_ids=None, condition=None, cond_drop_prob=0.1):
        # Token space:
        #  [0, codebook_size - 1]                       : those are the learned quantized image tokens
        #  codebook_size                                : the mask token used to mask image tokens
        #  [codebook_size + 1, codebook_size + nclass]  : the imagenet class tokens
        #  codebook_size + 1 + nclass                   : the class drop label
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        # Shift the classes
        condition = condition + self.target_codebook_size + 1  # [0, 999] -> [codebook_size + 1, codebook_size + 999]
        condition[drop_label_mask] = self.condition_num_classes + self.target_codebook_size + 1
        # prepend condition token
        if input_ids is not None:
            input_ids = torch.cat([condition.view(condition.shape[0], -1),
                                   input_ids.view(input_ids.shape[0], -1),], dim=1)
        else:
            # at least there should be masked token
            raise NotImplementedError
        model_output = self.model(input_ids=input_ids)
        model_output = model_output[0]
        return self.model.lm_head(model_output[:, 1:]) # remove cond
    
    # ref: https://github.com/baaivision/MUSE-Pytorch/blob/master/libs/muse.py#L40
    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale=3.0,
                 guidance_decay=False,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):
        device = condition.device
        ids = torch.full((condition.shape[0], self.image_seq_len),
                          self.mask_token_id, device=device)
        cfg_scale = 0. if guidance_decay else guidance_scale

        for step in range(num_sample_steps):
            ratio = 1. * (step + 1) / num_sample_steps
            annealed_temp = randomize_temperature * (1.0 - ratio)
            is_mask = (ids == self.mask_token_id)
            if cfg_scale != 0:
                cond_logits = self.forward(
                    ids, condition, cond_drop_prob=0.0
                )
                uncond_logits = self.forward(
                    ids, condition, cond_drop_prob=1.0
                )
                logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = self.forward(
                    ids, condition, cond_drop_prob=0.0
                )

            if softmax_temperature_annealing:
                softmax_temperature = 0.5 + 0.8 * (1 - ratio)
                logits = logits / softmax_temperature

            # Add gumbel noise
            def log(t, eps=1e-20):
                return torch.log(t.clamp(min=eps))
            def gumbel_noise(t):
                noise = torch.zeros_like(t).uniform_(0, 1)
                return -log(-log(noise))
            def add_gumbel_noise(t, temperature):
                return t + temperature * gumbel_noise(t)

            sampled_ids = add_gumbel_noise(logits, annealed_temp).argmax(dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_ratio = np.arccos(ratio) / (math.pi * 0.5)

            mask_len = torch.Tensor([np.floor(self.image_seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                     torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1,
                                                   mask_len))[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)
            if step == num_sample_steps - 1:
                ids = sampled_ids
            else:
                ids = torch.where(masking, self.mask_token_id, sampled_ids)

            if guidance_decay:
                cfg_scale = ratio * guidance_scale
        return ids

"""This file contains the model definition of TA-TiTok.

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
"""

import torch
from einops import rearrange

from .titok import TiTok
from modeling.modules.blocks import TATiTokDecoder
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from omegaconf import OmegaConf

from huggingface_hub import PyTorchModelHubMixin


class TATiTok(TiTok, PyTorchModelHubMixin, tags=["arxiv:2501.07730", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__(config)
        self.decoder = TATiTokDecoder(config)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
                clustering_vq=config.model.vq_model.clustering_vq)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError

    def decode(self, z_quantized, text_guidance):
        decoded = self.decoder(z_quantized, text_guidance)
        return decoded
    
    def decode_tokens(self, tokens, text_guidance):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized, text_guidance)
        return decoded
    
    def forward(self, x, text_guidance):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized, text_guidance)
        return decoded, result_dict

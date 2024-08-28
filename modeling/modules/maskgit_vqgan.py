"""This file contains code for MaskGIT-VQGAN.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/huggingface/open-muse/blob/main/muse/modeling_maskgit_vqgan.py
"""
# Copyright 2023 Google LLC and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""MaskGIT Tokenizer based on VQGAN.

This tokenizer is a reimplementation of VQGAN [https://arxiv.org/abs/2012.09841]
with several modifications. The non-local layers are removed from VQGAN for
faster speed.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


# Conv2D with same padding
class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv2dSame(self.in_channels, self.out_channels_, kernel_size=3, bias=False)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels_, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = Conv2dSame(self.out_channels_, self.out_channels_, kernel_size=3, bias=False)

        if self.in_channels != self.out_channels_:
            self.nin_shortcut = Conv2dSame(self.out_channels_, self.out_channels_, kernel_size=1, bias=False)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            residual = self.nin_shortcut(hidden_states)

        return hidden_states + residual


class DownsamplingBlock(nn.Module):
    def __init__(self, config, block_idx: int):
        super().__init__()

        self.config = config
        self.block_idx = block_idx

        in_channel_mult = (1,) + tuple(self.config.channel_mult)
        block_in = self.config.hidden_channels * in_channel_mult[self.block_idx]
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout_prob=self.config.dropout))
            block_in = block_out
        self.block = res_blocks

        self.downsample = self.block_idx != self.config.num_resolutions - 1

    def forward(self, hidden_states):
        for res_block in self.block:
            hidden_states = res_block(hidden_states)

        if self.downsample:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)

        return hidden_states


class UpsamplingBlock(nn.Module):
    def __init__(self, config, block_idx: int):
        super().__init__()

        self.config = config
        self.block_idx = block_idx

        if self.block_idx == self.config.num_resolutions - 1:
            block_in = self.config.hidden_channels * self.config.channel_mult[-1]
        else:
            block_in = self.config.hidden_channels * self.config.channel_mult[self.block_idx + 1]

        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = []
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout_prob=self.config.dropout))
            block_in = block_out
        self.block = nn.ModuleList(res_blocks)

        self.add_upsample = self.block_idx != 0
        if self.add_upsample:
            self.upsample_conv = Conv2dSame(block_out, block_out, kernel_size=3)

    def forward(self, hidden_states):
        for res_block in self.block:
            hidden_states = res_block(hidden_states)

        if self.add_upsample:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            hidden_states = self.upsample_conv(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # downsampling
        self.conv_in = Conv2dSame(self.config.num_channels, self.config.hidden_channels, kernel_size=3, bias=False)

        downsample_blocks = []
        for i_level in range(self.config.num_resolutions):
            downsample_blocks.append(DownsamplingBlock(self.config, block_idx=i_level))
        self.down = nn.ModuleList(downsample_blocks)

        # middle
        mid_channels = self.config.hidden_channels * self.config.channel_mult[-1]
        res_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(mid_channels, mid_channels, dropout_prob=self.config.dropout))
        self.mid = res_blocks

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=mid_channels, eps=1e-6, affine=True)
        self.conv_out = Conv2dSame(mid_channels, self.config.z_channels, kernel_size=1)

    def forward(self, pixel_values):
        # downsampling
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states)

        # middle
        for block in self.mid:
            hidden_states = block(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # compute in_channel_mult, block_in and curr_res at lowest res
        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        self.z_shape = (1, self.config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = Conv2dSame(self.config.z_channels, block_in, kernel_size=3)

        # middle
        res_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_in, dropout_prob=self.config.dropout))
        self.mid = res_blocks

        # upsampling
        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(UpsamplingBlock(self.config, block_idx=i_level))
        self.up = nn.ModuleList(list(reversed(upsample_blocks)))  # reverse to get consistent order

        # end
        block_out = self.config.hidden_channels * self.config.channel_mult[0]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out, eps=1e-6, affine=True)
        self.conv_out = Conv2dSame(block_out, self.config.num_channels, kernel_size=3)

    def forward(self, hidden_states):
        # z to block_in
        hidden_states = self.conv_in(hidden_states)

        # middle
        for block in self.mid:
            hidden_states = block(hidden_states)

        # upsampling
        for block in reversed(self.up):
            hidden_states = block(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    Discretization bottleneck part of the VQ-VAE.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        r"""
        Args:
            num_embeddings: number of vectors in the quantized space.
            embedding_dim: dimensionality of the tensors in the quantized space.
                Inputs to the modules must be in this format as well.
            commitment_cost: scalar which controls the weighting of the loss terms
                (see equation 4 in the paper https://arxiv.org/abs/1711.00937 - this variable is Beta).
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, hidden_states, return_loss=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete one-hot vector that is the index of the
        closest embedding vector e_j z (continuous) -> z_q (discrete) z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()

        distances = self.compute_distances(hidden_states)
        min_encoding_indices = torch.argmin(distances, axis=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(hidden_states)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(hidden_states.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute loss for embedding
        loss = None
        if return_loss:
            loss = torch.mean((z_q.detach() - hidden_states) ** 2) + self.commitment_cost * torch.mean(
                (z_q - hidden_states.detach()) ** 2
            )
            # preserve gradients
            z_q = hidden_states + (z_q - hidden_states).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices, loss

    def compute_distances(self, hidden_states):
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        hidden_states_flattended = hidden_states.reshape((-1, self.embedding_dim))
        emb_weights = self.embedding.weight.t()

        inputs_norm_sq = hidden_states_flattended.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = emb_weights.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            hidden_states_flattended,
            emb_weights,
            alpha=-2.0,
        )
        return distances

    def get_codebook_entry(self, indices):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        if len(indices.shape) == 2:
            batch, num_tokens = indices.shape
            z_q = self.embedding(indices)
            z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1).permute(0, 3, 1, 2)
        elif len(indices.shape) == 3:
            batch, height, width = indices.shape
            indices = indices.view(batch, -1)
            z_q = self.embedding(indices)
            z_q = z_q.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        else:
            print(indices.shape)
            raise NotImplementedError
        return z_q

    # adapted from https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqvae/quantizations.py#L372
    def get_soft_code(self, hidden_states, temp=1.0, stochastic=False):
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()  # (batch, height, width, channel)
        distances = self.compute_distances(hidden_states)  # (batch * height * width, num_embeddings)

        soft_code = F.softmax(-distances / temp, dim=-1)  # (batch * height * width, num_embeddings)
        if stochastic:
            code = torch.multinomial(soft_code, 1)  # (batch * height * width, 1)
        else:
            code = distances.argmin(dim=-1)  # (batch * height * width)

        code = code.reshape(hidden_states.shape[0], -1)  # (batch, height * width)
        batch, num_tokens = code.shape
        soft_code = soft_code.reshape(batch, num_tokens, -1)  # (batch, height * width, num_embeddings)
        return soft_code, code

    def get_code(self, hidden_states):
        # reshape z -> (batch, height, width, channel)
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        distances = self.compute_distances(hidden_states)
        indices = torch.argmin(distances, axis=1).unsqueeze(1)
        indices = indices.reshape(hidden_states.shape[0], -1)
        return indices
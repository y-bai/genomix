#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		genomix_sampling.py
@Time    :   	2024/12/20 22:39:54
@Author  :   	Yong Bai
@Contact :   	baiyong at genomics.cn
@License :   	(C)Copyright 2023-2024, Yong Bai

    Licensed under the Apache License, Version 2.0 (the 'License');
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an 'AS IS' BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

@Desc    :   	 down sampling and up sampling for input_ids data

adapted from:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/canine/modeling_canine.py#L292

"""

from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange

from transformers import CanineForTokenClassification

from .genomix_block import GenoMixMamba2Block


class DownSamplingProjection(nn.Module):
    """downsample the input_ids after embedding layer
    
    (B, L, D) -> (B, L//downsampling_rate, D)

    L//downsampling_rate is the length of the input_ids after down sampling, 

    NOTE: we do not need padding, as we are not using causal conv1d but normal conv1d just for downsampling   
    
    """
    
    def __init__(
        self, 
        hidden_size: int,                           # input hidden size of the model, e.g., d_model of the model
        sampling_rate: int = 4,                     # kernel_size of the conv1d
        groups: int = 1,                            # groups of the conv1d, 1 or hidden_size

        conv_bias: bool = True,                     # bias of the conv1d
        dilation: int = 1,                          # dilation of the conv1d
        layer_norm_eps: float = 1e-12,              # eps of the layer norm
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            stride=sampling_rate,
            kernel_size=sampling_rate,
            dilation=dilation,
            # padding=(downsampling_rate - 1) * dilation,          
            bias=conv_bias,
            groups=groups,
            **factory_kwargs,
        )

        self.act = nn.SiLU()
        self.layerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x):
        """_summary_

        Parameters
        ----------
        hidden_states : Tensor
            (B, L, D) tensor of hidden states

        Returns
        -------
        (B, L//downsample_factor, D)
        """

        x = rearrange(x, "b l d -> b d l")
        x = self.act(self.conv1d(x))
        x = rearrange(x, "b d l -> b l d")
        x = self.layerNorm(x)
        return x

#
# Adapt from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/canine/modeling_canine.py#L338
#
class UpSamplingProjection(nn.Module):
    """
    Project representations from hidden_size*2 back to hidden_size across a window of w = config.upsampling_kernel_size
    characters.

    (batch, downsampled_seq_len, hidden_size + hidden_size) -> (batch_size, init_seq_len, hidden_size)

    """

    def __init__(
        self, 
        config
    ):
        super().__init__()
        self.config = config

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=config.upsampling_kernel_size,
            stride=1,
            padding='same',
        )
        self.activation = nn.SiLU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        inputs: torch.Tensor,
        final_seq_char_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # inputs has shape [batch, mol_seq, molecule_hidden_size+char_hidden_final]
        # we transpose it to be [batch, molecule_hidden_size+char_hidden_final, mol_seq]
        inputs = rearrange(inputs, "b m c -> b c m")

        # `result`: shape (batch_size, char_seq_len, hidden_size)
        result = self.conv(inputs)
        result = rearrange(result, "b c m -> b m c")
        result = self.activation(result)
        result = self.LayerNorm(result)
        result = self.dropout(result)
        final_char_seq = result

        if final_seq_char_positions is not None:
            # TODO add support for MLM
            raise NotImplementedError("MaskedLM to be supported")
        else:
            query_seq = final_char_seq

        return query_seq


class HiddenStateUpSampling(nn.Module):
    def __init__(
        self, 
        sampling_rate: int = 4
    ):
        super().__init__()
        self.sampling_rate = sampling_rate


# init encoder for the input_ids embedding 
class GenomixLongTermInitEnocder(nn.Module):
    def __init__(
        self,
        config,                     # genomix config
        layer_ide: int = 0,
        d_intermediate: int = 0,    # intermediate size of the mlp, this allows to use a different intermediate size for the mlp
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.init_shallow_encoder = GenoMixMamba2Block(
            config.d_model,
            layer_idx=layer_ide,
            rms_norm=config.rms_norm,
            norm_epsilon=config.norm_epsilon,
            residual_in_fp32=config.residual_in_fp32, 
            fused_add_norm=config.fused_add_norm,
            ssm_cfg = config.ssm_cfg if config.ssm_cfg is not None else {},
            attn_enable = False,
            attn_cfg={},
            mlp_attn_only = False,
            moe_enable=False,
            moe_cfg={},
            bidirectional_cfg = config.bidirectional_cfg if config.bidirectional_cfg is not None else {},
            d_intermediate=d_intermediate,         
            **factory_kwargs,
        )
    
    def forward(
        self,
        hidden_states,
        residual: Optional[torch.Tensor] = None, 
        inference_params=None, 
        **mixer_kwargs
    ):
        return self.init_shallow_encoder(
            hidden_states, 
            residual=residual, 
            inference_params=inference_params, 
            **mixer_kwargs
        )
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.init_shallow_encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

# 
# Upsample the hidden states 
# https://arxiv.org/pdf/2103.06874
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/canine/modeling_canine.py#L1196
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/canine/modeling_canine.py#L1059
# 
def upsample_hiddenstate(
    hidden_state: torch.Tensor, 
    input_ids_length: torch.Tensor,
    downsampling_rate: int
) -> torch.Tensor:
    """Repeats molecules to make them the same length as the char sequence."""

    rate = downsampling_rate
    # `repeated`: [batch_size, input_ids_len, molecule_hidden_size]
    repeated = torch.repeat_interleave(hidden_state, repeats=rate, dim=-2)

    # So far, we've repeated the elements sufficient for any `input_ids_length`
    # that's a multiple of `downsampling_rate`. Now we account for the last
    # n elements (n < `downsampling_rate`), i.e. the remainder of floor
    # division. We do this by repeating the last hidden_state a few extra times.
    last_hidden_state = hidden_state[:, -1:, :]
    remainder_length = torch.fmod(torch.tensor(input_ids_length), torch.tensor(rate)).item()
    remainder_repeated = torch.repeat_interleave(
        last_hidden_state,
        # +1 molecule to compensate for truncation.
        repeats=remainder_length + rate,
        dim=-2,
    )

    # `repeated`: [batch_size, input_ids_length, hidden_size]
    return torch.cat([repeated, remainder_repeated], dim=-2)

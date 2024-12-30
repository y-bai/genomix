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

import logging

from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange

from transformers import CanineForTokenClassification

from .genomix_block import GenoMixMamba2Block

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    logger.warning(
        "Cannot import RMSNorm from mamba_ssm.ops.triton.layer_norm. "
        "Please make sure you have installed the Triton package."
    )
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class DownSamplingProjection(nn.Module):
    """downsample the input_ids after embedding layer
    
    (B, L, D) -> (B, L//downsampling_rate, D)

    L//downsampling_rate is the length of the input_ids after down sampling, 

    NOTE: we do not need padding, as we are not using causal conv1d but normal conv1d just for downsampling   
    
    """
    
    def __init__(
        self, 
        hidden_size: int,                           # input hidden size of the model, e.g., d_model of the model
        lt_conv_sampling_rate: int = 4,             # kernel_size of the conv1d
        lt_conv_groups: int = 1,                    # groups of the conv1d, 1 or hidden_size
        lt_conv_bias: bool = True,
        lt_conv_dilation: int = 1,                  # dilation of the conv1d            
        rms_norm: bool = False,                     # whether to use RMSNorm or not
        norm_eps: float = 1e-12,              # eps of the layer norm
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=lt_conv_sampling_rate,
            stride=lt_conv_sampling_rate,
            dilation=lt_conv_dilation,
            # padding=(downsampling_rate - 1) * dilation,          
            bias=lt_conv_bias,
            groups=lt_conv_groups,
            **factory_kwargs,
        )

        self.act = nn.SiLU()
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            hidden_size, eps=norm_eps, **factory_kwargs
        )

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
        x = self.norm_f(x)
        return x

#
# Adapt from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/canine/modeling_canine.py#L338
#
class ConvProjection(nn.Module):
    """
    Project representations from hidden_size*2 back to hidden_size across a window of w = config.upsampling_kernel_size
    characters.

    (batch, init_seq_len, hidden_size + hidden_size) -> (batch_size, init_seq_len, hidden_size)

    """

    def __init__(
        self, 
        hidden_size: int,
        lt_conv_sampling_rate: int = 4,
        lt_conv_groups: int = 1,
        lt_conv_bias: bool = True,                     # bias of the conv1d
        lt_conv_dilation: int = 1,                     # dilation of the conv1d
        rms_norm: bool = False,
        norm_eps: float = 1e-12,              # eps of the layer norm
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=hidden_size * 2,
            out_channels=hidden_size,
            kernel_size=lt_conv_sampling_rate,
            stride=1,
            groups=lt_conv_groups,
            padding='same',
            dilation=lt_conv_dilation,
            bias=lt_conv_bias,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            hidden_size, eps=norm_eps, **factory_kwargs
        )

    def forward(self,inputs: torch.Tensor) -> torch.Tensor:

        # inputs has shape [batch, mol_seq, molecule_hidden_size+char_hidden_final]
        # we transpose it to be [batch, molecule_hidden_size+char_hidden_final, mol_seq]
        inputs = rearrange(inputs, "b m c -> b c m")

        # `result`: shape (batch_size, char_seq_len, hidden_size)
        result = self.act(self.conv1d(inputs))
        result = rearrange(result, "b c m -> b m c")
        result = self.norm_f(result)
        return result


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

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = config.fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        self.residual_in_fp32 = config.residual_in_fp32

        self.act = nn.SiLU()
        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )

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

        long_term_config = config.long_term_cfg
        enable_long_term = long_term_config.pop("enable_long_term", True)

        self.downsampling_projection = DownSamplingProjection(
            config.d_model,
            rms_norm = config.rms_norm,                     # whether to use RMSNorm or not
            norm_eps = config.norm_epsilon,                 # eps of the layer norm
            **long_term_config,
            **factory_kwargs,
        )
    
    def forward(
        self,
        hidden_states,
        residual: Optional[torch.Tensor] = None, 
        inference_params=None, 
        **mixer_kwargs
    ):
        
        # return the hidden states and residual
        hidden_states, residual=self.init_shallow_encoder(
            hidden_states,                      # embedding hidden states
            residual=residual,                  # residual hidden states, should be None
            inference_params=inference_params, 
            **mixer_kwargs
        )

        if not self.fused_add_norm:
            residual_ = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual_.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        # downsample the hidden states
        downsampled_hidden_states = self.downsampling_projection(hidden_states)
        return downsampled_hidden_states, hidden_states
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.init_shallow_encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class GenomixLongTermLastDecoder(nn.Module):

    def __init__(
        self,
        config,                                 # genomix config
        device=None, 
        dtype=None
    ):

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        long_term_config = config.long_term_cfg
        enable_long_term = long_term_config.pop("enable_long_term", True)

        lt_conv_sampling_rate = long_term_config.get("lt_conv_sampling_rate", 4)

        self.sampling_rate = lt_conv_sampling_rate

        self.conv_projection = ConvProjection(
            config.d_model,
            rms_norm=config.rms_norm,
            norm_eps=config.norm_epsilon,
            **long_term_config,
            **factory_kwargs,
        )
    
    def forward(
        self, 
        init_hidden_state: torch.Tensor,                # (B, L, D) 
        downsampled_hidden_states: torch.Tensor,        # (B, L//downsampling_rate, D)
    ) -> torch.Tensor:
        
        # repeated_molecules: (B, L, D)
        repeated_molecules = self._repeat_molecules(
            downsampling_rate=self.sampling_rate,
            hidden_state=downsampled_hidden_states,
            input_length=init_hidden_state.size(-2),
        )

        # concat: (B, L, D+D)
        concat = torch.cat([init_hidden_state, repeated_molecules], dim=-1)

        # projection: (B, L, D+D) -> (B, L, D)
        seq_out = self.conv_projection(concat)

        return seq_out

    # 
    # Upsample the hidden states 
    # https://arxiv.org/pdf/2103.06874
    # 
    def _repeat_molecules(
        self,
        downsampling_rate: int,
        hidden_state: torch.Tensor, 
        input_length: torch.Tensor,
    ) -> torch.Tensor:
        """Repeats molecules to make them the same length as the char sequence."""

        rate = downsampling_rate
        # `repeated`: [batch_size, input_ids_len, molecule_hidden_size]
        repeated = torch.repeat_interleave(hidden_state, repeats=rate, dim=-2)

        # So far, we've repeated the elements sufficient for any `input_ids_length`
        # that's a multiple of `sampling_rate`. Now we account for the last
        # n elements (n < `downsampling_rate`), i.e. the remainder of floor
        # division. We do this by repeating the last hidden_state a few extra times.
        last_hidden_state = hidden_state[:, -1:, :]
        remainder_length = torch.fmod(torch.tensor(input_length), torch.tensor(rate)).item()
        remainder_repeated = torch.repeat_interleave(
            last_hidden_state,
            # +1 molecule to compensate for truncation.
            repeats=remainder_length + rate,
            dim=-2,
        )
        # `repeated`: [batch_size, input_ids_length, hidden_size]
        return torch.cat([repeated, remainder_repeated], dim=-2)
    

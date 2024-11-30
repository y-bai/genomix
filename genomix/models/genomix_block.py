#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		genomix_m2block.py
@Time    :   	2024/11/30 03:22:21
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

@Desc    :   	 Mamaba2 block

Adapted from:
- `create_block()` function in
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py

- `Block` class in
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py

"""

import logging
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
except ImportError:
    logger.warning(
        "Cannot import RMSNorm from mamba_ssm.ops.triton.layer_norm. "
        "Please make sure you have installed the Triton package."
    )
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class GenoMixMamba2Block(nn.Module):

    def __init__(
        self, 
        d_model: int,
        layer_idx=0, 
        rms_norm=True,
        norm_epsilon=1e-5,
        residual_in_fp32=True, 
        fused_add_norm=True,
        ssm_cfg={},
        attn_enable=False,
        attn_cfg={},
        moe_enable=False,
        moe_cfg={},
        d_intermediate=0,           # whether MLP is used
        device=None,
        dtype=None,
    ):
        """GenoMix mixer block

        any combination of Mamaba2  Attention and MLP. 

        Parameters
        ----------
        d_model : int
            input embedding dimension,

        layer_idx : int
            layer index

        rms_norm : bool, optional
            indicating whether using RMSNorm, by default True

        norm_epsilon : float, optional
            epsilon for LayerNorm or RMSNorm, by default 1e-5

        residual_in_fp32 : bool, optional
            indicating whether keep residual in torch.flost32 type, by default True

        fused_add_norm : bool, optional
            indicating whether using `layer_norm_fn` function for performance improvement, by default True

        ssm_cfg: dict, optional
            Mamba2 parameters, by default {}
            if ssm_cfg = {}, then use default parameters for Mamba2
            see: mamba_ssm.modules.mamba2.Mamba2

        attn_enable : bool
            indicating whether using attention layer, by default False

        attn_cfg : dict, optional
            MHA parameters, by default {}
            if attn_cfg = {}, then use default parameters for MHA
            see: mamba_ssm.modules.mha.MHA
        
        moe_enable : bool, optional
            indicating whether using MoE layer, by default False

        moe_cfg : dict, optional
            MoE parameters, by default {}
            if moe_cfg = {}, then use default parameters for MoE
        
        d_intermediate : int, optional
            indicating whether using GatedMLP(feedforward) layer, by default 0
            - 0: no GatedMLP layer

        device : _type_, optional
            device , by default None
        dtype : _type_, optional
            Tensor dtype, by default None
        """
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # get the ssm_cfg
        ssm_cfg = ssm_cfg if ssm_cfg is not None else {}
        attn_cfg = attn_cfg if attn_cfg is not None else {}

        # deinfe mixer layer
        self.mixer = Mamba2(                # Mamba2 layer
            d_model,                        
            layer_idx, 
            **ssm_cfg, 
            **factory_kwargs
        ) if not attn_enable else MHA(       # attention layer
            d_model,
            layer_idx=layer_idx, 
            **attn_cfg, 
            **factory_kwargs
        )
        
        # define norm layer
        self.norm  = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model,
            eps=norm_epsilon, 
            **factory_kwargs
        )

        # define mlp layer
        if d_intermediate == 0:
            # mlp_layer = nn.Identity
            self.mlp = None
        else:
            # NOTE: In original code, the slef.norm = norm_cls, 
            # and slef.norm = norm_cls as well. This may result in the same norm layer.
            # ie., self.norm = self.norm2. 
            # see
            # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py#L29-L33
            # 
            # But here, we create two different norm layers.
            self.norm2 = (nn.LayerNorm if not rms_norm else RMSNorm)(
                d_model,
                eps=norm_epsilon, 
                **factory_kwargs
            )
            self.mlp = GatedMLP(
                d_model,
                hidden_features=d_intermediate, 
                out_features=d_model, 
                **factory_kwargs
            )
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
    
    def forward(
        self,
        hidden_states: Tensor, 
        residual: Optional[Tensor] = None, 
        inference_params=None, 
        **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
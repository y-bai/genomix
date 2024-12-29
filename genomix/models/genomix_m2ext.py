#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		genomix_m2wrapper.py
@Time    :   	2024/12/20 12:03:28
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

@Desc    :   	 None

reference:
https://github.com/state-spaces/mamba/blob/52f57d66f8d6d02a73b1e38f5f3708eb8ccfa39c/mamba_ssm/models/mixer_seq_simple.py#L93

"""

import logging
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2

logger = logging.getLogger(__name__)

class GenoMixMamba2Extension(nn.Module):
    """Mamba2 extention for supporting bi-directionality."""

    def __init__(
        self,
        d_model,                        
        layer_idx=None,  # Absorb kwarg for general module

        bidirectional = False,
        bidirectional_strategy = "add", # "add", "concat" or "ewmul"

        ssm_cfg = {},   # Mamba2 options

        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy

        self.mixer_fwd = Mamba2(                # Mamba2 layer
            d_model,                        
            layer_idx=layer_idx, 
            **ssm_cfg, 
            **factory_kwargs
        ) 

        if bidirectional:
            self.mixer_bwd = Mamba2(                # Mamba2 layer
                d_model,                        
                layer_idx=layer_idx, 
                **ssm_cfg, 
                **factory_kwargs
            ) 
        else:
            self.mixer_bwd = None

        if bidirectional_strategy == 'concat':
            # TODO: This is not the best way to combine forward and backward hidden states
            self.concat_proj = nn.Linear(d_model * 2, d_model, bias=False)
        else:
            self.concat_proj = None
    
    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        """
        
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states

        Parameters
        ----------
        hidden_states : Tensor
        inference_params : _type_, optional
            _description_, by default None
        """
        out = self.mixer_fwd(hidden_states, inference_params, **mixer_kwargs)
        if self.bidirectional:
            bwd_out = self.mixer_bwd(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params, 
                **mixer_kwargs
            ).flip(dims=(1,))                   # Flip back for combining with forward hidden states

            if self.bidirectional_strategy == "add":
                out = out + bwd_out
            elif self.bidirectional_strategy == "concat":
                out = self.concat_proj(torch.cat([out, bwd_out], dim=-1))
            elif self.bidirectional_strategy == "ewmul":
                out = out * bwd_out
        return out
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate memory for inference cache

        Parameters
        ----------
        batch_size : int
        max_seqlen : int
        dtype : torch.dtype, optional
        """

        # TODO: need to test this dict return
        return {
            'mixer_fwd': self.mixer_fwd.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs),
            'mixer_bwd': (self.mixer_bwd.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs) 
                          if self.bidirectional 
                          else None)
        }
        

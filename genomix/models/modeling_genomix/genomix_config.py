#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		genomix_config.py
@Time    :   	2024/11/29 17:23:27
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

@Desc    :   	 

Adapted from 
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py

"""

from transformers import PretrainedConfig, Mamba2Config

class GenoMixMamba2Config(PretrainedConfig):

    model_type = "genomix-mamba2"

    def __init__(
        self,
        vocab_size: int = 16,
        d_model: int = 256,
        d_intermediate: int = 0,
        n_groups: int = 1,
        n_heads: int = 8,
        n_layers: int = 12,
        ssm_cfg={ # Mamba2 block config
            "d_state": 128,
            "d_conv": 4,
            "conv_init": None,
            "expand": 2,
            "headdim": 64,
            "d_ssm": None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
            "ngroups": 1,
            "A_init_range": (1, 16),
            "D_has_hdim": False,
            "rmsnorm": True,
            "norm_before_gate": False,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "dt_init_floor": 1e-4,
            "dt_limit": (0.0, float("inf")),
            "bias": False,
            "conv_bias": True,
            # Fused kernel and sharding options
            "chunk_size": 256,
            "use_mem_eff_path": True,
            "process_group": None,
            "sequence_parallel": True,
            "A_init_range": (1, 16),
        },
        attn_layer_idx=[9, 18, 27, 36, 45, 56],
        attn_cfg={
            "causal": True,
            "d_conv": 4,
            "head_dim": 128,
            "num_heads": 30,
            "out_proj_bias": False,
            "qkv_proj_bias": False,
            "rotary_emb_dim": 64
        },
        initializer_cfg={
            "initializer_range": 0.2,
            "rescale_prenorm_residual": True,
        },
        # for computation improvement
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,                        # for falsh attention
        tie_embeddings: bool = True,
        activation: str = "relu",

        # only for mixer
        rms_norm: bool = True,
        norm_epsilon: float = 1e-5,

        **kwargs
    ):
        super().__init__(**kwargs)

        

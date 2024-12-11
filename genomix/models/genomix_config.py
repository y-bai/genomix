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


Mamba2 slower than mamba1:
https://github.com/state-spaces/mamba/issues/355
sovled by: warm up


Mamba2 has only been tested with dim_model multiples of 512.
https://github.com/state-spaces/mamba/issues/347

"""

from transformers import PretrainedConfig

class GenoMixMamba2Config(PretrainedConfig):


    model_type = "genomix-mamba2"

    def __init__(
        self,
        vocab_size: int = 16,
        d_model: int = 256,
        d_intermediate: int=256*2+128,  # MLP hidden size
        n_layers: int=32,

        pad_token_id=2, # pad_token_id=eos_token_id
        bos_token_id=0,
        eos_token_id=2,

        attn_layer_idx=[],
        moe_layer_idx=[],

        pad_vocab_size_multiple: int = 8,   # for falsh attention

        # for float value embedding
        input_embedding_cfg={
            "use_tabular_embedding": False, 
            "fusion_type": "add",
        },

        # Mamba2 block config
        ssm_cfg={ 
            "d_state": 64,          # SSM state expansion factor, (`N` in [1] Algorithm 2), typically 64 or 128 
            # https://github.com/state-spaces/mamba/issues/439
            "headdim": 32,          # must d_model/headdim%8==0, default 64 in mamba2，should be small

            "d_conv": 4,            # default 4 in mamba2, no need to change
            "conv_init": None,      # init for conv layer, no need to change
            "expand": 2,            # default 2 in mamba2, no need to change
            # https://github.com/state-spaces/mamba/issues/360
            # self.d_ssm % self.headdim == 0
            # if d_ssm is None, d_ssm = d_model * expand
            "d_ssm": None,          # the dimension of D matrix, if None, we only apply SSM on this dimensions, the rest uses gated
            "ngroups": 1,           # default 1 in mamba2, no need to change
            "A_init_range": (1, 16),# default (1, 16) in mamba2, no need to change
            "D_has_hdim": False,    # default False in mamba2, no need to change
            "rmsnorm": True,        # default True in mamba2, no need to change
            "norm_before_gate": False,  # default False in mamba2, no need to change
            "dt_min": 0.001,        # dt: delta time, (ie., time step), default 0.001 in mamba2, no need to change
            "dt_max": 0.1,          # default 0.1 in mamba2, no need to change
            "dt_init_floor": 1e-4,  # default 1e-4 in mamba2, no need to change
            "dt_limit": (0.0, float("inf")),    # default (0.0, float("inf")) in mamba2, no need to change
            "bias": False,          # default False in mamba2, no need to change  
            "conv_bias": True,      # default True in mamba2, no need to change
            # Fused kernel and sharding options
            # https://github.com/state-spaces/mamba/issues/449
            "chunk_size": 256,          # default 256 in mamba2, no need to change
            "use_mem_eff_path": True,   # default True in mamba2, no need to change
            "process_group": None,      # default None in mamba2, no need to change
            "sequence_parallel": True,  # default True in mamba2, no need to change
        },

        # Attention config
        # see: https://huggingface.co/state-spaces/mamba2attn-2.7b/blob/main/config.json
        attn_cfg={
            "num_heads": 8,
            "rotary_emb_dim": 32,           # rotary_emb_dim = min(dim_head, MIN_DIM_HEAD)  
            "causal": True, 
            "head_dim": None,               # If None, use embed_dim // num_heads, no need to change
            
            "d_conv": 4,                    # if > 0, use nn.Conv1d in MHA, no need to change
            "out_proj_bias": False,         # no need to change
            "qkv_proj_bias": False,         # no neeed to change
               
        },
        moe_cfg=None,
        initializer_cfg={
            "initializer_range": 0.1,           # default 0.2 in mamba2, no need to change
            "rescale_prenorm_residual": False,  # default False in mamba2, no need to change
        },
        
        # for computation improvement
        residual_in_fp32: bool = True,  # default True in mamba2, no need to change
        fused_add_norm: bool = True,    # default True in mamba2, no need to change
        # activation: str = "silu",       # default "silu" in mamba2, no need to change

        # only for mixer
        rms_norm: bool = True,          # default True in mamba2, no need to change
        norm_epsilon: float = 1e-5,     # default 1e-5 in mamba2, no need to change

        tie_word_embeddings: bool = True,   # default True in mamba2, no need to change
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        vocab_size : int, optional
            the size of vocabulary, by default 16

        d_model : int, optional
            the dimension of input embedding, by default 256

        d_intermediate : int, optional
            the hidden size of MLP, by default 0
            - If 0, no MLP is used

        n_layers : int, optional
            the number of Mamba2 blocks, by default 32

        pad_token_id : int, optional
            the token id for padding, by default 2
            equal to eos_token_id
        
        bos_token_id : int, optional
            the token id for begin of sentence, by default 0

        eos_token_id : int, optional
            the token id for end of sentence, by default 2
        
        attn_layer_idx : list, optional
            the index of Mamba2 blocks that use attention layer, by default [3, 7, 11]

        moe_layer_idx : _type_, optional
            the index of Mamba2 blocks that use MoE layer, by default None

        pad_vocab_size_multiple : int, optional
            the multiple of pad_vocab_size, by default 8, or 16
            see https://huggingface.co/state-spaces/mamba2-130m/blob/main/config.json
        
        input_embedding_cfg : dict, optional
            by default 
            {
                "use_tabular_embedding": False, 
                "fusion_type": "add",
            }
        
        ssm_cfg : dict, optional
            the parameters for Mamba2 layer, by default
            {
                 "d_state": 64,          # SSM state expansion factor, (`N` in [1] Algorithm 2), typically 64 or 128 
                # https://github.com/state-spaces/mamba/issues/439
                "headdim": 64,          # must d_model/headdim%8==0, default 64 in mamba2，should be small

                "d_conv": 4,            # default 4 in mamba2, no need to change
                "conv_init": None,      # init for conv layer, no need to change
                "expand": 2,            # default 2 in mamba2, no need to change
                # https://github.com/state-spaces/mamba/issues/360
                # self.d_ssm % self.headdim == 0
                "d_ssm": None,          # the dimension of D matrix, if None, we only apply SSM on this dimensions, the rest uses gated
                "ngroups": 1,           # default 1 in mamba2, no need to change
                "A_init_range": (1, 16),# default (1, 16) in mamba2, no need to change
                "D_has_hdim": False,    # default False in mamba2, no need to change
                "rmsnorm": True,        # default True in mamba2, no need to change
                "norm_before_gate": False,  # default False in mamba2, no need to change
                "dt_min": 0.001,        # dt: delta time, (ie., time step), default 0.001 in mamba2, no need to change
                "dt_max": 0.1,          # default 0.1 in mamba2, no need to change
                "dt_init_floor": 1e-4,  # default 1e-4 in mamba2, no need to change
                "dt_limit": (0.0, float("inf")),    # default (0.0, float("inf")) in mamba2, no need to change
                "bias": False,          # default False in mamba2, no need to change  
                "conv_bias": True,      # default True in mamba2, no need to change
                # Fused kernel and sharding options
                # https://github.com/state-spaces/mamba/issues/449
                "chunk_size": 256,          # default 256 in mamba2, no need to change
                "use_mem_eff_path": True,   # default True in mamba2, no need to change
                "process_group": None,      # default None in mamba2, no need to change
                "sequence_parallel": True,  # default True in mamba2, no need to change
            }
            
        attn_cfg : dict, optional
            the parameters for attention layer, by default 
            { 
                "num_heads": 8, 
                "rotary_emb_dim": 32,           # rotary_emb_dim = min(dim_head, MIN_DIM_HEAD) 
                "causal": True, 
                "head_dim": None,               # If None, use embed_dim // num_heads, no need to change  
                "d_conv": 4,                    # if > 0, use nn.Conv1d in MHA, no need to change 
                "out_proj_bias": False,         # no need to change 
                "qkv_proj_bias": False,         # no neeed to change  
            }
            
        moe_cfg : dict, optional
            the parameters for MoE layer, by default None

        initializer_cfg : dict, optional
            the parameters for model initialization, by default 
            { 
                "initializer_range": 0.1,           # default 0.2 in mamba2, no need to change 
                "rescale_prenorm_residual": False,  # default False in mamba2, no need to change 
            }

        residual_in_fp32 : bool, optional
            the type of residual in float point 32, by default True

        fused_add_norm : bool, optional
            add + norm then mixer, only for performance improvemant, by default True

        rms_norm : bool, optional
            If using the RMSNorm, by default True

        norm_epsilon : float, optional
            norm epsilon, by default 1e-5

        tie_word_embeddings : bool, optional
            If tie word embeddings, by default True

        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.n_layers = n_layers

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.attn_layer_idx = attn_layer_idx
        self.moe_layer_idx = moe_layer_idx

        self.pad_vocab_size_multiple = pad_vocab_size_multiple

        self.ssm_cfg = ssm_cfg
        self.attn_cfg = attn_cfg
        self.moe_cfg = moe_cfg
        self.initializer_cfg = initializer_cfg

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.rms_norm = rms_norm
        self.norm_epsilon = norm_epsilon

        self.input_embedding_cfg = input_embedding_cfg
        self.tie_word_embeddings = tie_word_embeddings

        

#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		genomix_vallinam2.py
@Time    :   	2024/11/27 10:17:30
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

Vallina Mamba2 + Attention(optional) + MLP(optional)

Adapated from: 
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py

"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, Mamba2ForCausalLM, Mamba2Config
from transformers.generation.utils import GenerationMixin

from .genomix_config import GenoMixMamba2Config
from .genomix_block import GenoMixMamba2Block
from .genomix_output import GenoMixModelOutput, GenoMixForCausalLMOutput
from .genomix_embedding import GenoMixTabularEmbedding

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    logger.warning(
        "Cannot import RMSNorm from mamba_ssm.ops.triton.layer_norm. "
        "Please make sure you have installed the Triton package."
    )
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# def create_mamba2block(
#     config: GenoMixMamba2Config,
#     layer_idx,
#     device=None,
#     dtype=None,
# ):
#     """create Mamba2 block with attention layer

#     Parameters
#     ----------
#     config : GenoMixMamba2Config
#         config object
#     layer_idx : int
#         index of the layer
#     device : optional
#         by default None
#     dtype : _type_, optional
#         by default None

#     Returns
#     -------
#     Block
#         Mamba2 block
#     """
#     factory_kwargs = {"device": device, "dtype": dtype}
#     super().__init__()

#     # get attention layer index
#     attn_layer_idx = config.attn_layer_idx
#     if attn_layer_idx is None:
#         attn_layer_idx = []
#     # get the ssm_cfg
#     ssm_cfg = config.ssm_cfg if config.ssm_cfg is not None else {}
    
#     # define mix layer
#     if layer_idx not in attn_layer_idx:
#         # Mamba2 layer
#         mix_layer = partial(
#             Mamba2,
#             layer_idx=layer_idx, 
#             **ssm_cfg, 
#             **factory_kwargs
#         )
#     else:
#         attn_cfg = config.attn_cfg if config.attn_cfg is not None else {}
#         # attention layer
#         mix_layer = partial(
#             MHA,
#             layer_idx=layer_idx, 
#             **attn_cfg, 
#             **factory_kwargs
#         )
    
#     # define norm layer
#     norm_layer = partial(
#         nn.LayerNorm if not config.rms_norm else RMSNorm, 
#         eps=config.norm_epsilon, 
#         **factory_kwargs
#     )

#     # define MLP layer
#     if config.d_intermediate == 0:
#         mlp_layer = nn.Identity
#     else:
#         mlp_layer = partial(
#             GatedMLP, 
#             hidden_features=config.d_intermediate, 
#             out_features=config.d_model, 
#             **factory_kwargs
#         )

#     block = Block(
#         config.d_model,
#         mix_layer,
#         mlp_layer,
#         norm_cls=norm_layer,
#         fused_add_norm=config.fused_add_norm,
#         residual_in_fp32=config.residual_in_fp32,
#     )
#     block.layer_idx = layer_idx

#     return block


class GenoMixMamba2PreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization 
    and a simple interface for downloading and loading pretrained models. 

    reference:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L86

    """
    config_class = GenoMixMamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["GenoMixMamba2Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(
        self, 
        module,
        # n_layers, # number of layers
        # initializer_range=0.02,  # Now only used for embedding layer
        # rescale_prenorm_residual=True,
        # n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
        """Initialize the weights."""
        initializer_cfg = self.config.initializer_cfg
        initializer_range = initializer_cfg.get("initializer_range", 0.02)
        rescale_prenorm_residual = initializer_cfg.get("rescale_prenorm_residual", True)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        n_residuals_per_layer = 1 if self.config.d_intermediate == 0 else 2
                        p /= math.sqrt(n_residuals_per_layer * self.config.n_layers)


class GenoMixMamba2Model(GenoMixMamba2PreTrainedModel):
    """Vallina Mamba2 + Attention(optional) + MLP(optional)

    Adapated from:

    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L118

    """
    def __init__(
        self, 
        config: GenoMixMamba2Config,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        if use_tabular_embedding:
            config.tie_word_embeddings = False
        super().__init__(config)

        input_embdding_cfg = config.input_embedding_cfg if config.input_embedding_cfg is not None else {}
        use_tabular_embedding = input_embdding_cfg.get("use_tabular_embedding", False)
        if use_tabular_embedding:
            fusion_type = input_embdding_cfg.get("fusion_type", "add")
            self.embeddings = GenoMixTabularEmbedding(
                config.vocab_size, config.d_model, fusion_type=fusion_type, **factory_kwargs
            )
        else:
            self.embeddings = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = config.fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
            
        attn_layer_idx = config.attn_layer_idx if config.attn_layer_idx is not None else []
        moe_layer_idx = config.moe_layer_idx if config.moe_layer_idx is not None else []

        ssm_cfg = config.ssm_cfg if config.ssm_cfg is not None else {}
        attn_cfg = config.attn_cfg if config.attn_cfg is not None else {}
        moe_cfg = config.moe_cfg if config.moe_cfg is not None else {}

        self.layers = nn.ModuleList(
            [
                GenoMixMamba2Block(
                    config.d_model,
                    layer_idx=i,
                    rms_norm=config.rms_norm,
                    norm_epsilon=config.norm_epsilon,
                    residual_in_fp32=config.residual_in_fp32, 
                    fused_add_norm=config.fused_add_norm,
                    ssm_cfg=ssm_cfg,
                    attn_enable=True if i in attn_layer_idx else False,
                    attn_cfg=attn_cfg,
                    moe_enable=True if i in moe_layer_idx else False,
                    moe_cfg=moe_cfg,
                    d_intermediate=config.d_intermediate,         
                    **factory_kwargs,
                )
                for i in range(config.n_layers)
            ]
        )

        self.residual_in_fp32 = config.residual_in_fp32

        # norm layer
        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()
    
    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings
    
    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embeddings(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
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

        return GenoMixModelOutput(last_hidden_state=hidden_states)



class GenoMixMamba2ForCausalLM(GenoMixMamba2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(
        self,
        config: GenoMixMamba2Config,
        device=None,
        dtype=None,
    ) -> None:
        
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple

        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(config)

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.backbone = GenoMixMamba2Model(config, **factory_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        model_output = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)

        hidden_states = model_output.hidden_states
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        return GenoMixForCausalLMOutput(
            logits=logits,
            hidden_states=model_output.hidden_states,
        )



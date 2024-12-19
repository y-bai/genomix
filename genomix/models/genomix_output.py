#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		genomix_output.py
@Time    :   	2024/11/30 18:12:56
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

@Desc    :   	 GenoMix output

"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch

from transformers.utils import ModelOutput


@dataclass
# Copied from transformers.models.mamba.modeling_mamba.MambaOutput with MAMBA->MAMBA2,Mamba->Mamba2
class GenoMixModelOutput(ModelOutput):
    """
    Class for the GenoMixMamba2 model outputs.

    Adapted from:
    :class:`~transformers.Mamba2Output`.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class GenoMixForCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None


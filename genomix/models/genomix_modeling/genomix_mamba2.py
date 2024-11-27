#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		genomix_mamba2.py
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

@Desc    :   	 directly modeling vanilla mamaba2 using Huggingface implementation

Reference:
https://huggingface.co/docs/transformers/main/en/model_doc/mamba2

This implementation is mainly adapted from Mamba2ForCausalLM from transformers library.

"""

import os
from typing import Optional
import torch
import torch.nn as nn

from transformers import (
    Mamba2Config, 
    Mamba2ForCausalLM,

    PreTrainedModel
)

class GenomixMamba2(PreTrainedModel):

    base_model_prefix = "model.backbone"

    def __init__(
        self, 
        config: Mamba2Config,
        *,
        model_name_or_path: Optional[str]=None,
        local_files_only: Optional[bool]=None,
        **kwargs
    ):
        # tie the weights of the input embeddings and the output embeddings
        if not hasattr(config, 'tie_word_embeddings'):
            config.tie_word_embeddings = True

        super().__init__(config, **kwargs)
        if model_name_or_path is None:
            # train from scrath
            self.model = Mamba2ForCausalLM(config)
        else:
            # load from pre-trained model
            self.model = Mamba2ForCausalLM.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            )
            # check if the vocab_size has been changed
            if config.vocab_size != self.model.config.vocab_size:
                self.model.resize_token_embeddings(config.vocab_size)

        # update config
        self.config.name_or_path = 'genomix_mamba2-' +  f'{model_size(self.model):.1f}'

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,  # for now we need this for generation
    ):
        # this return `Mamba2CausalLMOutput` with loss, logits and hidden_states
        return self.model(
            input_ids = input_ids,
            labels=labels,
            **kwargs,  # for now we need this for generation
        )





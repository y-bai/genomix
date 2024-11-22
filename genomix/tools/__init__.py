#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		__init__.py
@Time    :   	2024/11/21 13:30:38
@Author  :   	Yong Bai 
@Contact :   	baiyong at genomics.cn
@License :   	(C)Copyright 2023-2024, Yong Bai

                Licensed under the Apache License, Version 2.0 (the "License");
                you may not use this file except in compliance with the License.
                You may obtain a copy of the License at

                    http://www.apache.org/licenses/LICENSE-2.0

                Unless required by applicable law or agreed to in writing, software
                distributed under the License is distributed on an "AS IS" BASIS,
                WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                See the License for the specific language governing permissions and
                limitations under the License.

@Desc    :   	None

"""

from .train_tokenizer_config import (
    TrainTokenizerBaseConfig, 
    TrainBPETokenizerConfig, 
    TrainUnigramTokenizerConfig, 
    TrainSPMTokenizerConfig
)

from .train_tokenizer_process import (
    bpe_tokenizer_train, 
    unigram_tokenizer_train,
    spm_tokenizer_train
)

from .update_tokenizer_vocab import (
    bpe_vocab_update,
    unigram_vocab_update,
    spm_vocab_update
)
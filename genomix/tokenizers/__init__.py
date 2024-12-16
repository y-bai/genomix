#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :      __init__.py
@Time    :      2024/02/27 09:52:02
@Author  :      Yong Bai 
@Contact :      baiyong at genomics.cn
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
                
@Desc    :      None

"""

from enum import Enum

from .bpe_tokenizer import BioSeqBaseBPETokenizer, BioSeqBPETokenizer
from .bpe_tokenizer_fast import BioSeqBPETokenizerFast
from .unigram_tokenizer import BioSeqBaseUnigramTokenizer, BioSeqUnigramTokenizer
from .unigram_tokenizer_fast import BioSeqUnigramTokenizerFast
from .sentencepiece_tokenizer import BioSeqSPMTokenizer
from .sentencepiece_tokenizer_fast import BioSeqSPMTokenizerFast
from .char_tokenizer import CharacterTokenizer

from .tokenization_map import BioSeqTokenizerMap

PRETRAINED_MODEL_NAME_CLS_MAP = {
    "BPE_SLOW": BioSeqBPETokenizer,
    "BPE_FAST": BioSeqBPETokenizerFast,
    "UNIGRAM_SLOW": BioSeqUnigramTokenizer,
    "UNIGRAM_FAST": BioSeqUnigramTokenizerFast,
    "SPM_SLOW": BioSeqSPMTokenizer,
    "SPM_FAST": BioSeqSPMTokenizerFast,
    "CHAR_TOKEN": CharacterTokenizer,
}

def get_tokenizer_cls(model_name: str):
    return PRETRAINED_MODEL_NAME_CLS_MAP[model_name]
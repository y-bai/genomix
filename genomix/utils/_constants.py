#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_constants.py
@Time    :   	2024/11/17 14:56:30
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

@Desc    :   	constants and configurations

"""

from enum import Enum
import os
from ._common import check_dir_exists

class ExtendedEnum(Enum):
    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

class TOKENIZER_MODELS(ExtendedEnum):
    BPE = "BPE"
    UNIGRAM = "UNIGRAM"
    SPM = "SPM"

# under for SPM tokenizer model prefix
SPM_VOCAB_MODEL_PREFIX = "spm_vocab"

class SPECIAL_TOKENS(ExtendedEnum):
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    MASK = "<MASK>"
    # `<PAD>` is not included in the special tokens because we think of it as a `<EOS>` token.
# initial alphabet to be used for BPE/UNIGRAM tokneizer training
INITIAL_ALPHABETS = list("ACGTN")

# CACHE_DIR
CACHE_ROOT_DIR = check_dir_exists(
    os.getenv('GENOMIX_HOME', os.path.expanduser('~/.genomix')), 
    create=True)
CACHE_OUTPUT_DIR = check_dir_exists(
    os.path.join(CACHE_ROOT_DIR, 'output'), 
    create=True)
CACHE_DATA_DIR = check_dir_exists(
    os.path.join(CACHE_ROOT_DIR, 'data'), 
    create=True)







#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_tokenizer_train_config.py
@Time    :   	2024/11/21 12:24:04
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

@Desc    :   	tokenizer training configuration

"""

from dataclasses import dataclass, field
from typing import Iterator, List

from ..utils.constants import (
    SPECIAL_TOKENS, 
    INITIAL_ALPHABETS,
    SPM_VOCAB_MODEL_PREFIX,
    SPM_MODELS
)

@dataclass
class TrainTokenizerBaseConfig:
    """Base configuration for training tokenizer model.

    Parameter in the base config may not be changed if unnecessary.

    Parameters
    ----------
    unk_token : str, optional
        the unknown token, by default SPECIAL_TOKENS.UNK.value
    initial_alphabet : List[str], optional
        the initial alphabet, by default INITIAL_ALPHABETS
    special_tokens : List[str], optional
        the special tokens, by default SPECIAL_TOKENS.values()
    show_progress : bool, optional
        whether to show the progress, by default True
    data_iterator : Iterator[str], optional
        the iterator of data, by default None
    output_dir : str, optional
        the output directory to save the model, by default CACHE_OUTPUT_DIR
    vocab_size : int, optional
        the vocabulary size, by default 3000
    length : int, optional
        the length of each input sequence, by default 10000

    """

    unk_token: str = SPECIAL_TOKENS.UNK.value
    initial_alphabet: List[str] = field(default_factory=lambda: INITIAL_ALPHABETS)
    special_tokens: List[str] = field(default_factory=lambda: SPECIAL_TOKENS.values())
    show_progress: bool = True

    data_iterator: Iterator[str] = None     # need to be set
    output_dir: str = None                  # need to be set
    vocab_size: int = 3000
    length: int = 10000

    def to_dict(self):
        return self.__dict__


@dataclass
class TrainBPETokenizerConfig(TrainTokenizerBaseConfig):
    """Configuration for training BPE tokenizer model
    
    Parameters
    ----------
    min_frequency : int, optional
        the minimum frequency of token, by default 2
    max_token_length : int, optional
        the maximum length of token, by default 16
    """
    min_frequency: int = 2
    max_token_length: int = 16


@dataclass
class TrainUnigramTokenizerConfig(TrainTokenizerBaseConfig):
    """Configuration for training Unigram tokenizer model
    
    Parameters
    ----------
    max_piece_length : int, optional
        the maximum length of token, by default 16
    """
    max_piece_length: int = 16


@dataclass
class TrainSPMTokenizerConfig:
    """Configuration for training SentencePiece tokenizer model

    see `spm.SentencePieceTrainer.Train` from the `sentencepiece` package at 
        https://github.com/google/sentencepiece/blob/master/doc/options.md)

    Parameters
    ----------
    input : str
        the path of input file, must be a txt file
    vocab_size : int
        the vocabulary size
    model_prefix : str, optional
        the model saved name, by default 'spm_vocab'
    model_type : str, optional
        the model type, by default 'unigram'
        could be 'bpe', 'unigram', 'char', 'word'
    character_coverage : float, optional
        see https://github.com/google/sentencepiece/issues/412
    bos_id : int, optional
        the beginning of sentence id, by default 0
    unk_id : int, optional
        the unknown id, by default 1
    eos_id : int, optional
        the end of sentence id, by default 2
    bos_piece : str, optional
        the beginning of sentence piece, by default SPECIAL_TOKENS.BOS.value
    unk_piece : str, optional
        the unknown piece, by default SPECIAL_TOKENS.UNK.value
    eos_piece : str, optional
        the end of sentence piece, by default SPECIAL_TOKENS.EOS.value
    user_defined_symbols : List[str], optional
        the user defined symbols, by default [SPECIAL_TOKENS.MASK.value]
    num_sub_iterations : int, optional
        the number of sub iterations, by default 2
    max_sentencepiece_length : int, optional
        the maximum length of token, by default 16
    max_sentence_length : int, optional
        the maximum length of sentence, by default 20000
    num_threads : int, optional
        the number of threads, by default 64
    """
    input: str
    vocab_size: int
    model_prefix: str = SPM_VOCAB_MODEL_PREFIX
    model_type: str = SPM_MODELS.UNIGRAM.value
    character_coverage: float = 1.0
    bos_id: int = 0
    unk_id: int = 1
    eos_id: int = 2
    bos_piece: str = SPECIAL_TOKENS.BOS.value
    unk_piece: str = SPECIAL_TOKENS.UNK.value
    eos_piece: str = SPECIAL_TOKENS.EOS.value
    user_defined_symbols: List[str] = field(default_factory= lambda: [SPECIAL_TOKENS.MASK.value])
    num_sub_iterations: int = 2
    max_sentencepiece_length: int = 16
    max_sentence_length: int = 20000
    num_threads: int = 64

    def to_dict(self):
        return self.__dict__

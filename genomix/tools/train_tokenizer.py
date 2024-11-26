#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_tokenizer_train_funcs.py
@Time    :   	2024/11/21 00:23:42
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

@Desc    :   	functions for training tokenizer model of BPE, UNIGRAM, SPM

"""

import logging

from ..tokenizers import (
    BioSeqBaseBPETokenizer, 
    BioSeqBaseUnigramTokenizer
)

from ..utils._validator import validate_args
from ..utils.constants import SPECIAL_TOKENS, GENOMIX_CACHE_DATA_DIR

__all__ = [
    'bpe_tokenizer_train',
    'unigram_tokenizer_train',
    'spm_tokenizer_train',
]

logger = logging.getLogger(__name__)


@validate_args(
    name='train_kwargs', 
    required_keys_or_values=['data_iterator','vocab_size'])
def bpe_tokenizer_train(
    train_kwargs: dict
):
    """Train BPE tokenizer model

    Parameters
    ----------
    train_kwargs : dict
        parameters for training tokenizer model

        see `BioSeqBaseBPETokenizer.train_from_iterator`, and  
        https://huggingface.co/docs/tokenizers/en/api/trainers
        
        For example:
        train_kwargs = {
            'data_iterator': Iterator[str],
            'unk_token': str,   # if not provided, the default value is SPECIAL_TOKENS.UNK
            'output_dir': str,  # if not provided, the model will be saved in CACHE_DIR_TMP_DATA
            'vocab_size': 5008, # the size of vocabulary
            'min_frequency': 2, # the minimum frequency of token
            'max_token_length': 16,  # the maximum length of token
            'initial_alphabet': ['A', 'C', 'G', 'T', 'N'], # equal to INITIAL_ALPHABETS
            'special_tokens': ["<BOS>", "<UNK>", "<EOS>", "<MASK>"], # equal to SPECIAL_TOKENS.values()
            'show_progress': True, # show the progress of training
            'length': int, # the number of sequences in the dataset, for progress bar
        }

    """
    data_iterator = train_kwargs.pop('data_iterator')
    unk_token = train_kwargs.pop('unk_token', SPECIAL_TOKENS.UNK.value)
    output_dir = train_kwargs.pop('output_dir', GENOMIX_CACHE_DATA_DIR)

    assert output_dir is not None, "output_dir must be provided"

    logger.info("Training BPE tokenizer starts...")
    _tokenizer = BioSeqBaseBPETokenizer(unk_token=unk_token)
    _tokenizer.train_from_iterator(
        data_iterator,
        **train_kwargs
    )
    _tokenizer.save_model(output_dir)
    logger.info(f"BPE finished, saved vocab at: {output_dir}")


@validate_args(
    name='train_kwargs', 
    required_keys_or_values=['data_iterator', 'vocab_size'])
def unigram_tokenizer_train(train_kwargs: dict):
    """Train Unigram tokenizer model

    Parameters
    ----------
    train_kwargs : dict
        parameters for training tokenizer model

        see `BioSeqBaseUnigramTokenizer.train_from_iterator`, and  
        https://huggingface.co/docs/tokenizers/en/api/trainers
        
        For example:
        train_kwargs = {
            'data_iterator': Iterator[str],
            'unk_token': str, # if not provided, the default value is SPECIAL_TOKENS.UNK
            'output_dir': str, # if not provided, the model will be saved in CACHE_DIR_TMP_DATA
            'vocab_size': 5008,    # the size of vocabulary
            'max_piece_length': 16, # the maximum length of token
            'initial_alphabet': ['A', 'C', 'G', 'T', 'N'],  # equal to INITIAL_ALPHABETS
            'special_tokens': ["<BOS>", "<UNK>", "<EOS>", "<MASK>"],  # equal to SPECIAL_TOKENS.values()
            'show_progress': True, # show the progress of training
            'length': int, # the number of sequences in the dataset
        }
    """

    data_iterator = train_kwargs.pop('data_iterator')
    output_dir = train_kwargs.pop('output_dir', GENOMIX_CACHE_DATA_DIR)

    assert output_dir is not None, "output_dir must be provided"

    logger.info("Training UNIGRAM tokenizer starts...")
    _tokenizer = BioSeqBaseUnigramTokenizer()
    _tokenizer.train_from_iterator(
        data_iterator,
        **train_kwargs
    )
    _tokenizer.save_model(output_dir)
    logger.info(f"UNIGRAM finished, saved vocab at: {output_dir}")


@validate_args(
    name='train_kwargs', 
    required_keys_or_values=['input', 'vocab_size', 'model_type'])
def spm_tokenizer_train(train_kwargs: dict):
    """Train SentencePiece tokenizer model

    Parameters
    ----------
    train_kwargs : dict
        parameters for training tokenizer model

        see `spm.SentencePieceTrainer.Train` from the `sentencepiece` package at 
            https://github.com/google/sentencepiece/blob/master/doc/options.md)
        
        For example:
        train_kwargs = {
            'input': 'path/to/corpus.txt', # input file path, must be .txt file
            'vocab_size': 5008,
            'model_prefix': 'spm_vocab',   # the model saved name
            'model_type': 'unigram',       # could be 'bpe', 'unigram', 'char', 'word'
            'character_coverage': 1.0,     # see https://github.com/google/sentencepiece/issues/412
            'bos_id': 0,
            'unk_id': 1,
            'eos_id': 2,
            'bos_piece': '<BOS>',               # equal to SPECIAL_TOKENS.BOS
            'unk_piece': '<UNK>',               # equal to SPECIAL_TOKENS.UNK
            'eos_piece': '<EOS>',               # equal to SPECIAL_TOKENS.EOS
            'user_defined_symbols':['MASK'],    # equal to SPECIAL_TOKENS.MASK
            'num_sub_iterations': 2,
            'max_sentencepiece_length': 16,     # the maximum length of token
            'max_sentence_length': 20000,       # the maximum length of sentence
            'num_threads': 64,              # the number of threads
            'train_extremely_large_corpus': False, # default value
        }
    """
    import sentencepiece as spm

    # enforce the default values
    train_kwargs.setdefault("add_dummy_prefix", False) # https://github.com/google/sentencepiece/issues/488
    train_kwargs.setdefault("train_extremely_large_corpus", True)

    logger.info(f"Training SPM with starts...")
    # Train parameters, see more details:
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    spm.SentencePieceTrainer.Train(
        **train_kwargs
    )
    logger.info(f"SPM finished, saved vocab at: {train_kwargs['model_prefix']}.model")



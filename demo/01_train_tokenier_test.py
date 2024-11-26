#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		01_train_tokenier_test.py
@Time    :   	2024/11/26 16:06:22
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

@Desc    :   	 None

"""

import os
import sys
from pathlib import Path
import logging

from datasets import load_from_disk

sys.path.append(str(Path(__file__).parents[1]))

from genomix.tools import (
    TrainBPETokenizerConfig,
    TrainUnigramTokenizerConfig,
    TrainSPMTokenizerConfig,

    bpe_tokenizer_train, 
    unigram_tokenizer_train,
    spm_tokenizer_train,

    write_dataset_to_txt,
    load_dataset_to_dataset
)

from genomix.utils.constants import (
    TOKENIZER_MODELS,
    SPECIAL_TOKENS,
    INITIAL_ALPHABETS,
    SPM_VOCAB_MODEL_PREFIX,
    SPM_MODELS,
)

from genomix.utils import (
    batch_iterator
)


logger = logging.getLogger(__name__)

home_dir = os.path.expanduser('~')

def test_train_tokenizer(method):

    # build data_iterator
    ds, feat_name = load_dataset_to_dataset(
        os.path.join(home_dir, "projects/genomix/tmp/testdata"),
        input_ds_key = 'train',
        input_ds_feature = 'sequence',
        max_num_examples=0,
        chunk_size=12,
        overlap_step=0,
        map_batch_size=2,
        num_proc=2
    )
    length = len(ds)

    data_iterator = batch_iterator(ds,ds_feature_name=feat_name, batch_size=2)
    output_dir = os.path.join(home_dir, "projects/genomix/tmp/vocab")

    if method == TOKENIZER_MODELS.BPE:
        config = TrainBPETokenizerConfig(
            unk_token = SPECIAL_TOKENS.UNK.value,
            initial_alphabet = INITIAL_ALPHABETS,
            special_tokens = SPECIAL_TOKENS.values(),
            show_progress = True,
            data_iterator = data_iterator,
            output_dir = output_dir,
            vocab_size=16,
            length = length,

            min_frequency = 2,
            max_token_length = 6,
        )
        bpe_tokenizer_train(config.to_dict())

    elif method == TOKENIZER_MODELS.UNIGRAM:
        config = TrainUnigramTokenizerConfig(
            unk_token = SPECIAL_TOKENS.UNK.value,
            initial_alphabet = INITIAL_ALPHABETS,
            special_tokens = SPECIAL_TOKENS.values(),
            show_progress = True,
            data_iterator = data_iterator,
            output_dir = output_dir,
            vocab_size=16,
            length = length,

            max_piece_length = 6,
        )
        unigram_tokenizer_train(config.to_dict())
    elif method == TOKENIZER_MODELS.SPM:

        output_train_txt_fname = os.path.join(home_dir, "projects/genomix/tmp/train.txt")
        output_spm_vocab_dir = os.path.join(home_dir, "projects/genomix/tmp/vocab")

        write_dataset_to_txt(
            os.path.join(home_dir, "projects/genomix/tmp/testdata"), 
            output_train_txt_fname, 
            
            input_ds_key = 'train',
            input_ds_feature = 'sequence',
            max_num_examples=0,
            chunk_size=12,
            overlap_step=0,
            map_batch_size=2,
            num_proc=2,
            disable_tqdm = False)

        config = TrainSPMTokenizerConfig(
            input = output_train_txt_fname,
            vocab_size=16,
            model_prefix = os.path.join(output_spm_vocab_dir, SPM_VOCAB_MODEL_PREFIX),
            model_type = SPM_MODELS.UNIGRAM.value,
            character_coverage = 1.0,
            bos_id = 0,
            unk_id = 1,
            eos_id = 2,
            bos_piece = SPECIAL_TOKENS.BOS.value,
            unk_piece = SPECIAL_TOKENS.UNK.value,
            eos_piece = SPECIAL_TOKENS.EOS.value,
            user_defined_symbols = [SPECIAL_TOKENS.MASK.value],
            num_sub_iterations = 2,
            max_sentencepiece_length = 6,
            max_sentence_length = 20000,
            num_threads = 10,
            train_extremely_large_corpus = False,
        )
        spm_tokenizer_train(config.to_dict())
    else:    
        logger.error(f"Unsupported tokenizer model: {method}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    test_train_tokenizer(TOKENIZER_MODELS.SPM)
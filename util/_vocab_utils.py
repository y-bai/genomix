#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_vocab_utils.py
@Time    :   	2024/11/15 14:38:10
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

import os
from typing import List
from ._common_utils import (
    check_file_exist, 
    check_dir_exist, 
    read_json, 
    write_json,
    partition
)

def update_vocab(
        input_dir: str,
        output_dir: str,
        tokenizer_type: str="BPE",
        vocab_fname: List[str]=["vocab.json", "merges.txt"],
        special_tokens: List[str]=["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
        new_vocab_size_exclude_special_token: int=5000
):
    """
    Update the vocabulary file for the model.

    Args:
    - input_dir: str, the model name or path
        the dictionary name that contains the vocabulary files

    - output_dir: str, the output directory

    - tokenizer_type: str, the tokenizer type, default is "BPE"
        Could be "BPE" | "UNIGRAM" | "SPM"
        NOTE: For "SPM", we only consider the SPM-unigram tokenizer for now.

    - vocab_fname: List[str], the vocabulary file name, default is ["vocab.json", "merges.txt"]
        For BPE tokenizer, the vocabulary file is "vocab.json" and "merges.txt"
        For UNIGRAM tokenizer, the vocabulary file is "unigram.json"
        For SPM tokenizer, the vocabulary file is "spm_vocab.model" and "spm_vocab.vocab"

    - special_tokens: List[str], the special tokens, default is ["<BOS>", "<UNK>", "<EOS>", "<MASK>"]
        Could be None if you do not want to update the special tokens

    - new_vocab_size_exclude_special_token: int, the new vocabulary size, default is 5000
        The new vocabulary size after updating the vocabulary file.
        NOTE: We can always input the vocal file with the largest size (in our case, the largest vocab size is 200000), 
            and the function will return the new vocab file with 
            the first `new_vocab_size_exclude_special_token` number of tokens.
        
        NOTE: This number DOES NOT contain the any special tokens.
    """

    check_dir_exist(input_dir, create=False)
    check_dir_exist(output_dir, create=True)

    if tokenizer_type == "BPE":
        
        vocab_json_fname, merge_txt_fname = vocab_fname[0], vocab_fname[1]
        if not vocab_json_fname.endswith(".json") or not merge_txt_fname.endswith(".txt"):
            raise ValueError(f"Invalid vocab_fname: {vocab_fname}. It should be [.json, .txt]")

        vocab_file = os.path.join(input_dir, vocab_json_fname)
        merge_file = os.path.join(input_dir, merge_txt_fname)
        check_file_exist(vocab_file)
        check_file_exist(merge_file)

        # read the vocab file
        vocab_json = read_json(vocab_file)
        # partition the vocab file into special tokens and regular tokens
        original_special_tokens, regular_tokens = partition(lambda x: x.startswith("<"), vocab_json.keys())
        n_original_special_token = len(original_special_tokens)

        # update the token_keys
        token_keys = special_tokens if special_tokens is not None else original_special_tokens + regular_tokens[:new_vocab_size_exclude_special_token]
        
        OrderedDict


        if os.path.exists(vocab_file):
            with open(vocab_file, 
                        mode='rt', 
                        encoding="utf-8"
                ) as vocab_f:    
                vocab_json = json.load(vocab_f, object_pairs_hook=OrderedDict)
            
            # remove the first 7 keys, ie, ["<CLS>", "<PAD>", "<SEP>", "<UNK>", "<MASK>", "<BOS>", "<EOS>"]
            # and add the updated keys 
            keys_list = ["<BOS>", "<UNK>", "<EOS>", "<MASK>","<PAD>"] + list(vocab_json.keys())[n_original_special_token:]
            vals = list(range(len(keys_list)))
            update_vocab = OrderedDict(zip(keys_list, vals))

            with open(vocab_file, 
                        mode='w', 
                        encoding="utf-8"
                ) as vocab2_f:    
                json.dump(update_vocab, vocab2_f)
    
    if token_type == "Unigram":
        vocab_file = os.path.join(token_dir, 'unigram.json')
        if os.path.exists(vocab_file): 
            with open(vocab_file, 
                    mode='rt', 
                    encoding="utf-8"
                ) as vocab_f:
                
                vocab_json = json.load(vocab_f, object_pairs_hook=OrderedDict)
            vocab_json.update(
                {
                    'unk_id': 1, 
                    'vocab': [['<BOS>', 0.0], ['<UNK>', 0.0], ['<EOS>', 0.0], ['<MASK>', 0.0],['<PAD>', 0.0]] + vocab_json['vocab'][n_original_special_token:]
                }
            )
            with open(vocab_file, 
                    mode='w', 
                    encoding="utf-8"
                ) as vocab2_f:    
                json.dump(vocab_json, vocab2_f)


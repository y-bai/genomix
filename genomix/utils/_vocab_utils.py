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

@Desc    :   	update vaoabulary file for tokenizer model of BPE, UNIGRAM, and SPM

"""
import os
from typing import List, Optional, Tuple, Union
from collections import OrderedDict
from functools import wraps

from ._common_utils import (
    check_file_exists, 
    check_dir_exists, 
    read_json, 
    write_json,
    read_txt,
    write_txt,
    partition
)
from ._constants import SPECIAL_TOKENS


def _bpe_vocab_update(
    input_dir: str,
    output_dir: str,
    vocab_fname: Union[List[str], Tuple[str, ...]]=["vocab.json", "merges.txt"],
    new_special_tokens: Optional[List[str]]=["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
    new_vocab_size: int=5009,
    **kwargs
):
    if len(vocab_fname) < 2:
        raise ValueError(f"Invalid vocab_fname: {vocab_fname}. It should be [vocab.json, merges.txt]")
    vocab_json_fname, merge_txt_fname = vocab_fname[0], vocab_fname[1]

    if not vocab_json_fname.endswith(".json") or not merge_txt_fname.endswith(".txt"):
        raise ValueError(f"Invalid vocab_fname: {vocab_fname}. It should be [.json, .txt]")

    vocab_file = os.path.join(input_dir, vocab_json_fname)
    merge_file = os.path.join(input_dir, merge_txt_fname)
    check_file_exists(vocab_file)
    check_file_exists(merge_file)

    print(f"Start updating the vocabulary file for BPE tokenizer.")

    # read the original vocab file
    vocab_json = read_json(vocab_file)
    # partition the original vocab file into special tokens and regular tokens
    original_special_tokens, regular_tokens = partition(lambda x: x.startswith("<"), list(vocab_json.keys()))
    # partition the regular tokens into initial alphabet (e.g., ['A', 'C', 'G', 'T', 'N']) and real tokens
    init_alphabet, real_tokens = partition(lambda x: len(x)==1, regular_tokens)

    # update the token_keys
    special_tokens = new_special_tokens if new_special_tokens is not None else original_special_tokens

    n_special_token = len(special_tokens)
    n_init_aplphabet = len(init_alphabet)
    n_original_real_tokens = len(real_tokens)

    if 0 < new_vocab_size <= (n_special_token + n_init_aplphabet):
        raise ValueError(f"new_vocab_size should be greater than {n_special_token + n_init_aplphabet} if it is not 0.")

    new_real_vocab_size = (n_original_real_tokens 
                            if new_vocab_size == 0 
                            else new_vocab_size - n_special_token - n_init_aplphabet)
    
    token_keys = special_tokens + init_alphabet + real_tokens[:new_real_vocab_size] 
    
    # write the updated vocab file into .json file
    update_vocab = OrderedDict({k:i for i, k in enumerate(token_keys)})
    output_fname = os.path.join(output_dir, vocab_json_fname)
    write_json(output_fname, update_vocab)

    # update merge.txt file
    merge_lines = read_txt(merge_file)

    output_fname = os.path.join(output_dir, merge_txt_fname)
    write_txt(output_fname, merge_lines[:new_real_vocab_size + 1]) # the first line is the head such as #version: 0.2

    print(f"DONE. Updated vocab file is saved to {output_dir} for BPE tokenizer.")
    

def _unigram_vocab_update(
    input_dir: str,
    output_dir: str,
    vocab_fname: Union[str, List[str], Tuple[str, ...]]=["unigram.json"],
    new_special_tokens: Optional[List[str]]=["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
    new_vocab_size: int=5004,
    **kwargs
):
    if isinstance(vocab_fname, str):
        vocab_json_fname = vocab_fname
    else:
        vocab_json_fname = vocab_fname[0]
    if not vocab_json_fname.endswith(".json"):
        raise ValueError(f"Invalid vocab_fname: {vocab_fname}. It should be [.json]")
    
    vocab_file = os.path.join(input_dir, vocab_json_fname)
    check_file_exists(vocab_file)

    print(f"Start updating the vocabulary file for UNIGRAM tokenizer.")

    vocab_json = read_json(vocab_file)
    vocab_part = vocab_json['vocab']

    # find out the original special tokens
    original_special_tokens = []
    original_real_tokens = []
    for i, token_ in enumerate(vocab_part):
        if token_[0].startswith("<"):
            original_special_tokens.append(token_)
        else:
            original_real_tokens.append(token_)
    
    if new_special_tokens is not None:
        new_special_tokens_in = []
        new_unk_id = -100000000
        for i, token_ in enumerate(new_special_tokens):
            new_special_tokens_in.append([token_, 0.0])
            if token_ == "<UNK>":
                new_unk_id = i
        # if no "<UNK>" in new_special_tokens
        if new_unk_id == -100000000:
            print(">>>As new_special_tokens DOES NOT contain '<UNK>', additinally add '<UNK>' to new_special_tokens.")
            new_special_tokens_in = new_special_tokens_in + [["<UNK>", 0.0]] 
            new_unk_id = len(new_special_tokens_in) - 1
    else:
        new_special_tokens_in = original_special_tokens
        new_unk_id = vocab_json['unk_id']
    
    n_new_special_tokens_in = len(new_special_tokens_in)

    if 0 < new_vocab_size <= n_new_special_tokens_in:
        raise ValueError(f"new_vocab_size should be greater than {n_new_special_tokens_in} if it is not 0.")
    
    new_real_vocab_size = (len(original_real_tokens) 
                            if new_vocab_size == 0 
                            else new_vocab_size - n_new_special_tokens_in)
    
    # update the vocab file
    vocab_json.update(
        {
            'unk_id': new_unk_id, 
            'vocab': new_special_tokens_in + original_real_tokens[:new_real_vocab_size]
        }
    )

    output_fname = os.path.join(output_dir, vocab_json_fname)
    write_json(output_fname, vocab_json)
    print(f"DONE. Updated vocab file is saved to {output_dir} for UNIGRAM tokenizer.")


def _spm_vocab_update(
    input_dir: str,
    output_dir: str,
    vocab_fname: Union[List[str], Tuple[str, ...]]=["spm_vocab.model", "spm_vocab.vocab"],
    new_special_tokens: Optional[List[str]]=["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
    new_vocab_size: int=5004,
    **kwargs
):
    """

    ref:     
    # https://ddimri.medium.com/sentencepiece-the-nlp-architects-tool-for-building-bridges-between-languages-7a0b8ae53130
    # https://blog.ceshine.net/post/trim-down-sentencepiece-vocabulary/
    # https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865

    Parameters
    ----------
    input_dir : str
        _description_
    output_dir : str
        _description_
    vocab_fname : Union[List[str], Tuple[str, ...]], optional
        _description_, by default ["spm_vocab.model", "spm_vocab.vocab"]
    new_special_tokens : Optional[List[str]], optional
        _description_, by default ["<BOS>", "<UNK>", "<EOS>", "<MASK>"]
    new_vocab_size : int, optional
        _description_, by default 5004

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """

    if new_vocab_size > 0 and new_special_tokens is not None:
        assert new_vocab_size > len(new_special_tokens), f"new_vocab_size should be greater than {len(new_special_tokens)} if it is not 0."
    
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model

    if len(vocab_fname) < 2:
        raise ValueError(f"Invalid vocab_fname: {vocab_fname}. It should be [spm_vocab.model, spm_vocab.vocab]")
    vocab_model_fname, vocab_txt_fname = vocab_fname[0], vocab_fname[1]

    if not vocab_model_fname.endswith(".model") or not vocab_txt_fname.endswith(".vocab"):
        raise ValueError(f"Invalid vocab_fname: {vocab_fname}. It should be [.model, .vocab]")
    
    vocab_model_file = os.path.join(input_dir, vocab_model_fname)
    vocab_txt_file = os.path.join(input_dir, vocab_txt_fname)
    check_file_exists(vocab_model_file)
    check_file_exists(vocab_txt_file)

    print(f"Start updating the vocabulary file for SPM tokenizer.")
    # default spm_kwargs values:
    # see: https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/__init__.py#L471
    # spm_kwargs = {
    #     "add_bos": False,
    #     "add_eos": False,
    #     "reverse": False,
    #     "emit_unk_piece": False,
    #     "enable_sampling": False, # default is False, if Ture, then enconded str is different every time
    #     "nbest_size": -1,
    #     "alpha":0.16,
    # }
    spm_model = spm.SentencePieceProcessor()
    spm_model.Load(vocab_model_file)

    original_vocab_size = spm_model.GetPieceSize()
    original_vocab = [spm_model.IdToPiece(i) for i in range(original_vocab_size)]
    original_special_tokens, regular_tokens = partition(lambda x: x.startswith("<"), original_vocab)

    new_spm_model_proto = sp_pb2_model.ModelProto()
    new_spm_model_proto.ParseFromString(open(vocab_model_file, "rb").read())

    n_new_vocab_size = original_vocab_size if new_vocab_size == 0 else new_vocab_size
    
    if new_special_tokens is None:
        n_remove_vocab_size = original_vocab_size - n_new_vocab_size
        print(f"Updating/removing {n_remove_vocab_size} tokens from the end of the vocabulary file...")
        for _ in range(n_remove_vocab_size):
            new_spm_model_proto.pieces.pop()
    else:  
        if SPECIAL_TOKENS.UNK.value not in new_special_tokens:
            print(">>>As new_special_tokens DOES NOT contain '<UNK>', additinally add '<UNK>' to new_special_tokens.")
            new_special_tokens = new_special_tokens + [SPECIAL_TOKENS.UNK.value]
        
        add_new_special_tokens = [t for t in new_special_tokens if t not in original_special_tokens]

        n_remove_vocab_size = original_vocab_size - n_new_vocab_size + len(add_new_special_tokens)
        print(f"Removing {n_remove_vocab_size} tokens from the end of the vocabulary file...")
        for _ in range(n_remove_vocab_size):
            new_spm_model_proto.pieces.pop()
        
        print(f"Adding {len(add_new_special_tokens)} new special tokens: {add_new_special_tokens}")
        # update/add special tokens
        for special_token_ in add_new_special_tokens:
            new_special_piece_ = sp_pb2_model.ModelProto().SentencePiece()
            new_special_piece_.piece = special_token_
            new_special_piece_.score = 0
            if special_token_ == SPECIAL_TOKENS.UNK.value:
                new_special_piece_.type = 2
            # new_special_piece_.type = 3  # sp_pb2_model.ModelProto().SentencePiece().CONTROL 
            # sp_pb2_model.ModelProto().SentencePiece().NORMAL,         = 1 
            # sp_pb2_model.ModelProto().SentencePiece().UNKNOWN,        = 2
            # sp_pb2_model.ModelProto().SentencePiece().CONTROL,        = 3
            # sp_pb2_model.ModelProto().SentencePiece().USER_DEFINED,   = 4
            # sp_pb2_model.ModelProto().SentencePiece().UNUSED          = 5
            # sp_pb2_model.ModelProto().SentencePiece().BYTE            = 6
            new_spm_model_proto.pieces.append(new_special_piece_)
        
    # write the updated vocab file into .model file
    output_fname = os.path.join(output_dir, vocab_model_fname)
    with open(output_fname, "wb") as f:
        f.write(new_spm_model_proto.SerializeToString())

    new_vocab_pieces = []
    for p in new_spm_model_proto.pieces:
        new_vocab_pieces.append(f"{p.piece}\t{p.score}\n")
    output_fname = os.path.join(output_dir, vocab_txt_fname)
    write_txt(output_fname, new_vocab_pieces)
    
    print(f"DONE. Updated vocab file is saved to {output_dir} for SPM tokenizer.")


# Define the decorator
def vocab_update(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the original update_vocab function
        func(*args, **kwargs)
        
        # Then call the appropriate vocab update function
        tokenizer_model = kwargs.get('tokenizer_model')
        
        if tokenizer_model == 'BPE':
            return _bpe_vocab_update(*args, **kwargs)
        elif tokenizer_model == 'UNIGRAM':
            return _unigram_vocab_update(*args, **kwargs)
        elif tokenizer_model == 'SPM':
            return _spm_vocab_update(*args, **kwargs)
        else:
            raise ValueError(f"Unknown tokenizer model: {tokenizer_model}")
    
    return wrapper


@vocab_update
def update_vocab(
        input_dir: str,
        output_dir: str,
        tokenizer_model: str="BPE",
        vocab_fname: Union[List[str], Tuple[str, ...]]=["vocab.json", "merges.txt"],
        new_special_tokens: Optional[List[str]]=["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
        new_vocab_size: int=5009,
):
    """
    Update the vocabulary file for the model. We only consider the BPE, UNIGRAM, and SPM tokenizer for now.

        Here, we shrink the vocabulary size by removing the tokens from the end of the vocabulary file for the UNIGRAM and SPM tokenizer.
        We DO NOT support adding new token into vocab. 
        If you want to add new tokens, you should use function `add_token` in tokenizer model.

    Args:
    - input_dir: str, the model name or path
        the dictionary name that contains the vocabulary files

    - output_dir: str, the output directory

    - tokenizer_model: str, the tokenizer type, default is "BPE"
        Could be "BPE" | "UNIGRAM" | "SPM"
        NOTE: For "SPM", we only consider the SPM-unigram tokenizer for now.

    - vocab_fname: List[str], the vocabulary file name, default is ["vocab.json", "merges.txt"]
        For BPE tokenizer, the vocabulary file is "vocab.json" and "merges.txt"
        For UNIGRAM tokenizer, the vocabulary file is "unigram.json"
        For SPM tokenizer, the vocabulary file is "spm_vocab.model" and "spm_vocab.vocab"

    - new_special_tokens: List[str], the new special tokens, default is ["<BOS>", "<UNK>", "<EOS>", "<MASK>"]
        Could be None if you do not want to update the special tokens and use the original ones.

    - new_vocab_size: int, the new vocabulary size, default is 5009
        The new vocabulary size after updating the vocabulary file.
        Set 0 if you keep the original vocabulary size.

        NOTE: This number (e.g., 5009) include the `new_special_tokens` (e.g., 4) and the `initial_alphabet` (e.g., ['A', 'C', 'G', 'T', 'N']).
            The `initial_alphabet` has been set, see function `train` or `train_from_iterator` in `tokenizer.BioSeqBaseBPETokenizer` for more details.
            The `initial_alphabet` was ONLY used by the BPE tokenizer.
            
            Therefore, For example,
            For BPE, if the original vocabulary size is 5009, and the `new_special_tokens` is ["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
                and the `initial_alphabet` is ['A', 'C', 'G', 'T', 'N'], then the number of real token should be 5009 - 4 - 5 = 5000.
            For UNIGRAM, if the original vocabulary size is 5009, and the `new_special_tokens` is ["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
                then the number of real token should be 5009 - 4 = 5005. This because the UNIGRAM tokenizer does not have the `initial_alphabet`.
                See function `train` or `train_from_iterator` in `tokenizer.BioSeqBaseUnigramTokenizer` for more details.


    """

    check_dir_exists(input_dir, create=False)
    check_dir_exists(output_dir, create=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer model: {tokenizer_model}")
    print(f"Vocabulary file name: {vocab_fname}")
    print(f"New special tokens: {new_special_tokens}")
    print(f"New vocabulary size: {new_vocab_size}")
    print(f"{'-' * 20}")


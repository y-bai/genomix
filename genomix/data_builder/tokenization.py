#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		tokenization.py
@Time    :   	2024/12/11 15:13:53
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

@Desc    :   	 tokenization for seqsence when training/evaluating model

"""

import os
import copy
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import json
import multiprocessing as mp
from tqdm import tqdm
from datasets import IterableDataset, Dataset, IterableDatasetDict, DatasetDict

from ..tokenizers import (
    get_tokenizer_cls, 
    PRETRAINED_MODEL_NAME_CLS_MAP,
    BioSeqTokenizerMap,
)
from ..utils.constants import SPECIAL_TOKENS, INITIAL_ALPHABETS

logger = logging.getLogger(__name__)

class TokenizationConfig:

    def to_dict(self):
        return self.__dict__

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, dict_obj: Dict[str, Any]):
        return cls(**dict_obj)

    @classmethod
    def from_json_string(cls, json_string: str):
        return cls.from_dict(json.loads(json_string))
    
    @classmethod
    def from_json_file(cls, json_file):
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)
    
    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_json_string()})"

    def __str__(self):
        return self.__repr__()
    

class GenoMixTokenizationConfig(TokenizationConfig):

    def __init__(
        self,
        tokenizer_type: str = 'CHAR_TOKEN',
        tokenizer_pretrained_model_path: Optional[str] = None,
        # for char tokenization
        chars = INITIAL_ALPHABETS,

        # for tokenizer initialization
        model_max_length: Optional[int] = 1024,
        padding_side: Optional[str] = "left",
        truncation_side: Optional[str] = "right",
        bos_token: Optional[str] = SPECIAL_TOKENS.BOS.value,
        eos_token: Optional[str] = SPECIAL_TOKENS.EOS.value,
        unk_token: Optional[str] = SPECIAL_TOKENS.UNK.value,
        mask_token: Optional[str] = SPECIAL_TOKENS.MASK.value,
        add_bos_token: Optional[bool] = False,
        add_eos_token: Optional[bool] = False,
        add_prefix_space: Optional[bool] = False,
        do_lower_case: Optional[bool] = False,
    ):

        super().__init__()

        assert tokenizer_type in PRETRAINED_MODEL_NAME_CLS_MAP.keys(), f"tokenizer_type must be one of {PRETRAINED_MODEL_NAME_CLS_MAP.keys()}"
        
        self.tokenizer_type = tokenizer_type
        self.tokenizer_pretrained_model_path = tokenizer_pretrained_model_path

        self.chars = chars

        self.model_max_length = model_max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.add_prefix_space = add_prefix_space
        self.do_lower_case = do_lower_case


class GenoMixTokenization:

    def __init__(
        self, 
        config: GenoMixTokenizationConfig
    ):
        """_summary_

        Parameters
        ----------
        config : GenoMixTokenizationConfig
            configuration for tokenization

        """
        config_dict = copy.deepcopy(config).to_dict()
        tokenizer_type = config_dict.pop("tokenizer_type", None)
        tokenizer_pretrained_model_path = config_dict.pop("tokenizer_pretrained_model_path", None)
        chars = config_dict.pop("chars", None)

        assert tokenizer_type is not None, "tokenizer_type must be provided"
        
        if tokenizer_pretrained_model_path is not None:
            _tokenizer = get_tokenizer_cls(tokenizer_type).from_pretrained(
                pretrained_model_name_or_path = tokenizer_pretrained_model_path,
                local_files_only=True,
                **config_dict)
        elif tokenizer_type == "CHAR_TOKEN" and chars is not None:
            _tokenizer = get_tokenizer_cls(tokenizer_type)(
                characters = chars, 
                **config_dict
            )
        else:
            raise ValueError(f"If tokenizer_pretrained_model_path in {config_dict} is not provided,  " +
                            "`tokenizer_type` must be CHAR_TOKEN, and `chars` must be provided")

        # we treat the pad_token as the eos_token
        _tokenizer.pad_token = _tokenizer.eos_token

        self.tokenizer = _tokenizer


    def get_tokenizer(self):
        return self.tokenizer
    
    
    def tokenize_with_dataset(
        self,
        input_dataset: Union[IterableDatasetDict, DatasetDict, IterableDataset, Dataset],
        column_names: Optional[List[str]] = None,
        sequence_column_name: Optional[str] = None,
        # for tokenizer.map
        padding: Optional[str] = "max_length",
        truncation: Optional[bool] = True,
        stride: Optional[int] = 16, # overlap between chunks
        return_overflowing_tokens: Optional[bool] = True,
        token_min_ctx_fraction: Optional[float] = 1.0,
        use_streaming: Optional[bool] = False, # used for IterableDataset or IterableDatasetDict
        map_num_proc: Optional[int] = 16,
        overwrite_cache: Optional[bool] = True,
    ):
        return BioSeqTokenizerMap(
            self.tokenizer,
            max_length=self.tokenizer.model_max_length,
            stride=stride,
            min_len_frac=token_min_ctx_fraction,
            streaming=use_streaming,
        ).do_map(
            input_dataset,
            dataset_col_remove = column_names,
            dataset_col_tokenize = sequence_column_name,
            padding = padding,
            truncation=truncation,
            return_overflowing_tokens = return_overflowing_tokens,
            load_from_cache_file = not overwrite_cache,  # not used in streaming mode
            num_proc = map_num_proc,  # not used in streaming mode
        ).get_chunked_tokenized_dataset()
    
    def tokenize_with_text_file(
        self,
        input_txt_fname: str, # a very long sequence
        batch_size: Optional[int] = None,
        stride: Optional[int] = 16, # overlap between chunks
        num_proc: Optional[int] = 16,
        token_min_ctx_fraction: Optional[float] = 1.0,
        disable_tqdm: Optional[bool] = False,
        save_path: Optional[str] = None,
        save_file_prefix: Optional[str] = None,
        remove_existing_output_files: Optional[bool] = True,
        save_attn_mask: Optional[bool] = True,
    ):
        """Tokenize a text file in parallel.
        
        Parameters
        ----------
        input_txt_fname : str
            input text file name

        batch_size : int, optional
            batch size for tokenization, by default None
            - if None, batch_size = total_lines // num_proc, where total_lines = total number of lines in the input file

        stride : int, optional
            overlap between chunks, by default 16

        num_proc : int, optional
            number of processes for tokenization, by default 16

        token_min_ctx_fraction : float, optional
            minimum tokenized sequence length fraction, by default 1.0

        disable_tqdm : bool, optional
            disable tqdm, by default False

        save_path : str, optional
            save path for tokenized output files, by default None

        save_file_prefix : str, optional
            save file prefix, by default None
            
        remove_existing_output_files : bool, optional
            remove existing output files, by default True
        
        save_attn_mask : bool, optional
            save attention mask, by default True
        
        """

        assert save_path is not None, "save_path must be provided"
        save_file_prefix = save_file_prefix + '-' if save_file_prefix is not None else ""

        input_ids_fname = os.path.join(save_path, f"{save_file_prefix}input_ids.txt")
        attention_mask_fname = os.path.join(save_path, f"{save_file_prefix}attention_mask.txt")

        if remove_existing_output_files:
            if os.path.exists(input_ids_fname):
                os.remove(input_ids_fname)
            if os.path.exists(attention_mask_fname):
                os.remove(attention_mask_fname)

        # read the text file
        """Read a large text file in parallel."""
        # Determine the total number of lines in the file
        with open(input_txt_fname, "rt", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        
        logger.info(f"Total number of lines in the input file: {total_lines}")

        batch_size = total_lines // num_proc if batch_size is None else batch_size
        if batch_size < 1:
            logger.warning(f"batch_size is less than 1, setting batch_size = 1, and num_proc = total_lines ({total_lines})")
            batch_size = 1
            num_proc = total_lines
        # Create chunks based on line numbers
        batchs = [(input_txt_fname, i, batch_size) for i in range(0, total_lines, batch_size)]

        logger.info(f"Total number of batches: {len(batchs)}, batch_size={batch_size}, with number of processes: {num_proc}")
        # resulting: batches[0] = batch_size  * len(data[0])
        # Use multiprocessing to tokenize each chunk
        with mp.Pool(processes=num_proc) as pool:
            results = pool.starmap(
                _tokenize_chunk_txt, [
                    (*batch, self.tokenizer, stride, token_min_ctx_fraction) for batch in 
                    tqdm(batchs, desc=f"Running tokenizer on text file", disable=disable_tqdm)
                ]
            ) 
        
        logger.info(f"Total number of chunks: {len(results)}, writing to output files...")
        total_tokenized_lines = 0
        if save_attn_mask:
            with open(input_ids_fname, 'a+', encoding='utf-8') as f_input_ids, open(attention_mask_fname, 'a+', encoding='utf-8') as f_mask_attn:
                for result in results:
                    total_tokenized_lines += len(result["input_ids"])
                    for i_input_ids in result["input_ids"]:
                        f_input_ids.write(','.join(map(str, i_input_ids)) + '\n')
                    for i_mask_attn in result["attention_mask"]:
                        f_mask_attn.write(','.join(map(str, i_mask_attn)) + '\n')
        else:
            with open(input_ids_fname, 'a+', encoding='utf-8') as f_input_ids:
                for result in results:
                    total_tokenized_lines += len(result["input_ids"])
                    for i_input_ids in result["input_ids"]:
                        f_input_ids.write(','.join(map(str, i_input_ids)) + '\n')
        del results
        logger.info(f"Tokenized output files (lines = {total_tokenized_lines}) are saved to {save_path}")

        # Combine the results
        # tokenized_text = {'input_ids': [], 'attention_mask': []}
        # for result in results:
        #     tokenized_text['input_ids'].extend(result['input_ids'])
        #     tokenized_text['attention_mask'].extend(result['attention_mask'])

        # del results
        # return tokenized_text  

def _tokenize_chunk_txt(
        input_txt_fname: str,
        start_line: int,
        num_lines: int, 
        tokenizer,
        stride: int,
        min_len_frac: float,
    ):
    """Tokenize a chunk of lines."""

    # Read the chunk of lines
    batch = []
    with open(input_txt_fname, "rt", encoding="utf-8") as f:
        for current_line, line in enumerate(f):
            if current_line >= start_line and current_line < start_line + num_lines:
                batch.append(line.strip())
            elif current_line >= start_line + num_lines:
                break

    bos_token_id = [tokenizer.bos_token_id] if tokenizer.add_bos_token else []
    eos_token_id = [tokenizer.eos_token_id] if tokenizer.add_eos_token else []

    bos_attn_mask = [1] if tokenizer.add_bos_token else []
    eos_attn_mask = [1] if tokenizer.add_eos_token else []
    
    if tokenizer.is_fast:
        tokenized_batch = tokenizer(
            batch,
            max_length=tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            stride=stride,
            add_special_tokens=True,
        )
        list_keys = list(tokenized_batch.keys())
        assert (set(['input_ids', 'attention_mask']).intersection(set(list_keys))==set(['input_ids', 'attention_mask']), 
                "Only support model_input_names = ['input_ids', 'attention_mask'].")
        
        min_tokenized_seq_len = min_len_frac * (tokenizer.model_max_length - len(bos_token_id + eos_token_id))
        # filter out the chunks with length less than min_len_frac
        for i, attention_mask in tokenized_batch["attention_mask"]:
                if sum(attention_mask) < min_tokenized_seq_len:
                    tokenized_batch["input_ids"].pop(i)
                    tokenized_batch["attention_mask"].pop(i)
        return tokenized_batch
    
    else:
        tokenized_batch = tokenizer(
            batch,
            max_length=tokenizer.model_max_length,
            padding=False,
            truncation=False,
            return_overflowing_tokens=False,
            stride=0,
            add_special_tokens=False,
        )

        concat_token_ids = {k: list(chain(*tokenized_batch[k])) for k in tokenized_batch.keys()}
        # tokenized_batch.keys(): ['input_ids', 'attention_mask']

        concat_token_ids_length = len(concat_token_ids[list(tokenized_batch.keys())[0]])

        _chunk_length = tokenizer.model_max_length - len(bos_token_id + eos_token_id)
        step_size = _chunk_length - stride

        num_chunks = (concat_token_ids_length - _chunk_length) // step_size + 1

        # chunked total length
        _total_length = step_size * num_chunks + stride
        _remain_length = concat_token_ids_length - _total_length

        _remain_length_with_stride = _remain_length + stride

        result_chunks = {}
        for k, t in concat_token_ids.items(): # the t is a list of token ids, very long
            chunked_ids = []
            for i in range(num_chunks):
                _start_pos = i * step_size
                _end_pos = _start_pos + _chunk_length 
                if k == 'input_ids':
                    chunk_sequence = bos_token_id + t[_start_pos:_end_pos] + eos_token_id
                elif k == 'attention_mask':
                    chunk_sequence = bos_attn_mask + t[_start_pos:_end_pos] + eos_attn_mask
                else:
                    raise KeyError("Only support model_input_names = ['input_ids', 'attention_mask'].")
                chunked_ids.append(chunk_sequence)

            # process the last chunk
            if _remain_length_with_stride >= _chunk_length * min_len_frac:
                if tokenizer.padding_side == 'right':
                    if k == 'input_ids':
                        _last_chunk = (bos_token_id + t[-_remain_length_with_stride:] + eos_token_id 
                                        + [tokenizer.pad_token_id] * (_chunk_length - _remain_length_with_stride))
                    elif k == 'attention_mask':
                        _last_chunk = (bos_attn_mask + t[-_remain_length_with_stride:] + eos_attn_mask 
                                        + [0] * (_chunk_length - _remain_length_with_stride))
                    else: 
                        raise KeyError("Only support model_input_names = ['input_ids', 'attention_mask'].")
                else:
                    if k == 'input_ids':
                        _last_chunk = ([tokenizer.pad_token_id] * (_chunk_length - _remain_length_with_stride) 
                                        + bos_token_id + t[-_remain_length_with_stride:] + eos_token_id)
                    elif k == 'attention_mask':
                        _last_chunk = ([0] * (_chunk_length - _remain_length_with_stride) 
                                        + bos_attn_mask + t[-_remain_length_with_stride:] + eos_attn_mask)
                    else: 
                        raise KeyError("Only support model_input_names = ['input_ids', 'attention_mask'].")
                chunked_ids.append(_last_chunk)

            result_chunks[k] = chunked_ids

        del tokenized_batch
        del concat_token_ids

        return result_chunks
    
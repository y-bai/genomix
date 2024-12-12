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

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import TrainingArguments
import json

from ..tokenizers import (
    get_tokenizer_cls, 
    PRETRAINED_MODEL_NAME_CLS_MAP,
    BioSeqTokenizerMap,
)
from ..utils.constants import SPECIAL_TOKENS, INITIAL_ALPHABETS

from datasets import IterableDataset, Dataset, IterableDatasetDict, DatasetDict


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
        config_dict = copy.deepcopy(config).to_dict()
        tokenizer_type = config_dict.pop("tokenizer_type", None)
        tokenizer_pretrained_model_path = config_dict.pop("tokenizer_pretrained_model_path", None)
        chars = config_dict.pop("chars", None)

        assert tokenizer_type is not None, "tokenizer_type must be provided"

        if tokenizer_type == "CHAR_TOKEN":
            assert chars is not None, "chars must be provided for char tokenization"

            _tokenizer = get_tokenizer_cls(tokenizer_type)(
                characters = chars, 
                **config_dict
            )
        else:
            assert tokenizer_pretrained_model_path is not None, "tokenizer_pretrained_model_path must be provided"

            _tokenizer = get_tokenizer_cls(tokenizer_type).from_pretrained(
                pretrained_model_name_or_path = tokenizer_pretrained_model_path,
                local_files_only=True,
                **config_dict
            )
        # we treat the pad_token as the eos_token
        _tokenizer.pad_token = _tokenizer.eos_token

        self.tokenizer = _tokenizer
    
    def tokenize_by_dataset_map(
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
        overwrite_cache: Optional[bool] = False,
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
        ).get_chunked_tokenized_dataset(
            add_bos_token = self.tokenizer.add_bos_token,
            add_eos_token = self.tokenizer.add_eos_token
        )

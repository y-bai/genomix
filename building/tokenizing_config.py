#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		tokenizing_config.py
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

@Desc    :   	 tokenizing config

"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from transformers import TrainingArguments
import json

from ..genomix.tokenizers import BioSeqPreTrainedTokenizer, get_tokenizer_cls
from ..genomix.utils.constants import SPECIAL_TOKENS

MODEL_MAX_LEN = 1024        # ax length of the input sequence (char tokenization)
USE_STREAM = False
STRIDE = 16
NUM_PROCS = 16


class TokenizerConfig:

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
    

class BioSeqTokenizationConfig(TokenizerConfig):

    def __init__(
        self,
        token_model_type: BioSeqPreTrainedTokenizer = BioSeqPreTrainedTokenizer.CHAR,
        token_pretrained_model_path: Optional[str] = None,
        model_max_length: Optional[int] = 1024,
        padding_side: Optional[str] = "left",
        bos_token: Optional[str] = SPECIAL_TOKENS.BOS.value,
        eos_token: Optional[str] = SPECIAL_TOKENS.EOS.value,
        unk_token: Optional[str] = SPECIAL_TOKENS.UNK.value,
        add_bos_token: Optional[bool] = False,
        add_eos_token: Optional[bool] = False,
        padding: Optional[str] = "max_length",
        truncation: Optional[bool] = True,
        stride: Optional[int] = 16,
        return_overflowing_tokens: Optional[bool] = True,
        token_min_ctx_fraction: Optional[float] = 1,
        use_streaming: Optional[bool] = False,
        tokenization_map_num_proc: Optional[int] = 16,
    ):

        super().__init__()
        self.token_model_type = token_model_type
        self.token_pretrained_model_path = token_pretrained_model_path
        self.model_max_length = model_max_length
        self.padding_side = padding_side
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.padding = padding
        self.truncation = truncation
        self.stride = stride
        self.return_overflowing_tokens = return_overflowing_tokens
        self.token_min_ctx_fraction = token_min_ctx_fraction
        self.use_streaming = use_streaming
        self.tokenization_map_num_proc = tokenization_map_num_proc
    

class GenoMixTokenization:
    def __init__(self, config: BioSeqTokenizationConfig):
        self.config = config
        if config.token_model_type != BioSeqPreTrainedTokenizer.CHAR:
            self.tokenizer = get_tokenizer_cls(config.token_model_type).from_pretrained(config.token_pretrained_model_path)

#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		dataset_building.py
@Time    :   	2024/12/12 08:44:15
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

@Desc    :   	 build pytorch dataset for training/evaluating model

"""

import os
from typing import Any, Dict, List, Mapping, NewType, Optional, Sequence, Union
from dataclasses import dataclass
import logging
from scipy.sparse import issparse
import itertools

import torch
import torch.nn.functional as F
import torch.utils.data
import datasets
import copy

import time

logger = logging.getLogger(__name__)

InputDataClass = NewType("InputDataClass", Any)

class GenoMixDataParallelIterableDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset for handling large-scale data
    """
    def __init__(
        self,
        input_txt_fname: str,
        tokenizer: Any,                 # tokenizer class, e.g., CharacterTokenizer  
        max_seq_length: int = 1024,
        stride:int=16,
    ):
        super().__init__()
        self.input_txt_fname = input_txt_fname
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride
    
    def _process_sequence(self, sequence: str):
        pass
    
    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_total_num = worker_info.num_workers if worker_info is not None else 1



        # Map each element using the self._process
        mapped_itr = map(self._process, )

        # Add multiworker functionality
        mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)

        return mapped_itr



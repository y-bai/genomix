#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		dataset_builder.py
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
import logging
import itertools

import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data

import datasets
from datasets import load_dataset, Features

logger = logging.getLogger(__name__)

InputDataClass = NewType("InputDataClass", Any)

class GenoMixDataIterableDatasetV1(torch.utils.data.IterableDataset):
    """
    IterableDataset for handling large-scale data by read txt file line by line.

    NOTE: We only handle one input text file, if you have multiple files, you need to merge them into one file.

    """
    def __init__(
        self,
        input_ids_fname: str, 
        input_mask_fname: Optional[str]=None,
    ):
        """IterableDataset by reading txt file

        Parameters
        ----------
        input_txt_fname : str
            input txt file name containing tokenized samples, with ',' separated
        """
        super().__init__()
        self.input_ids_fname = input_ids_fname
        self.input_mask_fname = input_mask_fname
        # get the total number of lines in the input file
        n_line_str = self.input_ids_fname.split('-')[-1].strip()
        if n_line_str.isnumeric():
            self.total_examples = int(n_line_str)
            self.can_distributed_sample = True
        else:
            self.total_examples = -1
            self.can_distributed_sample = False
        
        # self.distributed_sampler = distributed_sampler
        # with open(input_txt_fname, 'r') as f:
        #     self.total_lines = sum(1 for _ in f)
    
    def _process(self, input_ids: str, input_mask: str=None) -> Dict[str, str]:
        if input_mask is not None:
            return dict(
                input_ids = input_ids.strip(),
                attention_mask = input_mask.strip()
            )
        else:
            return dict(
                input_ids = input_ids.strip()
            )
    
    # NOTE: The output order is not guaranteed to match the input order, depending on the order of the workers.
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_total_num = worker_info.num_workers if worker_info is not None else 1

        # TODO: There could be protential issue as the file handler is not closed
        if self.input_mask_fname is not None:
            ids_itr = open(self.input_ids_fname, 'r')
            mask_itr = open(self.input_mask_fname, 'r')
            # Map each element using the self._process
            mapped_itr = map(self._process, ids_itr, mask_itr)
        else:
            ids_itr = open(self.input_ids_fname, 'r')
            # Map each element using the self._process
            mask_itr = None
            mapped_itr = map(self._process, ids_itr)
        
        mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)
        
        # TODO: We will implement the distributed sampler later
        # if not self.distributed_sampler:
        #     # Add multiworker functionality
        #     mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)
        # else:
        #     world_size = torch.distributed.get_world_size() # the number of GPU
        #     process_rank = torch.distributed.get_rank()     # the rank of current GPUs
        #     mapped_itr = torch.utils.data.DistributedSampler(
        #         self, 
        #         num_replicas=(worker_total_num * world_size), 
        #         rank=(process_rank * worker_total_num + worker_id), 
        #         shuffle=False
        #     )
        return mapped_itr
    
    def __len__(self):
        # Caveat: When using DistributedSampler, we need to know the number of samples in our dataset!
        # Hence, we need to implement `__len__`.
        # https://github.com/Lightning-AI/pytorch-lightning/issues/15734
        return self.total_examples

class GenoMixDataIterableDatasetV2(torch.utils.data.IterableDataset):
    """
    GenoMixDataIterableDatasetV2 using load_dataset() from text file with streaming=True.

    NOTE: 
    if a single text file loaded (ie., dataset.num_shards=1), 
    there would have the following warning when DataLoader is created with num_workers > 1:
        datasets.iterable_dataset - Too many dataloader workers: 4 (max is dataset.num_shards=1). Stopping 3 dataloader workers.

    """
    def __init__(
        self,
        input_ids_fname: Union[Union[str, List[str]], Dict[str, str]],
        input_mask_fname: Optional[Union[Union[str, List[str]], Dict[str, str]]]=None,
    ):
        super().__init__()
        self.input_ids_ds = load_dataset(
            'text',  
            trust_remote_code=True, 
            data_files = input_ids_fname,
            cache_dir = None,
            streaming=True,
            features=Features({'input_ids': datasets.Value("string")})
        )
        # we only allow one key in the datasetdict, eg, 'train' or 'test'
        self.dt_dict_key = list(self.input_ids_ds.keys())[0]
        if input_mask_fname is not None:
            self.input_mask_ds = load_dataset(
                'text',  
                trust_remote_code=True, 
                data_files = input_mask_fname,
                cache_dir = None,
                streaming=True,
                features=Features({'attention_mask': datasets.Value("string")})
            )
        else: 
            self.input_mask_ds = None

    def __iter__(self):
        if self.input_mask_ds is not None:
            for input_ids, attention_mask in zip(
                self.input_ids_ds[self.dt_dict_key], self.input_mask_ds[self.dt_dict_key]
                ):
                yield {
                    'input_ids': input_ids["input_ids"],
                    'attention_mask': attention_mask["attention_mask"]
                }
        else:
            for input_ids in self.input_ids_ds[self.dt_dict_key]:
                yield {
                    'input_ids': input_ids['input_ids']}


class GenoMixDataDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids_fname: Union[Union[str, List[str]], Dict[str, str]],
        input_mask_fname: Optional[Union[Union[str, List[str]], Dict[str, str]]]=None,
        num_proc: int = 24,
    ):
        super().__init__()

        self.input_ids_ds = load_dataset(
            'text',  
            trust_remote_code=True, 
            data_files = input_ids_fname,
            cache_dir = None,
            num_proc=num_proc,
            features=Features({'input_ids': datasets.Value("string")})
        )
        # we only allow one key in the datasetdict, eg, 'train' or 'test'
        self.dt_dict_key = list(self.input_ids_ds.keys())[0]
        if input_mask_fname is not None:
            self.input_mask_ds = load_dataset(
                'text',  
                trust_remote_code=True, 
                data_files = input_mask_fname,
                cache_dir = None,
                num_proc=num_proc,
                features=Features({'attention_mask': datasets.Value("string")})
            )
        else: 
            self.input_mask_ds = None

    def __getitem__(self, idx):
        if self.input_mask_ds is not None:
            return {
                'input_ids': self.input_ids_ds[self.dt_dict_key]['input_ids'][idx],
                'attention_mask': self.input_mask_ds[self.dt_dict_key]['attention_mask'][idx]
            }
        else:
            return {
                'input_ids': self.input_ids_ds[self.dt_dict_key]['input_ids'][idx]
            }    
    
    def __len__(self):
        return len(self.input_ids_ds)


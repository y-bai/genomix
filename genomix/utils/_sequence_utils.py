#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_sequence_utils.py
@Time    :   	2024/11/19 16:20:09
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

import numpy as np
from typing import List

from datasets import Dataset

def chunk_sequence(sequence: str, chunk_size: int = 20000, overlap_step: int = 200) -> List[str]:
    """Chunk a long sequence into small pieces with overlap

    Parameters
    ----------
    sequence : str
        long sequence with characters
    chunk_size : int, optional
        size of each chunk, by default 20000
    overlap_step : int, optional
        overlap size between two adjacent chunks, by default 200

    Returns
    -------
    List[str]
        list of chunks
    """
    chunks = []
    for i in range(0, len(sequence), chunk_size - overlap_step): # the last chunk may be smaller than chunk_size
        chunks.append(sequence[i : i + chunk_size])
    return chunks


def _batch_iterator_dataset(
        ds: Dataset, 
        ds_feature_name: str='sequence',
        batch_size: int=1000, 
        max_num_examples:int=0
    ):
    """Batch iterator for dataset

    Parameters
    ----------
    ds : Dataset
        dataset object of Huggingface datasets
    ds_feature_name : str, optional
        The feature name in the dataset that would be used, by default 'sequence'
    batch_size : int, optional
        The size of batch in each iteration, by default 1000
    max_num_examples : int, optional
        The max number of examples to be used, by default 0
        - 0: all examples in the dataset will be used,
            otherwise, a random subset of examples will be used.

    Yields
    ------
    list
        batch of sequences in the dataset
    """
    n_total_examples = len(ds)
    n_examples = n_total_examples if max_num_examples == 0 else max_num_examples
    rand_idx = (np.arange(n_total_examples) 
                if max_num_examples == 0 
                else np.random.permutation(np.arange(n_total_examples))[:n_examples])
    for i in range(0, n_examples, batch_size):
        yield ds[rand_idx[i : i + batch_size]][ds_feature_name]


def downsampling_dataset(
        ds: Dataset, 
        max_num_examples: int=400000
    ) -> Dataset:
    """Downsampling dataset

    Parameters
    ----------
    ds : Dataset
        dataset object of Huggingface datasets
    max_num_examples : int, optional
        The max number of examples to be used, by default 400000
        - 0: all examples in the dataset will be used,
            otherwise, a random subset of examples will be used.

    Returns
    -------
    Dataset
        downsampled dataset
    """
    n_total_examples = len(ds)
    n_examples = n_total_examples if max_num_examples == 0 else max_num_examples
    rand_idx = (np.arange(n_total_examples) 
                if max_num_examples == 0 
                else np.random.permutation(np.arange(n_total_examples))[:n_examples])
    return ds.select(rand_idx)
    

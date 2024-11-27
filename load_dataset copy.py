#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_iterable_dataset.py
@Time    :   	2024/11/21 16:41:03
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

@Desc    :   	process Huggingface dataset

"""

import logging
from pathlib import Path
from collections import OrderedDict
from functools import partial
from typing import Optional
import shutil

from datasets import Dataset, DatasetDict, load_from_disk, Features, Value

from ..utils import (
    partition_list,
    check_dir_exists,
    check_file_exists,
    write_txt_file,
    down_sampling,
    chunk_sequence,
)

from ..utils._cache import (
    _find_from_cache, 
    _write_to_cache,
    _resolve_cache_file
)

from ..utils.constants import (
    SEQENCE_FEATURE_NAME_IN_DATASET_CHUNKED
)

logger = logging.getLogger(__name__)

def write_dataset_to_txt(
    input_ds_fname: str,
    output_txt_fname: str,
    *,
    input_ds_key: str='train',
    input_ds_feature: str='sequence',
    max_num_examples: int=400000,
    chunk_size: int=20000,
    overlap_step: int=0,
    batch_size: int=5000,
    num_proc: int=16,
    disable_tqdm: bool=False,
    to_cache: bool = True,
):
    """Generate corpus txt file from the input hg dataset.

    When HG dataset is too large and it cosumes too much memory when `load_from_disk`,
    we should first convert the dataset into corpus txt file, and then load the corpus txt file.

    Parameters
    ----------

    input_ds_fname : str,
        HuggingFace dataset names/paths, such as '/path/to/dataset_name'
        The datasets were generated by the script dataset/dataset_gen.py
        - Could be a list of dataset names/paths
    
    output_txt_fname : str,
        output corpus txt file name
    
    input_ds_key : str,
        key of the dataset to be used, by default 'train'
        - Could be 'train', 'validation', 'test'
        - This is for the case that the input dataset contains multiple splits, 
            that is, the input dataset is a DatasetDict object.
        - It is not used if the input dataset is a Dataset object.
    
    input_ds_feature : str,
        feature name of the dataset to be used, by default 'sequence'
        - This is for the case that the input dataset contains multiple features,
            and the feature contains the sequence data.

    max_num_examples : int, optional
        number of downsampled examples, by default 400000
        - 0: no downsampling
        - This is used for downsampling the training corpus when the corpus is too large.
        - NOTE: when `input_ds_names` is list, the down sampling is applied to each dataset.
          As a result, the real number of examples in the output corpus could be 
        `max_num_examples * len(input_ds_names)`.

    chunk_size : int, optional
        size of each chunk, by default 20000
        - 0: no chunking

    overlap_step : int, optional
        overlap size between two adjacent chunks, by default 200
        - when chunk_size is -1, then overlap_step is not used.

    batch_size : int, optional
        batch size for `dataset.map`, by default 5000

    num_proc : int, optional
        number of processors to be used for parallel processing, by default 16
    
    disable_tqdm : bool, optional
        disable tqdm, by default False
    
    to_cache: bool, optional
        whether to cache the output file, by default True

    """

    check_dir_exists(input_ds_fname)

    if to_cache:
        meta_info = OrderedDict(
            input_ds_fname=Path(input_ds_fname).name,
            input_ds_key=input_ds_key,
            input_ds_feature=input_ds_feature,
            max_num_examples=max_num_examples,
            chunk_size=chunk_size,
            overlap_step=overlap_step,
            num_proc=num_proc,
            disable_tqdm=disable_tqdm,
            to_cache=to_cache
        )
        
        cache_fname, cache_fsize = _find_from_cache(
            file_meta = meta_info,
        )

        if cache_fname is not None and cache_fsize > 0:
            _resolve_cache_file(cache_fname, output_txt_fname)
            logger.info(f"corpus txt file saved at: {output_txt_fname}")
            return
    
    _ds, _seq_feat_name = load_dataset_to_dataset(
        input_ds_fname,
        input_ds_key=input_ds_key, 
        input_ds_feature=input_ds_feature,
        max_num_examples=max_num_examples,
        chunk_size=chunk_size,
        overlap_step=overlap_step,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    logger.info(f"writing corpus into file...")
    if check_file_exists(output_txt_fname, _raise_error=False):
        logger.warning(f"File {output_txt_fname} exists, will be overwritten.")
        shutil.rmtree(output_txt_fname)

    write_txt_file(
        output_txt_fname, 
        _ds[_seq_feat_name],
        n_proc=num_proc, 
        disable_tqdm=disable_tqdm,
    )
    if to_cache:
        logger.info(f"cache the file...")
        _write_to_cache(output_txt_fname, file_meta=meta_info)

    logger.info(f"corpus txt file saved at: {output_txt_fname}")


def load_dataset_to_dataset(
    input_ds_name: str, 
    *,
    input_ds_key: str = 'train', 
    input_ds_feature: str = 'sequence',
    max_num_examples: int = 0,  # for downsampling
    chunk_size: int = 20000,    # for trcunking the sequence
    overlap_step: int = 0,
    batch_size: int = 1000,
    num_proc: Optional[int] = 16,
)-> Dataset:
    """ the input HuggingFace datasets, downsampling and chunking the sequence
 

    Parameters
    ----------
    input_ds_name : str
        input dataset name, such as '/path/to/dataset_directory'
    input_ds_key : str, 
        when the input dataset is DatasetDict object, 
        then specify the key to be used for training tokenizer, by default 'train'.
        - This is for the case that the input dataset contains multiple splits,
        - e.g., it could be 'train', 'validation', 'test'
    input_ds_feature : str, optional
        the feature name of the input dataset that saves the sequence, by default 'sequence'

    max_num_examples : int, optional
        the max number of examples to be used, by default 0
        - 0: all examples in the dataset will be used,
        - otherwise, a random subset of examples will be used. 
          max_num_examples could be , for example, 400000.

    chunk_size : int, optional
        the size of each chunk for truncking long input sequence, by default 20000
        - 0: no chunking

    overlap_step : int, optional
        the overlap size between two adjacent chunks, by default 0
        - only used when chunk_size > 0

    batch_size : int, optional
        the batch size for `dataset.map`, by default 1000
        - only used when chunk_size > 0

    num_proc : Optional[int], optional
        the number of processes to be used for chunking 
        the sequence with `dataset.map function`, by default 16
        - only used when chunk_size > 0

    Returns
    -------
    ds : Dataset
        processed dataset with downsampled, chunked sequence, depending on the input parameters
    """

    assert isinstance(input_ds_name, str), "paramter input_ds_name should be a HG dataset path"
    check_dir_exists(input_ds_name)

    # 0. load the input datasets
    # NOTE: 
    # When reading dataset, a cache will be generated to the ~/. cache/huggingface/datasets directory
    # When using .map and .filter operations, runtime cache will be generated to the /tmp/hf_datasets-* directory
    ds = load_from_disk(input_ds_name)

    if isinstance(ds, DatasetDict):
        ds = ds[input_ds_key]
    
    seq_feat_name = input_ds_feature
    
    # 1. down sampling
    if max_num_examples > 0:
        logger.info(f"down sampling the dataset...")
        ds = down_sampling(ds, max_num_examples, n_proc=num_proc)

    # 2. chunking the sequence
    if chunk_size > 0:
        chunk_func = partial(
            chunk_sequence, 
            chunk_size=chunk_size, 
            overlap_step=overlap_step, 
            n_proc=num_proc
        )
        
        # NOTE: `num_proc` in map is the number of processes to be used for parallel processing.
        # When `batch_size` > the total number of examples, then the averge number of examples over `num_proc`
        # will be used for each process. 
        ds = ds.map(
            lambda examples: {
                SEQENCE_FEATURE_NAME_IN_DATASET_CHUNKED: chunk_func(
                    examples[input_ds_feature]
                )
            }, 
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc, 
            remove_columns=ds.column_names,
        )
        seq_feat_name = SEQENCE_FEATURE_NAME_IN_DATASET_CHUNKED
    # NOTE: removing columns rather than the sequence feature will be 5x faster in the downstreaming processing.
    if seq_feat_name is not None and seq_feat_name != SEQENCE_FEATURE_NAME_IN_DATASET_CHUNKED:
        cols = ds.column_names
        original_feature, will_remove_feature = partition_list(lambda x: x == input_ds_feature, cols)
        ds = ds.remove_columns(will_remove_feature)
    return ds, seq_feat_name


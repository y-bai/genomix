#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		00_write_txt_test.py
@Time    :   	2024/11/25 10:14:31
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

@Desc    :   	test functions

"""


import os
import sys
import logging
from pathlib import Path
import json

import time


from datasets import load_from_disk

sys.path.append(str(Path(__file__).parents[1]))

from genomix.tools.load_dataset import (
    load_dataset_to_dataset,
    write_dataset_to_txt
)

from genomix.tools import (
    TrainBPETokenizerConfig,
    TrainUnigramTokenizerConfig,
    TrainSPMTokenizerConfig
)

home_dir = os.path.expanduser('~')
raw_data_dir = os.path.join(home_dir, "projects/biomlm/biomlm/datasets")

ds_chm13 = 'raw_dataset_chm13_t2t'
ds_crgd = 'raw_dataset_crgd_t2t'
ds_multi = 'raw_dataset_multi'

logger = logging.getLogger(__name__)

def cal_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def test_load_dataset():
    
    params = dict(
        input_ds_fname = os.path.join(raw_data_dir, ds_chm13),

        input_ds_key = "train",
        input_ds_feature = "sequence",
        max_num_examples = 6,  # for downsampling
        chunk_size = 2000,
        overlap_step = 0,
        map_batch_size =2,
        num_proc = 2, # reduce the number of processes reduce the memory usage
    )

    stat_time = time.time()
    
    ds, feat_name = load_dataset_to_dataset(
        params['input_ds_fname'], 
        input_ds_key = params['input_ds_key'], 
        input_ds_feature = params['input_ds_feature'],
        max_num_examples=params['max_num_examples'],
        chunk_size=params['chunk_size'],
        overlap_step=params['overlap_step'],
        map_batch_size=params['map_batch_size'],
        num_proc=params['num_proc']
    )
    end_time = time.time()
    # --------------
    # This would take a lot of memory when read the whole dataset
    # ds_example = ds[feat_name][0]
    # # --------------
    logger.info(f'After process: \n{ds}')
    # logger.info(f'Example: \n{ds_example}\nlength: {len(ds_example)}')
    logger.info(f'Process params: \n{json.dumps(params, indent=4)}')
    logger.info(f'Memory usage: {cal_memory_usage():.2f} MB')
    logger.info(f"Time cost: {end_time - stat_time:.2f} s")

def test_write_dataset_to_text():
    params = dict(
        input_ds_fname = os.path.join(raw_data_dir, ds_chm13),
        output_txt_fname = '/home/share/huadjyin/home/baiyong01/projects/genomix/tmp/testdata.txt',

        input_ds_key = "train",
        input_ds_feature = "sequence",
        max_num_examples = 0,  # for downsampling
        chunk_size = 20000,
        overlap_step = 0,
        map_batch_size =1000,
        num_proc = 4, # reduce the number of processes to reduce the memory usage
        disable_tqdm = False,
    )

    stat_time = time.time()
    
    write_dataset_to_txt(
        params['input_ds_fname'], 
        params['output_txt_fname'], 

        input_ds_key = params['input_ds_key'],
        input_ds_feature = params['input_ds_feature'],
        max_num_examples=params['max_num_examples'],
        chunk_size=params['chunk_size'],
        overlap_step=params['overlap_step'],
        map_batch_size=params['map_batch_size'],
        num_proc=params['num_proc'],
        disable_tqdm = params['disable_tqdm'],
    )
    end_time = time.time()
    # ----------------
    logger.info(f'Process params: \n{json.dumps(params, indent=4)}')
    logger.info(f'Memory usage: {cal_memory_usage():.2f} MB')
    logger.info(f"Time cost: {end_time - stat_time:.2f} s")



if __name__ == '__main__':

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    # test_load_dataset()
    test_write_dataset_to_text()



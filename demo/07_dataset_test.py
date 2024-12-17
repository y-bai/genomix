#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		07_dataset_test.py
@Time    :   	2024/12/16 14:22:09
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

@Desc    :   	 None

"""

import sys
import os
import logging
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datasets
from datasets import load_dataset, IterableDataset, Dataset, Features
from torch.utils.data import DataLoader

sys.path.append('..')

from genomix.data_builder.datasets import GenoMixDataIterableDatasetV1, GenoMixDataIterableDatasetV2

logger = logging.getLogger(__name__)

def cal_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    stat_time = time.time()
    iter_ds = GenoMixDataIterableDatasetV1(
        '/home/share/huadjyin/home/baiyong01/projects/genomix/tmp/testdata/chm13t2t-train-input_ids.txt',
    )  
    dtloader = DataLoader(iter_ds, batch_size=4, num_workers=4)
    for i, x in enumerate(dtloader):
        print(x['input_ids'])
        # for y in x['input_ids']:
        #     print(y[:10])
        if i > 2:
            break
    end_time = time.time()

    logger.info(f'Memory usage: {cal_memory_usage():.2f} MB')
    logger.info(f"Time cost: {end_time - stat_time:.2f} s")
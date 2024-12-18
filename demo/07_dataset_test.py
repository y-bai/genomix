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
import glob
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datasets
from datasets import load_dataset, IterableDataset, Dataset, Features
from torch.utils.data import DataLoader

sys.path.append('..')

from genomix.data_builder.datasets import GenoMixDataIterableDatasetV1, GenoMixDataIterableDatasetV2
from genomix.data_builder.data_collator import GenoMixDataCollatorForLanguageModeling
from genomix.utils.constants import GENOMIX_DATA_DIR
from genomix.utils.common import cal_memory_usage

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    SUB_DIR = "chm13-t2t"
    OUTPUT_FILE_BASE_NAME = "GCF_009914755.1_T2T-CHM13v2.0_genomic"

    input_ids_files = glob.glob(os.path.join(
        GENOMIX_DATA_DIR, 
        f"{SUB_DIR}/{OUTPUT_FILE_BASE_NAME}_TRAIN-input_ids*.txt"
    ))
    stat_time = time.time()
    iter_ds = GenoMixDataIterableDatasetV1(
        input_ids_files[0],
    ) 
    # dtloader = DataLoader(iter_ds, batch_size=4, num_workers=4, collate_fn=GenoMixDataCollatorForLanguageModeling())
    # for i, x in enumerate(dtloader):
    #     print(x)
    #     # for y in x['input_ids']:
    #     #     print(y[:10])
    #     if i > 1:
    #         break

    for i, i_data in enumerate(iter_ds):
        print(i_data)
        if i > 2:
            break
        

    end_time = time.time()

    logger.info(f'Memory usage: {cal_memory_usage():.2f} MB')
    logger.info(f"Time cost: {end_time - stat_time:.2f} s")
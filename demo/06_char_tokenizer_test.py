#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		06_char_tokenizer_test.py
@Time    :   	2024/12/11 11:36:42
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

@Desc    :   	 char tokenizer test

"""

import sys
import os
import logging
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('..')

from genomix.data_builder.tokenization import GenoMixTokenizationConfig, GenoMixTokenization
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

    stat_time = time.time()

    # define the tokenization with configuration
    tokenization = GenoMixTokenization(GenoMixTokenizationConfig())

    tokenization.tokenize_with_text_file(
        os.path.join(GENOMIX_DATA_DIR, f"{SUB_DIR}/{OUTPUT_FILE_BASE_NAME}_TRAIN.txt"), # '/home/share/huadjyin/home/baiyong01/projects/genomix/tmp/testdata/GCF_009914755.1_T2T-CHM13v2.0_genomic_TEST.txt',
        stride = 16, # overlap between chunks
        token_min_ctx_fraction=1.0,
        batch_size = None,
        num_proc = 16,
        disable_tqdm = False,
        save_path = os.path.join(GENOMIX_DATA_DIR, SUB_DIR),
        save_file_prefix=f"{OUTPUT_FILE_BASE_NAME}_TRAIN",
        save_attn_mask = True,
    )
    end_time = time.time()
    logger.info(f'Memory usage: {cal_memory_usage():.2f} MB')
    logger.info(f"Time cost: {end_time - stat_time:.2f} s")

    
    





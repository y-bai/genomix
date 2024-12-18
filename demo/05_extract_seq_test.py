#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		05_extract_seq_test.py
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

@Desc    :   	 GenoMix mamba2 test

"""

import sys
import os
import logging
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('..')
from dstool.extract_seqence import extract_seq_from_fasta
from genomix.utils.constants import GENOMIX_DATA_DIR
from genomix.utils.common import cal_memory_usage, check_dir_exists

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    input_fa = os.path.join(
        os.path.expanduser('~'), 
        "projects/biomlm/data/T2T/ncbi_dataset/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
    )# r"/home/share/huadjyin/home/baiyong01/projects/biomlm/data/T2T/ncbi_dataset/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
    SUB_DIR = "chm13-t2t"
    OUTPUT_FILE_BASE_NAME = "GCF_009914755.1_T2T-CHM13v2.0_genomic"
    
    output_path = os.path.join(GENOMIX_DATA_DIR,  SUB_DIR) #r"/home/share/huadjyin/home/baiyong01/projects/genomix/tmp/testdata"
    check_dir_exists(output_path, _raise_error=False)

    stat_time = time.time()
    extract_seq_from_fasta(
        input_fa, 
        output_path, 
        output_file_base_name=OUTPUT_FILE_BASE_NAME,
        num_proc = 16,
        re_partten_str = r"NC_\d*.\d* Homo sapiens isolate \w*\d* chromosome (\d*|\w), alternate assembly T2T-CHM13v2.0",
        test_chr=['22']
    )
    end_time = time.time()
    logger.info(f'Memory usage: {cal_memory_usage():.2f} MB')
    logger.info(f"Time cost: {end_time - stat_time:.2f} s")
    
    





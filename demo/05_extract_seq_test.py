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

    input_fa = r"/home/share/huadjyin/home/baiyong01/projects/biomlm/data/T2T/ncbi_dataset/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
    output_path = r"/home/share/huadjyin/home/baiyong01/projects/genomix/tmp/testdata"

    stat_time = time.time()
    extract_seq_from_fasta(
        input_fa, 
        output_path, 
        num_proc = 16,
        re_partten_str = r"NC_\d*.\d* Homo sapiens isolate \w*\d* chromosome (\d*|\w), alternate assembly T2T-CHM13v2.0",
        test_chr=['22']
    )
    end_time = time.time()
    logger.info(f'Memory usage: {cal_memory_usage():.2f} MB')
    logger.info(f"Time cost: {end_time - stat_time:.2f} s")
    
    





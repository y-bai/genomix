#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		data_config.py
@Time    :   	2024/12/18 11:13:09
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
import os
import sys
import glob
from dataclasses import dataclass, field

sys.path.append('..')
from genomix.utils.constants import GENOMIX_DATA_DIR

@dataclass
class GenoMixCausalLMDataConfig:
    # this is the data name, such as "chm13_t2t", "multi_species", "1kg", "crgd_t2t" etc.
    data_sub_dir = "chm13-t2t"  
    # this is the raw input file name.
    # If the file name is given, the extracted raw sequence txt file is named as 
    # "{OUTPUT_FILE_BASE_NAME}_TRAIN.txt" or "{OUTPUT_FILE_BASE_NAME}_TEST.txt", 
    # And the tokenized file is named as "{OUTPUT_FILE_BASE_NAME}_TRAIN-input_ids.txt" and "{OUTPUT_FILE_BASE_NAME}_TEST-input_ids.txt", 
    # And the tokenized attention mask file is named as "{OUTPUT_FILE_BASE_NAME}_TRAIN-attention mask.txt" and "{OUTPUT_FILE_BASE_NAME}_TEST-attention_mask.txt", 
    file_base_name = "GCF_009914755.1_T2T-CHM13v2.0_genomic"

    input_train_tokenized_input_ids_file = glob.glob(os.path.join(
        GENOMIX_DATA_DIR, 
        f"{data_sub_dir}/{file_base_name}_TRAIN-input_ids*.txt"
    ))[0]

    input_test_tokenized_input_ids_file = glob.glob(os.path.join(
        GENOMIX_DATA_DIR, 
        f"{data_sub_dir}/{file_base_name}_TEST-input_ids*.txt"
    ))[0]
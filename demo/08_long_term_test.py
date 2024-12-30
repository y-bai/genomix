#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		08_long_term_test.py
@Time    :   	2024/12/28 09:38:11
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

TRITON_CACHE_DIR = "/tmp/.triton/cache"
os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE_DIR


import torch

sys.path.append('..')
from genomix.utils.common import cal_memory_usage
from genomix.models.genomix_sampling import GenomixLongTermInitEnocder, DownSamplingProjection
from genomix.models.genomix_config import GenoMixMamba2Config

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    if not os.path.exists(TRITON_CACHE_DIR):
        os.makedirs(TRITON_CACHE_DIR, mode=0o777, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    b = 1               # batch size  
    l = 1024*1024        # sequence length
    d = 256             # hidden size (d_model)

    toy_data = torch.randn(b, l, d).to("cuda")

    down_sample = DownSamplingProjection(
        d,                                      # input hidden size of the model, e.g., d_model of the model
        sampling_rate=64*8,                 # kernel_size of the conv1d
        groups=d,                               # groups of the conv1d, 1 or hidden_size
    ).to("cuda")
    out = down_sample(toy_data)
    free, total = torch.cuda.mem_get_info(toy_data.device)
    mem_used_mb = (total - free) / 1024 ** 2
    logger.info(f"Memory usage: {mem_used_mb:.2f} MB")

    logger.info(f"toy_data: {toy_data.shape}")
    logger.info(f"down sample: {out.shape}")

    # m = torch.nn.Conv1d(1, 1, 3, stride=3,padding=(3-1)).to("cuda")
    # input = torch.randn(1, 10, 1).to("cuda")
    # input = input.transpose(1, 2)
    # # output: torch.Size([20, 33, 24])
    # output = m(input)
    # output = output.transpose(1, 2)

    # logger.info(f"output: {output.shape}")

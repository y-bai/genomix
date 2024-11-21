#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_common_utils.py
@Time    :   	2024/03/05 13:24:46
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

@Desc    :   	loading local model and configuration files

"""

import torch

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from ..utils import read_json

def model_size(model):
    return sum(t.numel() for t in model.parameters())

# adapted from from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
def load_config_hf(
        model_name_or_dict, 
        file_name=None, 
        local_files_only=True
    ):
    resolved_archive_file = cached_file(
        model_name_or_dict, 
        filename=file_name if file_name is not None else CONFIG_NAME, 
        local_files_only=local_files_only, 
        _raise_exceptions_for_missing_entries=False)

    return read_json(resolved_archive_file)


def load_state_dict_hf(
        model_name_or_dict, 
        file_name=None, 
        local_files_only=True, 
        device=None, dtype=None
    ):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(
        model_name_or_dict, 
        filename=file_name if file_name is not None else WEIGHTS_NAME, 
        local_files_only=local_files_only, 
        _raise_exceptions_for_missing_entries=False)
    state_dict = torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict






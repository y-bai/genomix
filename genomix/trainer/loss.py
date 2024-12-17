#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		loss.py
@Time    :   	2024/12/17 14:27:27
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

@Desc    :   	 loss function

"""

import torch.nn as nn

GENOMIX_LOSS = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
}

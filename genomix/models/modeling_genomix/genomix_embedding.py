#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		genomix_embedding.py
@Time    :   	2024/11/29 14:14:46
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

@Desc    :   

"""

import torch.nn as nn


class GenoMixEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        emb_method: str='normal', 
        fusion_strategy: str='add',
        **kwargs
    ):
        super().__init__()
        



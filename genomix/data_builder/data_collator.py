#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		data_collator.py
@Time    :   	2024/12/17 12:59:20
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

@Desc    :   	 data collator for training/evaluating model

"""

from dataclasses import dataclass
from typing import Any, List, Mapping, NewType
import torch

InputDataClass = NewType("InputDataClass", Any)


@dataclass
class GenoMixDataCollator:
    """
    Data collator for GenoMix model.

    Take a list of samples from a Dataset and collate them into a batch.

    """
    def __call__(self, examples: List[InputDataClass]):
        """
        Collate the list of samples into a batch.

        """
        # Handle example is None
        examples = [example for example in examples if example is not None]

        if not isinstance(examples[0], Mapping):
            examples = [vars(example) for example in examples]

        # debug: print out the result
        # print(len(examples))
        # for example in examples:
        #     print(f"length: {len(example['input_ids'])}, {example['input_ids'][:10]}")

        first = examples[0] 
        batch = {}

        for k, v in first.items():
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([example[k] for example in examples])
            elif isinstance(v, list):
                batch[k] = torch.tensor([example[k] for example in examples])
            elif isinstance(v, str):
                batch[k] = torch.tensor([
                    list((map(int, example[k].split(',')))) for example in examples
                ], dtype=torch.long)
            else:
                raise ValueError(f"Unsupported type: {type(v)}")
        return batch
                
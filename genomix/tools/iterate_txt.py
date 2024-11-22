#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		iterate_txt.py
@Time    :   	2024/11/22 15:12:42
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

@Desc    :   	None

adapted from
https://github.com/huggingface/transformers/issues/12966

"""



from torch.utils.data import IterableDataset


class BatchProcessedDataset(IterableDataset):
    def __init__(self, files, tokenizer, batch_size=4096, limit=-1):
        self.files = files
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.limit = limit

    def __iter__(self):
        num_iter = 0
        for file_path in self.files:
            with open(file_path) as f:

                next_batch = [x.strip("\n") for _, x in zip(range(self.batch_size), f)]

                while next_batch:
                    tokenized_batch = self.tokenizer(next_batch, padding='max_length', truncation=True, return_special_tokens_mask=True)
                    for encoding in tokenized_batch.encodings:
                        if num_iter == self.limit:
                            return
                        yield {
                            "input_ids": encoding.ids,
                            "token_type_ids": encoding.type_ids,
                            "attention_mask": encoding.attention_mask,
                            "special_tokens_mask": encoding.special_tokens_mask
                        }
                        num_iter += 1
                    next_batch = [x.strip("\n") for _, x in zip(range(self.batch_size), f)]
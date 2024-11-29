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

import torch
import torch.nn as nn

class FloatValueEmbedding(nn.Module):

    def __init__(
        self,
        d_model: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.value_embedding = nn.Linear(1, d_model)

    def forward(self, x):
        return self.value_embedding(x.unsqueeze(-1))
    

class GenoMixGeneEmbedding(nn.Module):
    """GenoMix embedding for gene_id and gene_val

    Parameters
        ----------
        vocab_size : int
            the size of gene_id vocabulary
        d_model : int
            the size of gene embedding, same as d_model in model
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gene_id_embedding = nn.Embedding(vocab_size, d_model)
        self.gene_val_embedding = FloatValueEmbedding(d_model)

        self.fusion = nn.Linear(2*d_model, d_model)

    def forward(
        self, 
        gene_id, 
        gene_val,
        fusion_type='add'
    ):
        """embedding gene_id and gene_val, and fusion them together

        Parameters
        ----------
        gene_id : torch.Tensor
            gene input_ids
        gene_val : torch.Tensor
            gene values
        fusion_type : str, optional
            fusion type, could be 'add' | 'concat', by default 'concat'

        Returns
        -------
        torch.Tensor
            fused gene embedding

        Raises
        ------
        ValueError
            Unknown fusion type
        """
        if fusion_type == 'concat':
            return self.fusion(torch.cat(
                [
                    self.gene_id_embedding(gene_id),
                    self.gene_val_embedding(gene_val)
                ],
                dim=-1
            ))
        elif fusion_type == 'add':
            return (
                self.gene_id_embedding(gene_id) + 
                self.gene_val_embedding(gene_val)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    


        



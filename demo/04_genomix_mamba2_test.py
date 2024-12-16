#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		04_genomix_mamba2_test.py
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
import logging
import numpy as np
from transformers import Mamba2Config
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('..')

from genomix.models.genomix_vallim2 import GenoMixMamba2Model, GenoMixMamba2ForCausalLM
from genomix.models.genomix_config import GenoMixMamba2Config
from dstool.tokenization import GenoMixTokenizationConfig, GenoMixTokenization
from genomix.tokenizers.char_tokenizer import CharacterTokenizer

logger = logging.getLogger(__name__)

def test_genomix_mamba2_model():
    
    config = GenoMixMamba2Config()
    d_model = config.d_model
    ssm_headdim = config.ssm_cfg['headdim']
    assert d_model / ssm_headdim % 8 == 0, f'd_model / ssm_headdim % 8 must be 0'

    attn_num_heads = config.attn_cfg['num_heads']
    config.attn_cfg['head_dim'] = d_model // attn_num_heads
    config.attn_cfg['rotary_emb_dim'] = d_model // attn_num_heads

    # add attn layers 
    # each with 8 mamba2 layers interval
    n_layers = config.n_layers
    interval = 8
    idx1 = np.arange(0, n_layers, interval)[1:]
    idx2 = np.arange(1, len(idx1)+1)
    config.attn_layer_idx = (idx1 + idx2).tolist()

    logger.info(f"config: \n{config}")

    model = GenoMixMamba2ForCausalLM(config).to('cuda')

    logger.info(f"model: \n{model}")

    # calculate parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"num_params: {num_params}")


def test_tokenzation():
    tokenization_config = GenoMixTokenizationConfig()
    # tokenization_config.tokenizer_type = 'BPE_SLOW'
    # tokenization_config.tokenizer_pretrained_model_path = "/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/tokens/20000_200/T2T/BPE/5008"
    
    tokenization_config.model_max_length = 5
    logger.info(f"tokenization_config: \n{tokenization_config}")
    tokenization = GenoMixTokenization(tokenization_config)
    tokenizer = tokenization.tokenizer
    logger.info(f"tokenization.tokenizer: \n{tokenizer}")

    logger.info(f"tokenization.tokenizer.model_max_length: \n{tokenizer.model_max_length}")

    seq = 'CACCCTAAACCCTAACCCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCTAAACCCT'

    re = tokenizer(
        seq, 
        max_length=5,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=True,
        stride= 2,
        add_special_tokens=True, # default value in the __call__ function
        # return_special_tokens_mask=False if self.streaming else True, # for performance improvement
    )
    logger.info(f"re: \n{re}")



if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    # test_genomix_mamba2_model()
    # test_tokenzation()
    # char_tokenizer = CharacterTokenizer()
    # logger.info(char_tokenizer.init_kwargs)
    # char_tokenizer.save_pretrained('/home/share/huadjyin/home/baiyong01/projects/genomix/tmp')

    char_tokenizer = CharacterTokenizer.from_pretrained(
        pretrained_model_name_or_path='/home/share/huadjyin/home/baiyong01/projects/genomix/tmp',
        local_files_only=True)
    logger.info(char_tokenizer.get_vocab())
    





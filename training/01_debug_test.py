#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		train_causallm.py
@Time    :   	2024/12/17 16:26:19
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

@Desc    :   	 training script

"""

import sys
import os
import logging
import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch
import datasets
from transformers import (
    HfArgumentParser,
    set_seed,
    EvalPrediction,
)
import transformers
from transformers.utils import check_min_version
from transformers.trainer_utils import get_last_checkpoint

import evaluate

sys.path.append('..')

from training.trainer_config import GenoMixCausalLMTrainingConfig
from training.data_config import GenoMixCausalLMDataConfig
from genomix.data_builder.tokenization import GenoMixTokenizationConfig, GenoMixTokenization
from genomix.data_builder.datasets import (
    GenoMixDataIterableDatasetV1,
    GenoMixDataIterableDatasetV2,
    GenoMixDataDataset,
)
from genomix.data_builder.data_collator import GenoMixDataCollatorForLanguageModeling

from genomix.models.genomix_modeling import GenoMixMamba2Model, GenoMixMamba2ForCausalLM
from genomix.models.genomix_sampling import GenomixLongTermInitEnocder, SequenceDownSampling
from genomix.models.genomix_config import (
    GenoMixMamba2Config,
    GenoMixMamba2SSMConfig,
    GenoMixMamba2AttnConfig,
    GenoMixMamba2MoEConfig,
    GenoMixMamba2DownUpSampleConfig,
    GenoMixInputEmbeddingConfig,
    GenoMixMamba2InitializerConfig,
    GenoMixMamba2BiDirectionalConfig,
)
from genomix.trainer.trainer import GenoMixCausalLMTrainer

TRITON_CACHE_DIR = "/tmp/.triton/cache"

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((
        GenoMixCausalLMDataConfig,
        GenoMixCausalLMTrainingConfig,
    ))

    data_config, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, 
        # so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level() # log_level = 20
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    
    d_model = 256
    l = 1024*100
    b = 1

    genomix_causallm_config = GenoMixMamba2Config()
    logger.info(f"genomix_causallm_config: \n{genomix_causallm_config.to_json_string(use_diff=True)}")  
    ############################################################################
    # config model end
    ############################################################################

    toy_data = torch.randn((b, l, d_model)).to(training_args.device)
    logger.info(f"toy_data: {toy_data.shape}")
    init_encoder = GenomixLongTermInitEnocder(
        genomix_causallm_config,
        layer_ide=0,
        d_intermediate= d_model*2,
    ).to(training_args.device)
    down_sample = SequenceDownSampling(
        d_model,                                      # input hidden size of the model, e.g., d_model of the model
        downsampling_rate=256,                 # kernel_size of the conv1d
        stride=128,
    ).to(training_args.device)

    hidden_state, residual = init_encoder(toy_data)
    ds = down_sample(hidden_state)

    logger.info(f"hidden_state shape: {hidden_state.shape}")
    logger.info(f"residual: {residual.shape}")
    logger.info(f"down sample: {ds.shape}")

    logger.info("<<<<<<<<<<<<<<<<Done")


if __name__ == "__main__":
    # just used for triton cache
    # os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE_DIR

    # if not os.path.exists(TRITON_CACHE_DIR):
    #     os.makedirs(TRITON_CACHE_DIR, mode=0o777, exist_ok=True)

    main()




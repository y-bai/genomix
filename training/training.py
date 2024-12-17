#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		training.py
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datasets
from transformers import (
    HfArgumentParser,
    set_seed,
)
import transformers
from transformers.utils import check_min_version
from transformers.trainer_utils import get_last_checkpoint

sys.path.append('..')
from genomix.trainer.training_config import GenoMixCausalLMTrainingConfig
from genomix.data_builder.tokenization import GenoMixTokenizationConfig, GenoMixTokenization
from genomix.data_builder.datasets import GenoMixDataIterableDatasetV1

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((
        GenoMixCausalLMTrainingConfig,
    ))

    training_args = parser.parse_args_into_dataclasses()[0]

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
    
    ########################################
    # load tokenizer
    ########################################
    tokenization = GenoMixTokenization(
        GenoMixTokenizationConfig(
        tokenizer_type = 'CHAR_TOKEN',
        # for tokenizer initialization
        model_max_length = 1024,
    ))
    tokenizer = tokenization.get_tokenizer()

    with training_args.main_process_first(desc="loading tokenized data"):
        trn_dat = GenoMixDataIterableDatasetV1(
            '/home/share/huadjyin/home/baiyong01/projects/genomix/tmp/testdata/chm13t2t-train-input_ids.txt',
        )
        tst_dat = GenoMixDataIterableDatasetV1(
            '/home/share/huadjyin/home/baiyong01/projects/genomix/tmp/testdata/chm13t2t-test-input_ids.txt',
        )
    
    
    


if __name__ == "__main__":
    main()




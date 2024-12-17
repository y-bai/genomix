#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		training_config.py
@Time    :   	2024/12/11 11:58:46
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

@Desc    :   	 transformer trainer config

"""
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


BATCH_SIZE = 72

PRETRAIN_OUTPUT_DIR = "/home/share/huadjyin/home/baiyong01/projects/genomix/model_output"
PRETRAIN_LOG_DIR = os.path.join(PRETRAIN_OUTPUT_DIR, "log")





@dataclass
class GenoMixCausalLMTrainingConfig(TrainingArguments):
    # deepspeed = DEEPSPEED_CONFIG # r"/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/train/dspeed_config.json"
    output_dir: str = PRETRAIN_OUTPUT_DIR
    overwrite_output_dir: bool = True 

    #   
    # NOTE: to complete the whole training data during one epoch, 
    # there needs  steps = (n_whole_training_samples / (per_device_train_batch_size * n_GPU * gradient_accumulation_steps))
    # 1 epoch ~= almost 4533 steps (T2T training, 512 token_seq_len) 
    #
    learning_rate: float = 6e-4 #if not SELF_PRETAINED_MODEL else 6e-5
    # linear: transformers.get_linear_schedule_with_warmup
    # cosine: transformers.get_cosine_schedule_with_warmup
    # cosine_with_restarts: transformers.get_cosine_with_hard_restarts_schedule_with_warmup
    # polynomial: transformers.get_polynomial_decay_schedule_with_warmup
    # constant: transformers.get_constant_schedule
    # constant_with_warmup: transformers.get_constant_schedule_with_warmup
    # inverse_sqrt: transformers.get_inverse_sqrt_schedule
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L402 
    # lr_scheduler_type: str = "constant_with_warmup"
    # lr_scheduler_type: str = "linear"
    lr_scheduler_type: str = "cosine"

    warmup_steps: int = 800 
    max_steps:int = 60000  
    # num_train_epochs = 2  # https://discuss.huggingface.co/t/trainer-only-doing-3-epochs-no-matter-the-trainingarguments/19347/5

    dataloader_num_workers: int = 4  # the number of processes per GPU to use for data loading

    per_device_train_batch_size: int = BATCH_SIZE
    per_device_eval_batch_size: int = BATCH_SIZE

    gradient_accumulation_steps: int = 10

    eval_accumulation_steps: int = 50

    # NOTE: config for evaluation
    evaluation_strategy: str = "steps"
    # evaluation_strategy: str = "no"  # no evaluation during training
    do_eval: bool = True

    #
    # if evaluation_strategy="steps". eval_steps will default to the same 
    # value as logging_steps if not set.
    # eval_steps must be an integer if bigger than 1
    eval_steps: int = 50
    
    # NOTE: logging config 
    # TensorBorad log dir
    logging_dir: str = PRETRAIN_LOG_DIR
    logging_steps: int = 50 #
    logging_strategy: str = "steps"
    report_to: str = "tensorboard" 

    # NOTE: save config
    save_steps: int = 2000 
    save_strategy: str = "steps"
    save_total_limit: int = 3

    weight_decay: float = 0.1           
    adam_beta1:float = 0.9              # default for AdamW
    adam_beta2:float = 0.999            # default: 0.999
    adam_epsilon:float = 1e-8

    do_train: bool = True

    max_grad_norm:float = 1.0  # lib defult value

    sharded_ddp: bool = True   # speed up training under multi-GPU
    ddp_timeout: int = 60 * 60 * 2 # 1-hour

    # find_unused_parameters in DistributedDataParallel
    # NOTE
    ddp_find_unused_parameters: bool = False

    resume_from_checkpoint: bool = False 

    seed: int = 3407
    data_seed: int = 3407

    #
    # # If input does not contained labels, then we need to use this
    # include_inputs_for_metrics: bool = True

    #
    disable_tqdm: bool = True

    tf32: bool = True
    # fp16: bool = True
    # gradient_checkpointing: bool = True

    # for debug
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )








    







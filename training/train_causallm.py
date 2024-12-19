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
)
from genomix.data_builder.data_collator import GenoMixDataCollatorForLanguageModeling

from genomix.models.genomix_vallim2 import GenoMixMamba2Model, GenoMixMamba2ForCausalLM
from genomix.models.genomix_config import (
    GenoMixMamba2Config,
    GenoMixMamba2SSMConfig,
    GenoMixMamba2AttnConfig,
    GenoMixMamba2MoEConfig,
    GenoMixMamba2DownUpSampleConfig,
    GenoMixInputEmbeddingConfig,
    GenoMixMamba2InitializerConfig
)
from genomix.trainer.trainer import GenoMixCausalLMTrainer

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
    
    ################################################################################
    # load tokenizer
    ################################################################################
    # This is only used for save the tokenizer once the training is done.
    # because we have already tokenized the data, we don't need to train the tokenizer
    tokenization = GenoMixTokenization(
        GenoMixTokenizationConfig(
        tokenizer_type = 'CHAR_TOKEN',
        # for tokenizer initialization
        model_max_length = 1024,
    ))

    with training_args.main_process_first(desc="loading tokenized data"):
        trn_dat = GenoMixDataIterableDatasetV1(
            data_config.input_train_tokenized_input_ids_file,
        )
        tst_dat = GenoMixDataIterableDatasetV1(
            data_config.input_test_tokenized_input_ids_file,
        )

    if training_args.do_train:
        if training_args.max_train_samples is not None:
            data_list = []
            for i, i_data in enumerate(trn_dat):
                data_list.append(i_data)
                if i >= training_args.max_train_samples:
                    break
            trn_dat = data_list
    if training_args.do_eval:
        if training_args.max_eval_samples is not None:
            data_list = []
            for i, i_data in enumerate(tst_dat):
                data_list.append(i_data)
                if i >= training_args.max_eval_samples:
                    break
            tst_dat = data_list
        
        # preprocess logits for metric when evaluation
        def preprocess_logits(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]

            # print(f"metrics_labels:{labels.shape}")
            # print(f"metrics_logits:{logits.shape}")
            # NOTE: important:
            # https://discuss.huggingface.co/t/evalprediction-returning-one-less-prediction-than-label-id-for-each-batch/6958/6
            # https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/src/transformers/trainer.py#L2647
            #
            # if isinstance(outputs, dict):
            #     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            # else:
            #     logits = outputs[1:]
            #
            # NOTE: This will discard the first element if your outputs is not a dictionary. 
            # My original outputs is a tensor and I wrap it to a dictionary to solve the question.
            #
            return logits.argmax(dim=-1)
        
        metric_acc = evaluate.load("metrics/accuracy")

        def compute_metrics(eval_preds: EvalPrediction):
            # for evaluation
            preds, labels = eval_preds

            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric_acc.compute(predictions=preds, references=labels)
    
    ############################################################################
    # config model start
    ############################################################################
    # Here, we use the default configuration 
    d_model = 256
    n_layers = 32
    ssm_headdim = 32
    assert d_model / ssm_headdim % 8 == 0, "d_model / ssm_headdim % 8 != 0"
    
    attn_num_heads = 8
    attn_head_dim = d_model // attn_num_heads
    attn_rotary_emb_dim = min(attn_head_dim, 64) # 32 is the max rotary emb dim
    ssm_cfg = GenoMixMamba2SSMConfig(
        headdim=ssm_headdim,
    )
    attn_cfg = GenoMixMamba2AttnConfig(
        num_heads=attn_num_heads,
        head_dim=attn_head_dim,
        rotary_emb_dim=attn_rotary_emb_dim,
    )
    moe_cfg = GenoMixMamba2MoEConfig()
    down_up_cfg = GenoMixMamba2DownUpSampleConfig()
    input_emb_cfg = GenoMixInputEmbeddingConfig()
    initializer_cfg = GenoMixMamba2InitializerConfig()

    # add attn layers 
    interval = 4
    idx1 = np.arange(0, n_layers, interval)[1:]
    idx2 = np.arange(0, len(idx1))
    attn_idx = idx1 + idx2
    if attn_idx[-1] >= n_layers:
        attn_idx = attn_idx[:-1]

    assert attn_idx[-1] < n_layers, "attn_idx[-1] >= n_layers"
    attn_layer_idx = attn_idx.tolist()

    logger.info(f"number of layers: {n_layers}, attn_layer_idx: {attn_layer_idx}")
    
    genomix_causallm_config = GenoMixMamba2Config(
        vocab_size=tokenization.tokenizer.vocab_size,
        d_model =d_model,
        n_layers = n_layers,
        
        pad_token_id=tokenization.tokenizer.pad_token_id,
        bos_token_id=tokenization.tokenizer.bos_token_id,
        eos_token_id=tokenization.tokenizer.eos_token_id,

        attn_layer_idx = attn_layer_idx,
        moe_layer_idx=[],

        down_up_sample=False,
        down_up_sample_cfg=down_up_cfg.to_dict(),

        input_embedding_cfg=input_emb_cfg.to_dict(),
        ssm_cfg=ssm_cfg.to_dict(),
        attn_cfg=attn_cfg.to_dict(),
        moe_cfg=moe_cfg.to_dict(),
        initializer_cfg=initializer_cfg.to_dict(),
    )
    logger.info(f"genomix_causallm_config: \n{genomix_causallm_config}")
    ############################################################################
    # config model end
    ############################################################################

    # initialize mode
    genomix_causallm_model = GenoMixMamba2ForCausalLM(
        genomix_causallm_config
    )

    logger.info(f"model: \n{genomix_causallm_model}")
    # logger.info(f"device: {genomix_causallm_model.device}")
    logger.info(f"num params: {genomix_causallm_model.num_parameters()}")
    logger.info(f"num trainable params: {genomix_causallm_model.num_parameters(only_trainable=True)}")

    logger.info(f"^^^^^^^^tf32 is set: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"^^^^^^^^fp16 = {training_args.fp16}")
    logger.info(f"^^^^^^^^Learning rate: {training_args.learning_rate}")
    logger.info(f"^^^^^^^^LR scheduler type : {training_args.lr_scheduler_type}")


    ############################################################################
    # training
    ############################################################################
    # Initialize our Trainer
    trainer = GenoMixCausalLMTrainer(
        model=genomix_causallm_model,
        args=training_args,
        train_dataset=trn_dat if training_args.do_train else None,
        eval_dataset=tst_dat if training_args.do_eval else None,
        tokenizer=tokenization.tokenizer,
        data_collator=GenoMixDataCollatorForLanguageModeling(),
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits if training_args.do_eval else None,
    )

    logger.info(">>>>>>>>>>>>>>>>Start training and evaluatoin......")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        # max_train_samples = (
        #     training_args.max_train_samples if training_args.max_train_samples is not None else len(trn_pt_ds)
        # )

        # metrics["train_samples"] = min(max_train_samples, len(trn_pt_ds))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        logger.info(">>> GenoMix Start evaluation......")

        metrics = trainer.evaluate()

        # max_eval_samples = training_args.max_eval_samples if training_args.max_eval_samples is not None else len(training_args)
        
        # metrics["eval_samples"] = min(max_eval_samples, len(val_pt_ds))
        
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # with training_args.main_process_first(desc="clean scRNA cache files"):
    #     trn_dat.cleanup_cache_files()

    logger.info("<<<<<<<<<<<<<<<<Done")


if __name__ == "__main__":
    main()




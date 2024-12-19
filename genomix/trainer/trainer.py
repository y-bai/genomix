#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		trainer.py
@Time    :   	2024/12/17 14:36:41
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

@Desc    :   	 trainer for training model

"""
import time
from typing import Optional
import torch
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class GenoMixCausalLMTrainer(Trainer):

    def __init__(
        self,
        *args,
        loss_fn=None,
        **kwargs
    ):
        super(GenoMixCausalLMTrainer, self).__init__(*args, **kwargs)
        self.loss_fn = (
            loss_fn
            if loss_fn is not None 
            else torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor)
        )

    # override the compute_loss function
    # https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for the given inputs.

        Args:
            model (:obj:`torch.nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                - for the causal language model, the inputs are the `input_ids`.

        Returns:
            :obj:`torch.FloatTensor`:
                The loss.

        """
        # t_start = time.time()
        inputs = self._prepare_inputs(inputs) # inputs is a dict with keys: ['input_ids']

        input_ids = inputs.pop('input_ids')
        labels = inputs.pop('labels')

        if self.args.fp16:
            with torch.autocast(self.args.device.type):
                # outputs = model(**inputs)
                outputs = model(input_ids=input_ids)

                shift_logits = outputs["logits"][:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = self.loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1))

                ## NOTE: The following code will introduce issue of 
                ## "RuntimeError: view size is not compatible with input tensor's size and stride"
                # shift_logits = outputs["logits"][:, :-1, :]
                # shift_labels = labels[:, 1:]
                # loss = self.loss_fn(
                #     shift_logits.view(-1, shift_logits.size(-1)).contiguous(), 
                #     shift_labels.view(-1).contiguous())
        else:
            # outputs = model(**inputs)
            outputs = model(input_ids=input_ids)

            shift_logits = outputs["logits"][:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            y = shift_logits.view(-1, shift_logits.size(-1))

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1))
        
        # torch.cuda.current_stream().synchronize()  
        # t_seq_emd_end = time.time()
        # print(f"LOSS time: {t_seq_emd_end - t_start}")
            
        return (loss, outputs) if return_outputs else loss
        
    # Overried the `create_scheduler` function.
    def create_scheduler(
        self, 
        num_training_steps: int, 
        optimizer: torch.optim.Optimizer = None
    ):
        if self.lr_scheduler is None:
            n_warm_steps = self.args.get_warmup_steps(num_training_steps)
            _no_warm_steps = num_training_steps - n_warm_steps
            if self.args.lr_scheduler_type == "cosine_with_restarts":
                _scheduler_specific_kwargs = {
                    "num_cycles": max(10, _no_warm_steps//self.args.num_steps_per_cycle)}
                    # "num_cycles": 2}
            else: 
                _scheduler_specific_kwargs = {}

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=n_warm_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=_scheduler_specific_kwargs,
            )
            self._created_lr_scheduler = True

        return self.lr_scheduler
    
    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        # logger.warning("Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.")
        return self.processing_class
    
    # To solve the issue:
    # RuntimeError: Some tensors share memory, this will lead to duplicate memory on disk and potential 
    # differences when loading them again: [{'lm_head.weight', 'backbone.embedding.weight'}].
    # A potential way to correctly save your model is to use `save_model`.
    # More information at https://huggingface.co/docs/safetensors/torch_shared_tensors
    def save_model(self, output_dir, _internal_call:bool = False):
        self.model.save_pretrained(output_dir)
        # self.model.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


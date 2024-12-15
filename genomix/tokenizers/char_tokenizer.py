"""
Just a simple character level tokenizer.

adapted from: https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py

CharacterTokenzier for Hugging Face Transformers.
This is heavily inspired from CanineTokenizer in transformers package.
"""

import os
import json
from pathlib import Path
import logging

from typing import Dict, List, Optional, Sequence, Tuple, Union
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer    

from ..utils import read_json_file
from ..utils.constants import SPECIAL_TOKENS

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "char_vocab.json"}

# Adapted from HyenaDNA tokenizer 
class CharacterTokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self, 
        vocab_file: str = None,
        characters: Sequence[str] = ['A', 'C', 'G', 'T', 'N'], 
        bos_token: Optional[str] = '<BOS>',
        eos_token: Optional[str] = '<EOS>',
        unk_token: Optional[str] = '<UNK>',
        mask_token: Optional[str] = '<MASK>',
        model_max_length: int = 16002, 
        padding_side: str='left',
        truncation_side: str='right',
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        add_prefix_space=False, 
        do_lower_case=False,
        clean_up_tokenization_spaces=True,
        **kwargs,
    ):
        self._config = {}
        if vocab_file is not None:
            # Load from file
            self._config = read_json_file(vocab_file)
        else:
            # config
            self._config = {
                bos_token: 0,
                eos_token: 1,
                unk_token: 2,
                mask_token: 3,
                **{ch: i + 4 for i, ch in enumerate(characters)},
            }

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self._vocab_str_to_int = self._config
        assert len(self._vocab_str_to_int) > 0, "No vocabulary is provided."
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        mask_token = AddedToken(mask_token, lstrip=False, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            do_lower_case=do_lower_case,
            model_max_length=model_max_length,
            padding_side=padding_side,
            truncation_side=truncation_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs if vocab_file is None else {},
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)
    
    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int[SPECIAL_TOKENS.UNK.value])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An 
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
        | first sequence      | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """

        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []
        
        output = len(bos_token_id + token_ids_0 + eos_token_id) * [0]

        if token_ids_1 is not None:
            output += len(token_ids_1 + eos_token_id) * [1]
        
        return output

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
    #     cfg_file = os.path.join(pretrained_model_name_or_path, 
    #                             (filename_prefix + "-" if filename_prefix else "") + "char_vocab.json")
    #     with open(cfg_file) as f:
    #         cfg = json.load(f)
    #     return cls.from_config(cfg)

    def save_vocabulary(self, save_directory, filename_prefix = None):

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        cfg_file = os.path.join(save_directory, 
                                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        
        with open(cfg_file, 'w') as f:
            json.dump(self._config, f, indent=2)
        return tuple(cfg_file)
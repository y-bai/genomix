#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_validator.py
@Time    :   	2024/11/21 12:05:07
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

@Desc    :   	define decorators for functions

"""

from collections.abc import Mapping
from functools import wraps
from itertools import chain
from typing import Any, List, Union
from inspect import signature

def validate_args(*, name: str, required_keys_or_values: Union[Any, List]=None):
    """Validate the dict or list parameters in the function signature

    Parameters
    ----------
    name : str
        the name of the parameter in the function signature
    required_keys_or_values : Any, optional
        if `name` is a `dict` type parameter in the original function, 
            then `required_keys_or_values` is the required keys in the dict.
        if `name` is a `list` type parameter in the original function, 
            then `required_keys_or_values` is the required values in the list.
        if `name` is a `int` or `float` or `str` type parameter in the original function, 
            then `required_keys_or_values` is just a value.
    Raises
    ------
    ValueError
        if the required keys are not in the kwargs


    Example:
    --------
    >>> @validate_args(name='train_kwargs', required_keys_or_values=['data_iterator','vocab_size'])
    ... def bpe_tokenizer_train(
    ...     train_kwargs: dict
    ... ):
    ...     pass


    """

    if not isinstance(required_keys_or_values, List):
        required_keys_or_values = [required_keys_or_values]
    def _inner_check_parameters(func):
        sig = signature(func)
        @wraps(func)
        def inner_func(*args, **kwargs):
            for arg_name, arg_value in chain(
                zip(sig.parameters, args),  # Args values
                kwargs.items(),  # Kwargs values
            ):
                if arg_name == name:
                    for key_or_vals in required_keys_or_values:
                        if isinstance(arg_value, Mapping):
                            if key_or_vals not in arg_value:
                                raise ValueError(f"Missing required '{key_or_vals}' in '{name}' parameters")
                        else:
                            if key_or_vals != arg_value:
                                raise ValueError(f"Provided {arg_value} is not equal to the " 
                                                 f"required {key_or_vals} by '{name}' parameters")
           
            return func(*args, **kwargs)
        return inner_func
    return _inner_check_parameters


#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_decorators.py
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

from functools import wraps
from typing import Any, List
from inspect import signature

def validate_parameter(*, name: str, required_keys_or_values: List=None):
    """Validate the dict or list parameters in the function signature

    Parameters
    ----------
    name : str
        the name of the parameter in the function signature
    required_keys_or_values : Any, optional
        if name is a dict, then required_keys_or_values is the required keys in the dict.
        if name is a list, then required_keys_or_values is the required values in the list.
    Raises
    ------
    ValueError
        if the required keys are not in the kwargs
    """
    if isinstance(required_keys, str):
        required_keys = [required_keys]
    def _inner_check_parameters(func):
        sig = signature(func)
        @wraps(func)
        def inner_func(*args, **kwargs):
            # bind the passed arguments to the function signature
            bound_args = sig.bind(*args, **kwargs)
            # # ensure that default values are applied to any missing arguments
            # bound_args.apply_defaults()
            # find out kwargs parameters
            func_kwargs = bound_args.arguments
            param = func_kwargs[name]
            for key_or_vals in required_keys_or_values:
                if key_or_vals not in param:
                    raise ValueError(f"Missing required key '{key_or_vals}' in '{name}' parameters")
           
            return func(*args, **kwargs)
        return inner_func
    return _inner_check_parameters


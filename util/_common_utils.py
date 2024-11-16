#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_common_utils.py
@Time    :   	2024/11/15 16:03:54
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

@Desc    :   	None

"""


from collections import OrderedDict
import os
import json
from itertools import tee, filterfalse


def check_file_exist(f_name):
    if not os.path.exists(f_name):
        raise FileNotFoundError(f"{f_name} not found.")


def check_dir_exist(dir_path, create=True):
    if not os.path.exists(dir_path):
        if create:
            os.makedirs(dir_path)
        else:
            raise FileNotFoundError(f"{dir_path} not found.")


def partition(condition, iterable_input):
    """partition the input list into two lists based on the condition

    >>> patition_list(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])
    ([2, 4], [1, 3, 5])

    Parameters
    ----------
    condition : labmda function
        the condition to partition the input list
    iterable_input : can be list, tuple, set
        iterable input data to be partitioned
    
    Returns
    -------
    tuple
        a tuple containing two lists partitioned based on the condition
        
    """
    t1, t2 = tee(iterable_input)
    return list(filter(condition, t1)), list(filterfalse(condition, t2))


def read_json(f_name):
    with open(f_name, "rt", encoding="utf-8") as f:
        # return ordered dict
        dt_conf = json.load(f, object_pairs_hook=OrderedDict)
    return dt_conf


def write_json(f_name, data):
    with open(f_name, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Save data to {f_name} done!")
#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		common.py
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

@Desc    :   	common functions for file operations, data processing, etc.

"""

from collections import OrderedDict
import os
import json
from itertools import tee, filterfalse
from typing import List
import multiprocessing as mp
from tqdm import tqdm
import logging
import shutil

from .constants import LARGE_FILE_SIZE, BATCH_NUM_SEQS

logger = logging.getLogger(__name__)

__all__ = [
    'check_file_exists',
    'check_dir_exists',
    'read_json_file',
    'write_json_file',
    'read_txt_file',
    'write_txt_file',
    'copy_file',
    'partition_list',
]

def check_file_exists(f_name, _raise_error=True):
    if not os.path.exists(f_name):
        if _raise_error:
            raise FileNotFoundError(f"{f_name} not found.")
        else:
            return False
    else:
        return True


def check_dir_exists(dir_path, create=True, _raise_error=True):
    if not os.path.exists(dir_path):
        if create:
            print(f'{dir_path} not existing, create a new one.')
            os.makedirs(dir_path)
            return dir_path
        else:
            if _raise_error:
                raise FileNotFoundError(f"{dir_path} not found.")
            else:
                return None
    else:
        return dir_path


def read_json_file(f_name):
    with open(f_name, "rt", encoding="utf-8") as f:
        # return ordered dict
        dt_conf = json.load(f, object_pairs_hook=OrderedDict)
    return dt_conf


def write_json_file(f_name, data, indent=4):
    with open(f_name, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


# TODO: add gzip support
def read_txt_file(
        f_name, 
        n_proc: int=16, 
    ):
    """Read text file """
    
    check_file_exists(f_name)

    if os.stat(f_name).st_size <= LARGE_FILE_SIZE:
        return _read_samll_txt(f_name)
    else:
        return _parallel_read_txt(
            f_name, 
            batch_size=BATCH_NUM_SEQS, 
            n_proc=n_proc)


def write_txt_file(
        f_name: str, 
        data: List[str], 
        n_proc: int=16, 
        disable_tqdm: bool=False,
    ):
    """Write a list of strings to a text file."""

    if len(data) <= BATCH_NUM_SEQS:
        _write_small_txt(
            f_name, data, 
            mode='w',
            disable_tqdm=disable_tqdm
        )
    else:
        _parallel_write_txt(
            f_name, 
            data, 
            batch_size=BATCH_NUM_SEQS, 
            n_proc=n_proc, 
            disable_tqdm=disable_tqdm
        )


def copy_txt_file(src_file, dst_file):
    """Copy a txt file from source to destination."""
    if os.stat(src_file).st_size <= LARGE_FILE_SIZE:
        shutil.copy(src_file, dst_file)
    else:
        with open(src_file, "rt") as src_f, open(dst_file, "wt") as dst_f:
            while True:
                chunk = src_f.read(BATCH_NUM_SEQS)
                if not chunk:
                    break
                dst_f.write(chunk)


def partition_list(condition, iterable_input):
    """partition the input list into two lists based on the condition

    >>> patition(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])
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

def _read_samll_txt(f_name):
    data = []
    with open(f_name, "rt", encoding="utf-8") as f:
        # read line by line
        for line in f:
            data.append(line.strip('\n')) # remove '\n' at the end
    return data

def _write_small_txt(f_name, data, mode: str = 'w', disable_tqdm: bool=False):
    """Write a list of strings with small count to a text file."""
    with open(f_name, mode, encoding="utf-8") as f:
        # write a list of lines, each line contains a '\n' at the end
        for _seq in tqdm(data, desc="writing file", disable=disable_tqdm):
            f.write(_seq + '\n')

def _write_chunk_txt(
        f_name: str, 
        batch: List[str], 
        ith_batch: int, 
        mode: str, 
        disable_tqdm: bool
    ):
    """Write a chunk of lines to a file."""

    with open(f_name, mode, encoding='utf-8') as f:
        # write a list of lines, each line contains a '\n' at the end
        for _seq in tqdm(batch, desc=f"{ith_batch}th batch writing", disable=disable_tqdm):
            f.write(_seq + '\n')

def _write_chunk_helper(args):
    return _write_chunk_txt(*args)

def _parallel_write_txt(
        file_name: str, 
        data: List[str], 
        batch_size: int = 5000, 
        n_proc: int = 16, 
        disable_tqdm: bool = False
    ):
    """Write a list of strings to a text file in parallel."""
    # Split data into chunks
    batchs = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    logger.info(f"writing {len(batchs)} batchs to {file_name}")

    # Use multiprocessing Pool to write chunks in parallel
    with mp.Pool(processes=n_proc) as pool:
        pool.starmap(
            _write_chunk_txt, 
            [(file_name, batch, i, 'a', disable_tqdm) for i, batch in enumerate(batchs)]
        )
        # for result in pool.imap(
        #     _write_chunk_helper, 
        #     [(file_name, batch, i, 'a', disable_tqdm) for i, batch in enumerate(batchs)]
        # ):
        #     pass # The function _write_chunk_txt handles the writing


def _read_chunk_txt(file_name: str, start_line: int, num_lines: int) -> List[str]:
    """Read a chunk of lines from a file."""
    lines = []
    with open(file_name, "rt", encoding="utf-8") as f:
        for current_line, line in enumerate(f):
            if current_line >= start_line and current_line < start_line + num_lines:
                lines.append(line.strip())
            elif current_line >= start_line + num_lines:
                break
    return lines

def _parallel_read_txt(file_name: str, batch_size: int = 5000, n_proc: int = 16) -> List[str]:
    """Read a large text file in parallel."""
    # Determine the total number of lines in the file
    with open(file_name, "rt", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    # Create chunks based on line numbers
    batchs = [(file_name, i, batch_size) for i in range(0, total_lines, batch_size)]

    logging.info(f"reading {len(batchs)} batchs, total {total_lines} lines from {file_name}")


    # Use multiprocessing Pool to read chunks in parallel
    with mp.Pool(processes=n_proc) as pool:
        results = pool.starmap(_read_chunk_txt, batchs)
    
    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results
    


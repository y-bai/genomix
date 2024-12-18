#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_cache.py
@Time    :   	2024/11/22 09:19:25
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

@Desc    :   	functions for caching files

"""

import os
from collections import OrderedDict
import json
import logging
import gzip
import shutil

from .constants import GENOMIX_DATA_DIR, LARGE_FILE_SIZE
from .common import check_file_exists, check_dir_exists

logger = logging.getLogger(__name__)


def _find_from_cache( 
    file_meta: OrderedDict=None, 
    cache_dir:str=None,
):
    """check if file alrady exists in cache

    Parameters
    ----------
    file_meta : OrderedDict, optional
        file meta information, by default None

    cache_dir : str, optional
        cache directary, by default None
        - If None, use GENOMIX_CACHE_DATA_DIR

    Returns
    -------
    str, int
        cahce file path, file size

    None
        if file not found
    """
    if cache_dir is None:
        cache_dir = GENOMIX_DATA_DIR
    
    fname = _file_name(cache_dir, file_meta)
    if not check_file_exists(fname, _raise_error=False):
        logger.warning(f"Cache file not found.")
        return None, 0
    
    f_size = os.stat(fname).st_size
    return str(fname), f_size


def _write_to_cache(
    fname_to_cache: str,
    file_meta: OrderedDict=None, 
    cache_dir:str=None,
):
    """write file to cache as gzip file

    Parameters
    ----------
    fname_to_cache : str
        file name, with full path

    file_meta : OrderedDict, optional
        file meta information, by default None

    cache_dir : str, optional
        cache directary, by default None
        - If None, use GENOMIX_CACHE_DATA_DIR

    Returns
    -------
    bool
        True if save successfully, otherwise False
    """
    if cache_dir is None:
        cache_dir = GENOMIX_DATA_DIR
    
    _cache_fname = _file_name(cache_dir, file_meta)

    # write file into cache as gzip file
    with open(fname_to_cache, 'rb') as f_in, gzip.open(_cache_fname, 'wb') as f_out:
        # shutil.copyfileobj(f_in, f_out, length=LARGE_FILE_SIZE)
        _write_file_by_buffer(f_in, f_out, buffer_size=LARGE_FILE_SIZE)
    

def _md5(data_meta: str):
    import hashlib
    hex_dig = hashlib.md5(data_meta.encode()).hexdigest()
    return hex_dig

def _meta_md5(file_meta: OrderedDict=None):
    meta_info = '' if file_meta is None else json.dumps(file_meta)
    meta_md5 = _md5(json.dumps(meta_info))
    return meta_md5

def _file_name(cache_dir: str, file_meta: OrderedDict=None):
    meta_dir = _meta_md5(file_meta)
    cache_full_dir = os.path.join(cache_dir, meta_dir)
    check_dir_exists(cache_full_dir, create=True)
    return os.path.join(cache_full_dir, f"{meta_dir}")


def _resolve_cache_file(src_cache_file, dst_file):
    """Copy a txt file from source to destination."""

    # src_cache_file: gzipped file
    with gzip.open(src_cache_file, "rb") as src_f, open(dst_file, "wb") as dst_f:
        _write_file_by_buffer(src_f, dst_f, buffer_size=LARGE_FILE_SIZE)


def _write_file_by_buffer(f_in, f_out, buffer_size):
    while True:
        block = f_in.read(buffer_size)
        if not block:
            break
        f_out.write(block)
    
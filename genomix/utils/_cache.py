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

from collections import OrderedDict
import shutil
from pathlib import Path
import json
import logging

from .constants import GENOMIX_CACHE_DATA_DIR
from .common import check_file_exists, check_dir_exists, copy_file

logger = logging.getLogger(__name__)


def load_from_cache(
    file_name: str, 
    file_meta: OrderedDict=None, 
    cache_dir:str=None,
):
    """Load file from cache

    Currently, only support txt file

    Parameters
    ----------
    file_name : str
        file name, not with full path
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
        cache_dir = GENOMIX_CACHE_DATA_DIR
    
    meta_dir = _meta_dir(file_meta)
    
    f_path = Path(cache_dir) / meta_dir / file_name

    if not check_file_exists(str(f_path), _raise_error=False):
        logger.warning(f"Cache file {f_path} not found.")
        return None, 0
    
    f_size = f_path.stat().st_size
    return str(f_path), f_size


def save_to_cache(
    file_name: str, 
    file_meta: OrderedDict=None, 
    cache_dir:str=None,
):
    """Save file to cache

    Currently, only support txt file

    Parameters
    ----------
    file_name : str
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
        cache_dir = GENOMIX_CACHE_DATA_DIR
    
    meta_dir = _meta_dir(file_meta)
    
    f_path = Path(cache_dir) / meta_dir

    cache_dir = check_dir_exists(str(f_path), create=True, _raise_error=False)
    if cache_dir is None:
        logger.warning(f"Cache directory {f_path} not found.")
        return False
    
    # check the source file
    if not check_file_exists(file_name, _raise_error=False):
        logger.warning(f"cache file {file_name} failed.")
        return False
    
    fscr = Path(file_name)
    fdst = f_path / fscr.name
    copy_file(file_name, str(fdst))

    return True


def _meta_dir(file_meta: OrderedDict=None):
    meta_info = '' if file_meta is None else json.dumps(file_meta)
    meta_dir = _md5(json.dumps(meta_info))
    return meta_dir

# filename, file_length,
def _md5(data_meta: str):
    import hashlib
    hex_dig = hashlib.md5(data_meta.encode()).hexdigest()
    return hex_dig


#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		__init__.py
@Time    :   	2024/11/17 14:24:37
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

from .common import (
    check_file_exists,
    check_dir_exists,
    read_json_file,
    write_json_file,
    read_txt_file,
    write_txt_file,
    partition_list,
    copy_txt_file,
)

from .sequence import (
    chunk_sequence,
    down_sampling,
    batch_iterator,
)


# # number of processors to be used for parallel processing
# # N_PROC = 16 # os.cpu_count(), minmum is 1
# def set_n_proc(n: int):
#     global N_PROC
#     N_PROC = n

# # tqdm disable flag
# # TQDM_DISABLE = False
# def set_tqdm_disable(flag: bool):
#     global TQDM_DISABLE
#     TQDM_DISABLE = flag





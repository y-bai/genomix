#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_corpus_utils.py
@Time    :   	2024/11/19 16:31:56
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

import os
from typing import List, Optional, Union
from datasets import DatasetDict, load_from_disk

from ._common_utils import check_dir_exists
from ._sequence_utils import chunk_sequence

def gen_corpus_txt(
        input_ds_names: Union[str, List[str]],
        ds_dict_key: str='train',
        ds_feature_name: str='sequence',
        output_dir: Optional[str]=None,
        output_txt_suffix: Optional[str]=None,
        num_downsamples: int=400000,
        chunk_size: int=20000,
        overlap_step: int=200
):
    """Generate training corpus txt file from the input dataset

    Parameters
    ----------

    input_ds_names : str,
        input HuggingFace dataset names/paths, such as '/path/to/dataset_name'
        The datasets were generated by the script dataset/dataset_gen.py
        - Could be a list of dataset names/paths
    
    ds_dict_key : str,
        key of the dataset to be used, by default 'train'
        - Could be 'train', 'validation', 'test'
        - This is for the case that the input dataset contains multiple splits, 
            that is, the input dataset is a DatasetDict object.
        - It is not used if the input dataset is a Dataset object.
    
    ds_feature_name : str,
        feature name of the dataset to be used, by default 'sequence'
        - This is for the case that the input dataset contains multiple features,
            and the feature contains the sequence data.
             
    output_dir : str, optional
        output directory to save the generated txt file, by default None
        if None, then save the txt file in the input directory

    output_txt_suffix : str
        suffix of the output txt file

    num_downsamples : int, optional
        number of downsampled examples, by default 400000
        - 0: no downsampling
        - This is used for downsampling the training corpus when the corpus is too large.
        - Such as the multi-species dataset and the 1000G dataset.

    chunk_size : int, optional
        size of each chunk, by default 20000
        - 0: no chunking

    overlap_step : int, optional
        overlap size between two adjacent chunks, by default 200
        - when chunk_size is -1, then overlap_step is not used.

    """

    assert output_dir is not None, "output_dir must be provided."
    check_dir_exists(output_dir, create=True)

    if isinstance(input_ds_names, str):
        input_ds_names = [input_ds_names]
    
    output_txt_suffix = "" if output_txt_suffix is None else output_txt_suffix


    seqs = []
    for _name in input_ds_names:
        check_dir_exists(_name, create=False)

        print(f">>>working on {_name}")
        # load dataset
        _ds = load_from_disk(_name)
        print(_ds)
        sequences = _ds[ds_dict_key][ds_feature_name] if isinstance(_ds, DatasetDict) else _ds[ds_feature_name]
        if isinstance(chunk_size, int) and chunk_size > 0:
            print(f"{''.join([' '] * 6)}chunking sequences...")
            for _seq in sequences:
                seqs.extend(chunk_sequence(_seq, chunk_size, overlap_step))
        else:
            # only retrive train corpus
            seqs.extend(sequences)

    if isinstance(num_downsamples, int) and num_downsamples > 0:
        print(f"{''.join([' '] * 5)} ---- downsampling training corpus...")
        
        rand_indx = np.random.permutation(np.arange(len(seqs)))[:num_downsamples]
        out_file_name = os.path.join(out_dir, f"train_corpus_{num_downsamples//1000}K_{output_txt_suffix}.txt")
        print(f"generating whole training corpus...")
        with open(out_file_name, 'w', encoding='utf-8') as init_f:
            for n_example in rand_indx: # range(num_downsamples):
                init_f.write(seqs[n_example] + '\n')
    else: 
        out_file_name = os.path.join(out_dir, f"train_corpus_{output_txt_suffix}.txt")

        print(f"generating whole training corpus...")
        with open(out_file_name, 'w', encoding='utf-8') as f:
            for _seq in seqs:
                f.write(_seq + '\n')

    print("======Done")
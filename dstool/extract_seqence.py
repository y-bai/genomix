#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		extract_seqence.py
@Time    :   	2024/12/15 14:56:34
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

@Desc    :   	 None

"""
import os
from pathlib import Path
from Bio import SeqIO
import re 
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)

MIN_SEQ_LENGTH = 20000

def extract_seq_from_fasta(
    fasta_file: str,
    output_path: str,
    num_proc = 16,
    re_partten_str = r"cluster_(\d*|\w)_contig_(\d*|\w)",
    test_chr: str = ['22']
):
    """_summary_

    Parameters
    ----------
    fasta_file : str
        input fasta file path.
    output_path : str
        output path to save the extracted sequences.
    num_proc : int, optional
        the number of processes to use, by default 16

    re_partten_str: regexp, optional
        the regular expression pattern to match the fasta description, by default r"cluster_(\d*|\w)_contig_(\d*|\w)"
        - chm13 t2t fasta: r"NC_\d*.\d* Homo sapiens isolate \w*\d* chromosome (\d*|\w), alternate assembly T2T-CHM13v2.0"
        - crgd t2t : r"chr(\d*|\w)"
        - 1kg: r"cluster(\d*|\w)_contig_\d*|\w"
        - multispecies: None

    test_chr: the chromosome ids used for test set
        - if None, then whole data used for train set

    Returns
    -------
    _type_
        _description_
    """

    input_file_base_name = Path(fasta_file).stem
    fasta_sequences = SeqIO.parse(fasta_file, 'fasta')
    with mp.Pool(processes=num_proc) as pool:
        results = pool.starmap(
            _parse_seq, 
            [(record, re_partten_str) for record in fasta_sequences]
        ) 
    
    if test_chr is not None and len(test_chr)>0:
        test_output_fname = os.path.join(output_path, f"{input_file_base_name}_TEST.txt")
        if os.path.exists(test_output_fname):
            os.remove(test_output_fname)
        f_test = open(test_output_fname, 'a+')

    train_output_fname = os.path.join(output_path, f"{input_file_base_name}_TRAIN.txt")
    if os.path.exists(train_output_fname):
        os.remove(train_output_fname)
    f_train = open(train_output_fname, 'a+')

    for chr_id, seq in results:
        if len(seq) < MIN_SEQ_LENGTH:
            continue
        if test_chr is not None and chr_id in test_chr:
            f_test.write(f"{seq}\n")
        else:
            f_train.write(f"{seq}\n")
    f_test.close()
    f_train.close()
    

def _parse_seq(record, re_partten_str):
    
    sequence, description = str(record.seq), record.description
    # CHM13 t2t fasta description:
    # NC_060946.1 Homo sapiens isolate CHM13 chromosome 22, alternate assembly T2T-CHM13v2.0
    # NC_060947.1 Homo sapiens isolate CHM13 chromosome X, alternate assembly T2T-CHM13v2.0
    # NC_060948.1 Homo sapiens isolate NA24385 chromosome Y, alternate assembly T2T-CHM13v2.0
    if re_partten_str is not None:
        prog = re.compile(re_partten_str)
        regex_match = prog.match(description)
        chr_id = regex_match[1]
        return chr_id, _clean_sequence(sequence)
    else: 
        return 'None', _clean_sequence(sequence)


def _filter_func(char: str) -> str:
    """
    Transforms any letter different from a base nucleotide into an 'N'.

    Parameters
    ----------
    char : str
        input char in upper.

    Returns
    -------
    str
        output char after being transformed.
    """

    if char in {'A', 'T', 'C', 'G'}:
        return char
    else:
        return 'N'


def _clean_sequence(seq: str) -> str:
    """
    Process a chunk of DNA to have all letters in upper and restricted to
    A, T, C, G and N.

    Parameters
    ----------
    seq : str
        input DNA sequence. 

    Returns
    -------
    str
        output DNA sequence after being filtered and cleaned.
    """

    seq = seq.upper()
    seq = map(_filter_func, seq)
    seq = ''.join(list(seq))
    return seq
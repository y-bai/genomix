#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		setup.py
@Time    :   	2024/11/22 16:52:20
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

from setuptools import setup, find_packages

setup(
   name='genomix',
   version='0.0.1',
   description='genomix: LLM for genomics data analysis',
   license="MIT",
   author='Bai Yong',
   author_email='yong.bai@hotmail.com',
   url="https://github.com/y-bai/genomix",
   package_dir={"": "genomix"},
   packages=find_packages('genomix'), 
   install_requires=[
        'numpy>=1.26.4,<2',
        'pandas>=2.2.2',
        'torch>=2.4.1',
        'transformers>=4.46.2',
        'sentencepiece>=0.2.0',
        'tqdm>=4.66.5',
        'biopython>=1.84',
   ], 
   scripts=[
            'genomix/tokenizers',
            'genomix/utils',
            'genomix/tools',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.9.0',
)
#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
			
cxx_args = ['-std=c++17']
nvcc_args = [
	'-D__CUDA_NO_HALF_OPERATORS__',
    '-std=c++17',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    #'-gencode', 'arch=compute_60,code=sm_60',
    #'-gencode', 'arch=compute_61,code=sm_61'
]

setup(
    name='PCONV2',
    packages=['PCONV2_operator'],
    ext_modules=[
        CUDAExtension('PCONV2', [
            './extension/main.cpp',
            './extension/math_cuda.cu',
            './extension/projects_cuda.cu',
            './extension/sphere_slice_cuda.cu',
			'./extension/sphere_uslice_cuda.cu',
			'./extension/pseudo_context_cuda.cu',
			'./extension/pseudo_pad.cu',
			'./extension/pseudo_fill_cuda.cu',
			'./extension/pseudo_entropy_context_cuda.cu',
			'./extension/pseudo_entropy_pad_cuda.cu',
			'./extension/string2class.cc',
   			'./extension/pseudo_merge_cuda.cu',
			'./extension/pseudo_split_cuda.cu',
			'./extension/entropy_context_cuda.cu',
            './extension/acc_grad_cuda.cu'
        ],
        include_dirs=['./extension'], 
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}, 
        libraries=['cublas'])
    ],
    cmdclass={
        'build_ext': BuildExtension
})

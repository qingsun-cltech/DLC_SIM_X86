import math
import os
import re

DLC_Custom_Kernel_PATH = r'/home/qingsun/DLC_Custom_Kernel/dlc_kernels'
X86_PATH = r'/home/qingsun/x86/dlc_kernels'

# 清理 .h
def clear_file(file, suf, file_name, depth):
  F = []
  F.append('#pragma once\n')
  F.append(f'#include "{suf}../dlc-intrinsics.h"\n')
  F.append(f'#include "{suf}../typehint.h"\n')
  F.append('\n')

  for line in file:
    res = line

    if r'tensor' in res:
      pattern = r'\btensor\b'
      res = re.sub(pattern, r'SIM_X86::tensor', res)
    if r'constexpr' in res:
      pattern = r'\bconstexpr\b'
      res = re.sub(pattern, r'/* constexpr */', res)
    if r'concept' in res:
      pattern = r'\bconcept\b'
      res = re.sub(pattern, r'/* concept */', res)
    if r'requires' in res:
      pattern = r'\brequires\b'
      res = re.sub(pattern, r'/* requires */', res)
    if r'->' in res:
      pattern_smem = r'(= )(\b\w+\b)(->smem_addr)'
      pattern_vmem = r'(= )(\b\w+\b)(->vmem_addr)'
      pattern_cmem = r'(= )(\b\w+\b)(->cmem_addr)'
      res = re.sub(pattern_smem, r'\1*(SIM_X86::tensor*)\2\3', res)
      res = re.sub(pattern_vmem, r'\1*(SIM_X86::tensor*)\2\3', res)
      res = re.sub(pattern_cmem, r'\1*(SIM_X86::tensor*)\2\3', res)
    if r'address' in res:
      pattern = r'(\b\w+\[\d+\])(\.address)'
      res = re.sub(pattern, r'*(SIM_X86::tensor*)\1\2', res)
    if r'(void *)((int)' in res:
      res = res.replace(r'(void *)((int)', r'(')
    if r'(void*)((int)' in res:
      res = res.replace(r'(void*)((int)', r'(')
    if r'(void*)' in res:
      res = res.replace(r'(void*)', r'*(SIM_X86::tensor*)')
    if r'(void *)' in res:
      res = res.replace(r'(void *)', r'*(SIM_X86::tensor*)')
    if r'void*' in res:
      res = res.replace(r'void*', r'SIM_X86::tensor')
    if r'void *' in res:
      res = res.replace(r'void *', r'SIM_X86::tensor ')
    if r'DLCMem' in res:
      pattern = r'\bDLCMem\b'
      res = re.sub(pattern, r'SIM_X86::DLCMem', res)
    if r'DLCTensor' in res:
      pattern = r'\bDLCTensor\b'
      res = re.sub(pattern, r'SIM_X86::DLCTensor', res)
    if r'DLCTensorLite' in res:
      pattern = r'\bDLCTensorLite\b'
      res = re.sub(pattern, r'SIM_X86::DLCTensorLite', res)
    if r'TensorInfo' in res:
      res = res.replace(r'TensorInfo', r'SIM_X86::TensorInfo')

    if r'__attribute__((address_space(2)))' in res:
      res = res.replace(r'__attribute__((address_space(2)))', r'/*__attribute__((address_space(2)))*/')
    if r'Print' in res:
      pattern = r'\bPrint\b'
      res = re.sub(pattern, r'// Print', res)

    if r'#ifndef' in res and r'_H' in res:
      res = res.replace(r'_H', r'_H_X86')
    if r'#define' in res and r'_H' in res:
      res = res.replace(r'_H', r'_H_X86')

    if r'((int *)((int)temp_smem))' in res and file_name == r'max_min.h':
      res = res.replace(r'((int *)((int)temp_smem))', r'temp_smem')

    if r'#include' in res and r'/' in res and file_name != r'libdevice.h' and depth == 2:
      res = res.replace(r'"', r'"../', 1)
    if r'#include "kernel_arg_types.h"' in res:
      res = "// " + res
    if r'#include "typehint.h"' in res:
      res = "// " + res
    if r'#pragma' in res:
      res = res.replace(r'#', r'// #')

    if r'store128_128_ex' in res and file_name == r'permute.h':
      res = res.replace(r'store128_128_ex', r'store128_128_ex_permute')

    F.append(res)

  return F

# 将 .h 清理后 写入 x86 的目录
def replace_file(src_file, dst_file, suf, depth):
  # 读文件
  with open(src_file, 'r') as F:
    lines = F.readlines()

    lines = clear_file(lines, suf, os.path.basename(src_file), depth)

    # 写文件
    with open(dst_file, 'w') as F_out:
      F_out.writelines(lines)

# 同步 dlc_kernels 文件夹
def sync(src_root, dst_root):
  depth_1_files = []

  # 遍历源目录的每一个文件和文件夹
  for dirpath, dirnames, filenames in os.walk(src_root):
    # 为每个文件夹创建相应的目标目录
    relative_path = os.path.relpath(dirpath, src_root)
    dst_dir = os.path.join(dst_root, relative_path)

    # 如果不存在就创建
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)

    suf = "../"
    depth = 1
    if relative_path == ".":
      suf = ""
    else:
      depth = 2

    # 复制以 .h 结尾的文件
    for filename in filenames:
      if filename == "typehint.h":
        continue

      if filename.endswith('.h') or filename.endswith('.hpp'):
        src_file = os.path.join(dirpath, filename)
        dst_file = os.path.join(dst_dir, filename)

        replace_file(src_file, dst_file, suf, depth)

        if depth == 1:
          depth_1_files.append('#include "dlc_kernels/' + filename + '"')

  # depth_1_files.sort()
  # for k in depth_1_files:
  #   print(k)

# 开始同步所有库
sync(DLC_Custom_Kernel_PATH, X86_PATH)
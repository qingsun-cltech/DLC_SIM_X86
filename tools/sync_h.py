import math
import os
import re

DLC_Custom_Kernel_PATH = r'/home/qingsun/DLC_Custom_Kernel/dlc_kernels'
X86_PATH = r'/home/qingsun/x86/dlc_kernels'

# 清理 .h
def clear_file(file, suf, file_name):
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
    if r'void*' in res:
      res = res.replace(r'void*', r'SIM_X86::tensor')
    if r'void *' in res:
      res = res.replace(r'void *', r'SIM_X86::tensor ')
    if r'DLCMem' in res:
      res = res.replace(r'DLCMem', r'SIM_X86::DLCMem')
    if r'TensorInfo' in res:
      res = res.replace(r'TensorInfo', r'SIM_X86::TensorInfo')

    if r'__attribute__((address_space(2)))' in res:
      res = res.replace(r'__attribute__((address_space(2)))', r'/*__attribute__((address_space(2)))*/')

    if r'#ifndef' in res and r'_H' in res:
      res = res.replace(r'_H', r'_H_X86')
    if r'#define' in res and r'_H' in res:
      res = res.replace(r'_H', r'_H_X86')

    if r'#include' in res and r'/' in res and file_name != r'libdevice.h':
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
def replace_file(src_file, dst_file, suf):
  # 读文件
  with open(src_file, 'r') as F:
    lines = F.readlines()
    
    lines = clear_file(lines, suf, os.path.basename(src_file))
      
    # 写文件
    with open(dst_file, 'w') as F_out:
      F_out.writelines(lines)

# 同步 dlc_kernels 文件夹
def sync(src_root, dst_root):
  # 遍历源目录的每一个文件和文件夹
  for dirpath, dirnames, filenames in os.walk(src_root):
    # 为每个文件夹创建相应的目标目录
    relative_path = os.path.relpath(dirpath, src_root)
    dst_dir = os.path.join(dst_root, relative_path)

    # 如果不存在就创建
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)
      
    suf = "../"
    if relative_path == ".":
      suf = ""

    # 复制以 .h 结尾的文件
    for filename in filenames:
      if filename == "typehint.h":
        continue

      if filename.endswith('.h') or filename.endswith('.hpp'):
        src_file = os.path.join(dirpath, filename)
        dst_file = os.path.join(dst_dir, filename)
        
        replace_file(src_file, dst_file, suf)

# 开始同步所有库
sync(DLC_Custom_Kernel_PATH, X86_PATH)
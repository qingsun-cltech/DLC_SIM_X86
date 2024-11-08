import math
import os
import re

kernel_name = [r'matmul_t_pingpong.c', r'matmul_t_bf16_pingpong.c']

DLC_Custom_Kernel_PATH = r'/home/qingsun/DLC_Custom_Kernel/dlc_kernels'
X86_PATH = r'/home/qingsun/x86'

# 清理 .h
def clear_file(file, file_name):
  F = []
  F.append(f'#include "x86.h"\n')
  F.append('\n')

  for line in file:
    res = line

    if r'tensor' in res:
      pattern = r'\btensor\b'
      res = re.sub(pattern, r'SIM_X86::tensor', res)
    if r'(SIM_X86::tensor)' in res:
      res = res.replace(r'(SIM_X86::tensor)', r'*(SIM_X86::tensor*)')
    if r'main' in res:
      pattern = r'\bmain\b'
      res = re.sub(pattern, r'main_x86', res)
    if r'void*' in res:
      res = res.replace(r'void*', r'SIM_X86::tensor')
    if r'void *' in res:
      res = res.replace(r'void *', r'SIM_X86::tensor ')

    if r'DLCMem' in res:
      res = res.replace(r'DLCMem', r'SIM_X86::DLCMem')
    if r'DLCTensor' in res:
      res = res.replace(r'DLCTensor', r'SIM_X86::DLCTensor')
    if r'DLCScalar' in res:
      res = res.replace(r'DLCScalar', r'SIM_X86::DLCScalar')
    if r'TensorInfo' in res:
      res = res.replace(r'TensorInfo', r'SIM_X86::TensorInfo')

    if r'__attribute__((address_space(2)))' in res:
      res = res.replace(r'__attribute__((address_space(2)))', r'/*__attribute__((address_space(2)))*/')

    if r'#include' in res:
      res = res.replace(r'"', r'"dlc_kernels/', 1)
    if r'typehint.h' in res:
      res = "// " + res
    if r'kernel_arg_types.h' in res:
      res = "// " + res
    if r'#pragma' in res:
      res = res.replace(r'#', r'// #')

    F.append(res)

  return F

# 将 .h 清理后 写入 x86 的目录
def sync(src_file, dst_file):
  # 读文件
  with open(src_file, 'r') as F:
    lines = F.readlines()
    
    lines = clear_file(lines, os.path.basename(src_file))
      
    # 写文件
    with open(dst_file, 'w') as F_out:
      F_out.writelines(lines)

for name in kernel_name:
  sync(os.path.join(DLC_Custom_Kernel_PATH, name),
      os.path.join(X86_PATH, name))
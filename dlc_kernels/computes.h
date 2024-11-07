#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "chunk.h"
#include "convert_element_type_h.h"
#include "align.h"

typedef int8_128 (*transform_fn1_t)(float8_128, float8_128);
typedef int8_128 (*transform_fn1_t_i)(int8_128, int8_128);

inline void HBMtoVMem(SIM_X86::tensor hbm_src, SIM_X86::tensor vmem_dst, int src_addr, int dst_addr, int length) {
    int handle = dlc_dma(tensor_slice(hbm_src, src_addr / 32), HBM, tensor_slice(vmem_dst, dst_addr / 32),
                         VMEM, length, 128, 128, 128, 7);
    dlc_sync(handle);
}

inline void VMEMtoHBM(SIM_X86::tensor vmem_src, SIM_X86::tensor hbm_dst, int src_addr, int dst_addr, int length) {
    int handle = dlc_dma(tensor_slice(vmem_src, src_addr / 32), VMEM, tensor_slice(hbm_dst, dst_addr / 32),
                         HBM, length, 128, 128, 128, 7);
    dlc_sync(handle);
}

inline void HBMtoVMemNoSync(SIM_X86::tensor hbm_src, SIM_X86::tensor vmem_dst, int src_addr, int dst_addr, int length,
                            int *syncFlag) {
    *syncFlag = dlc_dma(tensor_slice(hbm_src, src_addr / 32), HBM, tensor_slice(vmem_dst, dst_addr / 32),
                        VMEM, length, 128, 128, 128, 7);
}

inline void VMEMtoHBMNoSync(SIM_X86::tensor vmem_src, SIM_X86::tensor hbm_dst, int src_addr, int dst_addr, int length,
                            int *syncFlag) {
    *syncFlag = dlc_dma(tensor_slice(vmem_src, src_addr / 32), VMEM, tensor_slice(hbm_dst, dst_addr / 32),
                        HBM, length, 128, 128, 128, 7);
}

inline void innerLoop(SIM_X86::tensor vmem_tensor, int VMEMsize, float8_128 other_v, transform_fn1_t fn) {
    int i = 0;
    for (; i + 1024 <= VMEMsize; i += 1024) {
        float8_128 x = v_f32_ld_tnsr_b(i / 32, vmem_tensor);
        v_st_generic(i / 32, vmem_tensor, 1, 255, fn(x, other_v));
    }
    if (i < VMEMsize) {
        int size = VMEMsize - i;
        int mask = pre_exp2(size / 128);
        float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, vmem_tensor, 1, mask);
        v_st_generic(i / 32, vmem_tensor, 1, mask, fn(x, other_v));
    }
}

inline void innerLoop_i(SIM_X86::tensor vmem_tensor, int VMEMsize, int8_128 other_v, transform_fn1_t_i fn) {
    int i = 0;
    for (; i + 1024 <= VMEMsize; i += 1024) {
        int8_128 x = v_i32_ld_tnsr(i / 32, vmem_tensor, 1, 255);
        v_st_generic(i / 32, vmem_tensor, 1, 255, fn(x, other_v));
    }
    if (i < VMEMsize) {
        int size = VMEMsize - i;
        int mask = pre_exp2(size / 128);
        int8_128 x = v_i32_ld_tnsr(i / 32, vmem_tensor, 1, mask);
        v_st_generic(i / 32, vmem_tensor, 1, mask, fn(x, other_v));
    }
}

inline void dma_pipeline_lanhu_uint8(SIM_X86::tensor hbm_in, SIM_X86::tensor hbm_out, SIM_X86::tensor vmem_tensor, int len, int len_512,
                                     int _VMEMsize, int _VMEMsize_512, int height_vmem, float other, int dim0,
                                     int dim0_128, int dim0_512, transform_fn1_t fn) {

    float8_128 other_v = v_u32_move_f(other);

    int l = 0, j = 0;
    // handle the data till (len / _VMEMsize * _VMEMsize), since div is expansive, we use an extra addition
    
    int half_VMEMsize = height_vmem / 2 * dim0_128;
    int half_VMEMsize_512 = height_vmem / 2 * dim0_512;
    if (height_vmem == 1) {
        half_VMEMsize = height_vmem * dim0_128;
        half_VMEMsize_512 = height_vmem  * dim0_512;
    }
    // 需要判断当分成两半之后，ali到512的数据量不能超过一半的大小，超过了，只能按照al 512的最大的来,
    // 并且也是需要对齐到512的 前半部分需要，VMEMsize和VMEMsize_512一样

    int flagIn0 = DONE, flagIn1 = DONE, flagOut0 = DONE, flagOut1 = DONE;
    bool firstForLoop = l + _VMEMsize <= len;

    // we pipeline the dma
    if (firstForLoop) {
        HBMtoVMemNoSync(hbm_in, vmem_tensor, l, 0, half_VMEMsize, &flagIn0);
    }
    int h = soft_sdiv(half_VMEMsize_512 / 4, dim0_512 / 4);
    for (; (l + _VMEMsize <= len) && (_VMEMsize > 4096); l += _VMEMsize, j += _VMEMsize_512) {
        // if it's the first time we are in the for loop, we clear done bit,
        // other wise we wait for the previous dma to finish
        dlc_sync(flagOut1);
        HBMtoVMemNoSync(hbm_in, vmem_tensor, l + half_VMEMsize, half_VMEMsize, half_VMEMsize, &flagIn1);
        dlc_sync(flagIn0);
        // the input for this loop is guard by flagIn0, output is guard by flagOut0
        innerLoop(vmem_tensor, half_VMEMsize, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, half_VMEMsize_512 / 4, h, dim0);
        VMEMtoHBMNoSync(vmem_tensor, hbm_out, 0, j / 4, half_VMEMsize_512 / 4, &flagOut0);
        dlc_sync(flagOut0);
        HBMtoVMemNoSync(hbm_in, vmem_tensor, l + 2 * half_VMEMsize, 0, half_VMEMsize, &flagIn0);
        dlc_sync(flagIn1);
        // the input for this loop is guard by flagIn1, output is guard by flagOut1
        innerLoop(tensor_slice(vmem_tensor, half_VMEMsize / 32), half_VMEMsize, other_v, fn);
        i32Touint8_h(tensor_slice(vmem_tensor, half_VMEMsize / 32),
                     tensor_slice(vmem_tensor, half_VMEMsize / 32), half_VMEMsize_512 / 4, h, dim0);
        VMEMtoHBMNoSync(vmem_tensor, hbm_out, half_VMEMsize, j / 4 + half_VMEMsize_512 / 4,
                        half_VMEMsize_512 / 4, &flagOut1);
    }
    if (firstForLoop) {
        dlc_sync(flagOut1);
    }
    // handle the remaining data
    if (l < len) {
        int VMEMsize_rem = len - l;
        int VMEMsize_512_rem = len_512 - j;
        int h_ = soft_sdiv(VMEMsize_512_rem / 4, dim0_512 / 4);
        HBMtoVMem(hbm_in, vmem_tensor, l, 0, VMEMsize_rem);
        innerLoop(vmem_tensor, VMEMsize_rem, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, VMEMsize_512_rem / 4, h_, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, VMEMsize_512_rem / 4);
    }
}

inline void dma_pipeline_lanhu_uint8_i32(SIM_X86::tensor hbm_in, SIM_X86::tensor hbm_out, SIM_X86::tensor vmem_tensor, int len, int len_512,
                                     int _VMEMsize, int _VMEMsize_512, int height_vmem, int other, int dim0,
                                     int dim0_128, int dim0_512, transform_fn1_t_i fn) {

    int8_128 other_v = v_u32_move_i(other);

    int l = 0, j = 0;
    // handle the data till (len / _VMEMsize * _VMEMsize), since div is expansive, we use an extra addition
    
    int half_VMEMsize = height_vmem / 2 * dim0_128;
    int half_VMEMsize_512 = height_vmem / 2 * dim0_512;
    if (height_vmem == 1) {
        half_VMEMsize = height_vmem * dim0_128;
        half_VMEMsize_512 = height_vmem  * dim0_512;
    }
    // 需要判断当分成两半之后，ali到512的数据量不能超过一半的大小，超过了，只能按照al 512的最大的来,
    // 并且也是需要对齐到512的 前半部分需要，VMEMsize和VMEMsize_512一样

    int flagIn0 = DONE, flagIn1 = DONE, flagOut0 = DONE, flagOut1 = DONE;
    bool firstForLoop = l + _VMEMsize <= len;

    // we pipeline the dma
    if (firstForLoop) {
        HBMtoVMemNoSync(hbm_in, vmem_tensor, l, 0, half_VMEMsize, &flagIn0);
    }
    int h = soft_sdiv(half_VMEMsize_512 / 4, dim0_512 / 4);
    for (; (l + _VMEMsize <= len) && (_VMEMsize > 4096); l += _VMEMsize, j += _VMEMsize_512) {
        // if it's the first time we are in the for loop, we clear done bit,
        // other wise we wait for the previous dma to finish
        dlc_sync(flagOut1);
        HBMtoVMemNoSync(hbm_in, vmem_tensor, l + half_VMEMsize, half_VMEMsize, half_VMEMsize, &flagIn1);
        dlc_sync(flagIn0);
        // the input for this loop is guard by flagIn0, output is guard by flagOut0
        innerLoop_i(vmem_tensor, half_VMEMsize, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, half_VMEMsize_512 / 4, h, dim0);
        VMEMtoHBMNoSync(vmem_tensor, hbm_out, 0, j / 4, half_VMEMsize_512 / 4, &flagOut0);
        dlc_sync(flagOut0);
        HBMtoVMemNoSync(hbm_in, vmem_tensor, l + 2 * half_VMEMsize, 0, half_VMEMsize, &flagIn0);
        dlc_sync(flagIn1);
        // the input for this loop is guard by flagIn1, output is guard by flagOut1
        innerLoop_i(tensor_slice(vmem_tensor, half_VMEMsize / 32), half_VMEMsize, other_v, fn);
        i32Touint8_h(tensor_slice(vmem_tensor, half_VMEMsize / 32),
                     tensor_slice(vmem_tensor, half_VMEMsize / 32), half_VMEMsize_512 / 4, h, dim0);
        VMEMtoHBMNoSync(vmem_tensor, hbm_out, half_VMEMsize, j / 4 + half_VMEMsize_512 / 4,
                        half_VMEMsize_512 / 4, &flagOut1);
    }
    if (firstForLoop) {
        dlc_sync(flagOut1);
    }
    // handle the remaining data
    if (l < len) {
        int VMEMsize_rem = len - l;
        int VMEMsize_512_rem = len_512 - j;
        int h_ = soft_sdiv(VMEMsize_512_rem / 4, dim0_512 / 4);
        HBMtoVMem(hbm_in, vmem_tensor, l, 0, VMEMsize_rem);
        innerLoop_i(vmem_tensor, VMEMsize_rem, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, VMEMsize_512_rem / 4, h_, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, VMEMsize_512_rem / 4);
    }
}

inline void __bf16ToF32_256_compares(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, float8_128 other_v,
                                     transform_fn1_t fn) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_st_msk(i / 32, mem, 1, 255);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);

            int8_128 y1 = fn(bfloat16_to_float(x1), other_v);
            int8_128 y2 = fn(bfloat16_to_float(x2), other_v);

            store8_128_stride_i(j / 32, 2, dst, y1);
            store8_128_stride_i((j + 128) / 32, 2, dst, y2);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);

        int8_128 y1 = fn(bfloat16_to_float(x1), other_v);
        int8_128 y2 = fn(bfloat16_to_float(x2), other_v);

        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, y1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask, dst, y2);
    }
}

inline void __bf16ToF32_128_compare(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, float8_128 other_v, transform_fn1_t fn) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len - 1024, j = len * 2 - 2048; i >= 0; i -= 1024, j -= 2048) {
            float8_128 x = v_f32_ld_tnsr_b(i / 32, mem);
            short8_128 x2 = unpack_16b(__$S(x), 1);
            short8_128 x1 = unpack_16b(__$S(x), 0);

            int8_128 y1 = fn(bfloat16_to_float(x1), other_v);
            int8_128 y2 = fn(bfloat16_to_float(x2), other_v);

            store8_128_stride_i(j / 32, 2, dst, y1);
            if (i == (len - 1024)) {
                store8_128_stride_stmk_i((j + 128) / 32, 2, dst, y2, 127);
            } else {
                store8_128_stride_i((j + 128) / 32, 2, dst, y2);
            }
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int ldst_mask2 = pre_exp2((len128 / 128) - 1);
        if (len >= 1024) {
            ldst_mask2 = pre_exp2(len128 / 128);
        }
        float8_128 x = load8_128_stride_with_ldmask(0, 1, ldst_mask, mem);
        short8_128 x2 = unpack_16b(__$S(x), 1);
        short8_128 x1 = unpack_16b(__$S(x), 0);

        int8_128 y1 = fn(bfloat16_to_float(x1), other_v);
        int8_128 y2 = fn(bfloat16_to_float(x2), other_v);
        
        store8_128_stride_with_stmask_i(0, 2, ldst_mask, dst, y1);
        store8_128_stride_with_stmask_i(4, 2, ldst_mask2, dst, y2);
    }
}

inline void innerLoop_bf16ToF32_compares(SIM_X86::tensor mem, int len, int h, int d0, float8_128 other_v,
                                         transform_fn1_t fn) {
    d0 = ALIGN128(d0);
    int d0k128 = d0 / 128;
    if (d0k128 % 2 == 0) {
        __bf16ToF32_256_compares(mem, mem, len, other_v, fn);
    } else {
        int bd0 = ALIGN256(d0);
        for (int i = h - 1; i >= 0; i--) {
            __bf16ToF32_128_compare(mem + i * bd0 / 2 / 32, mem + i * d0 / 32, bd0 / 2, other_v, fn);
        }
    }
}

inline void outside_loop_compares(SIM_X86::tensor hbm_in, SIM_X86::tensor vmem_tensor, SIM_X86::tensor hbm_out, float8_128 other_v,
                                  int len_256, int len_512, int vmemsize, int _VMEMsize, int _VMEMsize_512,
                                  int dim0_256, int dim0_512, int dim0, transform_fn1_t fn) {
    int l = 0, j = 0;
    int h = soft_sdiv(_VMEMsize / 2, dim0_256 / 2);
    int h1 = soft_sdiv(_VMEMsize_512 / 4, dim0_512 / 4);
    for (; l + _VMEMsize < len_256; l += _VMEMsize, j += _VMEMsize_512) {
        HBMtoVMem(hbm_in, vmem_tensor, l / 2, 0, vmemsize);
        innerLoop_bf16ToF32_compares(vmem_tensor, _VMEMsize / 2, h, dim0, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, _VMEMsize_512 / 4, h1, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, _VMEMsize_512 / 4);
    }
    if (l < len_256) {
        int vmemsize_ = len_256 - l;
        int VMEMsize_512 = len_512 - j;
        int h_ = soft_sdiv(vmemsize_ / 2, dim0_256 / 2);
        int h1_ = soft_sdiv(VMEMsize_512 / 4, dim0_512 / 4);
        HBMtoVMem(hbm_in, vmem_tensor, l / 2, 0, vmemsize_ / 2);
        innerLoop_bf16ToF32_compares(vmem_tensor, vmemsize_ / 2, h_, dim0, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, VMEMsize_512 / 4, h1_, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, VMEMsize_512 / 4);
    }
}

inline void compare_int32Tobool(SIM_X86::tensor hbm_in, SIM_X86::tensor hbm_out, SIM_X86::tensor vmem_tensor, int len, int len_512,
                                int _VMEMsize, int _VMEMsize_512, int h, int other, int dim0, int dim0_512,
                                transform_fn1_t_i fn) {
    int8_128 other_v = v_u32_move_i(other);

    int l = 0, j = 0;
    for (; l + _VMEMsize < len; l += _VMEMsize, j += _VMEMsize_512) {
        HBMtoVMem(hbm_in, vmem_tensor, l, 0, _VMEMsize);
        // the input for this loop is guard by flagIn0, output is guard by flagOut0
        innerLoop_i(vmem_tensor, _VMEMsize, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, _VMEMsize_512 / 4, h, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, _VMEMsize_512 / 4);
    }
    // handle the remaining data
    if (l < len) {
        int VMEMsize_rem = len - l;
        int VMEMsize_512_rem = len_512 - j;
        int h_ = soft_sdiv(VMEMsize_512_rem / 4, dim0_512 / 4);
        HBMtoVMem(hbm_in, vmem_tensor, l, 0, VMEMsize_rem);
        innerLoop_i(vmem_tensor, VMEMsize_rem, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, VMEMsize_512_rem / 4, h_, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, VMEMsize_512_rem / 4);
    }
}

inline void compare_fp32Tobool(SIM_X86::tensor hbm_in, SIM_X86::tensor hbm_out, SIM_X86::tensor vmem_tensor, int len, int len_512,
                                int _VMEMsize, int _VMEMsize_512, int h, float other, int dim0, int dim0_512,
                                transform_fn1_t fn) {
    float8_128 other_v = v_u32_move_f(other);

    int l = 0, j = 0;
    for (; l + _VMEMsize < len; l += _VMEMsize, j += _VMEMsize_512) {
        HBMtoVMem(hbm_in, vmem_tensor, l, 0, _VMEMsize);
        // the input for this loop is guard by flagIn0, output is guard by flagOut0
        innerLoop(vmem_tensor, _VMEMsize, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, _VMEMsize_512 / 4, h, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, _VMEMsize_512 / 4);
    }
    // handle the remaining data
    if (l < len) {
        int VMEMsize_rem = len - l;
        int VMEMsize_512_rem = len_512 - j;
        int h_ = soft_sdiv(VMEMsize_512_rem / 4, dim0_512 / 4);
        HBMtoVMem(hbm_in, vmem_tensor, l, 0, VMEMsize_rem);
        innerLoop(vmem_tensor, VMEMsize_rem, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, VMEMsize_512_rem / 4, h_, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, VMEMsize_512_rem / 4);
    }
}

inline void innerLoop_long(SIM_X86::tensor vmem_tensor, int VMEMsize, int8_128 other_v, transform_fn1_t_i fn) {
    int i = 0, j = 0;
    for (; i + 2048 <= VMEMsize; i += 2048, j += 1024) {
        int8_128 x = load8_128_stride2_i(i / 32, 2, vmem_tensor);
        v_st_generic(j / 32, vmem_tensor, 1, 255, fn(x, other_v));
    }
    if (i < VMEMsize) {
        int size = VMEMsize - i;
        int mask = pre_exp2(size / 128);
        int8_128 x = load8_128_stride2_with_ldmask_i(i / 32, 2, mask, vmem_tensor);
        v_st_generic(j / 32, vmem_tensor, 1, mask, fn(x, other_v));
    }
}

inline void compare_longTobool(SIM_X86::tensor hbm_in, SIM_X86::tensor hbm_out, SIM_X86::tensor vmem_tensor, int len, int len_512,
                               int _VMEMsize, int _VMEMsize_512, int h, int other, int dim0, int AlignTolongdim0,
                               int dim0_512, transform_fn1_t_i fn) {
    int8_128 other_v = v_u32_move_i(other);
    int l = 0, j = 0;
    for (; l + _VMEMsize < len; l += _VMEMsize, j += _VMEMsize_512) {
        HBMtoVMem(hbm_in, vmem_tensor, l, 0, _VMEMsize);
        // the input for this loop is guard by flagIn0, output is guard by flagOut0
        innerLoop_long(vmem_tensor, _VMEMsize, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, _VMEMsize_512 / 4, h, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, _VMEMsize_512 / 4);
    }
    // handle the remaining data
    if (l < len) {
        int VMEMsize_rem = len - l;
        int VMEMsize_512_rem = len_512 - j;
        int h_ = soft_sdiv(VMEMsize_512_rem / 4, dim0_512 / 4);
        HBMtoVMem(hbm_in, vmem_tensor, l, 0, VMEMsize_rem);
        innerLoop_long(vmem_tensor, VMEMsize_rem, other_v, fn);
        i32Touint8_h(vmem_tensor, vmem_tensor, VMEMsize_512_rem / 4, h_, dim0);
        VMEMtoHBM(vmem_tensor, hbm_out, 0, j / 4, VMEMsize_512_rem / 4);
    }
}
#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"


#include "align.h"
#include "chunk.h"

#include "math.h"

typedef int8_128 (*transform_fn1_t_i)(int8_128);
typedef int8_128 (*transform_fn_2_int)(int8_128, int8_128);

inline int8_128 __dlc_char_as_int(char8_128 a) {
    int8_128 result0;
    asm volatile("{ V0@(pr0)       vr10 = mov.u32 %[input]; }" : : [input] "x"(a) : "vr10");
    asm volatile("{"
                 "V1@(pr0)	vr10 = mov.u32 vr10;"
                 "}"
                 :
                 :
                 : "vr10");

    asm volatile("{V0@(pr0)        %[res] = mov.u32 vr10;}" : [res] "=x"(result0) : : "vr10");

    return result0;
}

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

// and, or, xor
inline void bitwise_2input_tensor_innerloop_int32(SIM_X86::tensor input_vmem, SIM_X86::tensor other_vmem, int VMEMsize,
                                                  transform_fn_2_int fn) {
    int vs = 0;
    for (; vs + 1024 < VMEMsize; vs += 1024) {
        int8_128 x0 = v_i32_ld_tnsr(vs / 32, input_vmem, 1, 255);
        int8_128 x1 = v_i32_ld_tnsr(vs / 32, other_vmem, 1, 255);
        v_st_generic(vs / 32, input_vmem, 1, 255, fn(x0, x1));
    }
    if (vs < VMEMsize) {
        int len = min(VMEMsize - vs, 1024);
        int ldst_vmask = pre_exp2(len / 128);
        int8_128 x0 = v_i32_ld_tnsr(vs / 32, input_vmem, 1, ldst_vmask);
        int8_128 x1 = v_i32_ld_tnsr(vs / 32, other_vmem, 1, ldst_vmask);
        v_st_generic(vs / 32, input_vmem, 1, ldst_vmask, fn(x0, x1));
    }
}

const int D_HBM = 1;
const int D_VMEM = 2;

inline void bitwise_2input_tensor_outloop_int32(SIM_X86::tensor input, SIM_X86::tensor other, SIM_X86::tensor input_vmem,
                                                SIM_X86::tensor other_vmem, SIM_X86::tensor output_hbm, int len, int _VMEMsize,
                                                transform_fn_2_int fn) {
    int l = 0;
    for (; l + _VMEMsize < len; l += _VMEMsize) {
        int handle1 = dlc_dma(tensor_slice(input, l / 32), D_HBM, tensor_slice(input_vmem, 0 / 32), D_VMEM,
                              _VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle1);
        int handle2 = dlc_dma(tensor_slice(other, l / 32), D_HBM, tensor_slice(other_vmem, 0 / 32), D_VMEM,
                              _VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle2);

        bitwise_2input_tensor_innerloop_int32(input_vmem, other_vmem, _VMEMsize, fn);

        int handle3 = dlc_dma(tensor_slice(input_vmem, 0 / 32), D_VMEM, tensor_slice(output_hbm, l / 32),
                              D_HBM, _VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle3);
    }
    if (l < len) {
        int VMEMsize = len - l;
        int handle1 = dlc_dma(tensor_slice(input, l / 32), D_HBM, tensor_slice(input_vmem, 0 / 32), D_VMEM,
                              VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle1);
        int handle2 = dlc_dma(tensor_slice(other, l / 32), D_HBM, tensor_slice(other_vmem, 0 / 32), D_VMEM,
                              VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle2);

        bitwise_2input_tensor_innerloop_int32(input_vmem, other_vmem, VMEMsize, fn);

        int handle3 = dlc_dma(tensor_slice(input_vmem, 0 / 32), D_VMEM, tensor_slice(output_hbm, l / 32),
                              D_HBM, VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle3);
    }
}

inline void bitwise_2input_tensor_innerloop_bool(SIM_X86::tensor input_vmem, SIM_X86::tensor other_vmem, int VMEMsize,
                                                 transform_fn_2_int fn) {
    int vs = 0;
    for (; vs + 1024 < VMEMsize; vs += 1024) {
        int8_128 x0 = v_i32_ld_tnsr(vs / 32, input_vmem, 1, 255);
        int8_128 x1 = v_i32_ld_tnsr(vs / 32, other_vmem, 1, 255);

        short8_128 y1 = unpack_16b(x0, 1);
        short8_128 y0 = unpack_16b(x0, 0);

        short8_128 z1 = unpack_16b(x1, 1);
        short8_128 z0 = unpack_16b(x1, 0);

        char8_128 y2 = unpack_8b(y0, 0);
        char8_128 y3 = unpack_8b(y0, 1);
        char8_128 y4 = unpack_8b(y1, 0);
        char8_128 y5 = unpack_8b(y1, 1);

        char8_128 z2 = unpack_8b(z0, 0);
        char8_128 z3 = unpack_8b(z0, 1);
        char8_128 z4 = unpack_8b(z1, 0);
        char8_128 z5 = unpack_8b(z1, 1);

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(fn(__dlc_char_as_int(y5), __dlc_char_as_int(z5)), 24),
                                          v_u32_shl(fn(__dlc_char_as_int(y4), __dlc_char_as_int(z4)), 16)),
                                v_u32_shl(fn(__dlc_char_as_int(y3), __dlc_char_as_int(z3)), 8)),
                      fn(__dlc_char_as_int(y2), __dlc_char_as_int(z2)));

        v_st_generic(vs / 32, input_vmem, 1, 255, result);
    }
    if (vs < VMEMsize) {
        int len = min(VMEMsize - vs, 1024);
        int ldst_vmask = pre_exp2(len / 128);
        int8_128 x0 = v_i32_ld_tnsr(vs / 32, input_vmem, 1, ldst_vmask);
        int8_128 x1 = v_i32_ld_tnsr(vs / 32, other_vmem, 1, ldst_vmask);

        short8_128 y1 = unpack_16b(x0, 1);
        short8_128 y0 = unpack_16b(x0, 0);

        short8_128 z1 = unpack_16b(x1, 1);
        short8_128 z0 = unpack_16b(x1, 0);

        char8_128 y2 = unpack_8b(y0, 0);
        char8_128 y3 = unpack_8b(y0, 1);
        char8_128 y4 = unpack_8b(y1, 0);
        char8_128 y5 = unpack_8b(y1, 1);

        char8_128 z2 = unpack_8b(z0, 0);
        char8_128 z3 = unpack_8b(z0, 1);
        char8_128 z4 = unpack_8b(z1, 0);
        char8_128 z5 = unpack_8b(z1, 1);

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(fn(__dlc_char_as_int(y5), __dlc_char_as_int(z5)), 24),
                                          v_u32_shl(fn(__dlc_char_as_int(y4), __dlc_char_as_int(z4)), 16)),
                                v_u32_shl(fn(__dlc_char_as_int(y3), __dlc_char_as_int(z3)), 8)),
                      fn(__dlc_char_as_int(y2), __dlc_char_as_int(z2)));

        v_st_generic(vs / 32, input_vmem, 1, ldst_vmask, result);
    }
}

inline void bitwise_2input_tensor_outloop_bool(SIM_X86::tensor input, SIM_X86::tensor other, SIM_X86::tensor input_vmem,
                                               SIM_X86::tensor other_vmem, SIM_X86::tensor output_hbm, int len, int _VMEMsize,
                                               transform_fn_2_int fn) {
    int l = 0;
    for (; l + _VMEMsize < len; l += _VMEMsize) {
        int handle1 = dlc_dma(tensor_slice(input, l / 32), D_HBM, tensor_slice(input_vmem, 0 / 32), D_VMEM,
                              _VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle1);
        int handle2 = dlc_dma(tensor_slice(other, l / 32), D_HBM, tensor_slice(other_vmem, 0 / 32), D_VMEM,
                              _VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle2);

        bitwise_2input_tensor_innerloop_bool(input_vmem, other_vmem, _VMEMsize, fn);

        int handle3 = dlc_dma(tensor_slice(input_vmem, 0 / 32), D_VMEM, tensor_slice(output_hbm, l / 32),
                              D_HBM, _VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle3);
    }
    if (l < len) {
        int VMEMsize = len - l;
        int handle1 = dlc_dma(tensor_slice(input, l / 32), D_HBM, tensor_slice(input_vmem, 0 / 32), D_VMEM,
                              VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle1);
        int handle2 = dlc_dma(tensor_slice(other, l / 32), D_HBM, tensor_slice(other_vmem, 0 / 32), D_VMEM,
                              VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle2);

        bitwise_2input_tensor_innerloop_bool(input_vmem, other_vmem, VMEMsize, fn);

        int handle3 = dlc_dma(tensor_slice(input_vmem, 0 / 32), D_VMEM, tensor_slice(output_hbm, l / 32),
                              D_HBM, VMEMsize, 128, 128, 128, 7);
        dlc_sync(handle3);
    }
}

inline void bitwise_boolTobool_512(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0, transform_fn1_t_i fn) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len1024 - 1024; i >= 0; i -= 1024) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);

            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);

            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y1 = __dlc_char_as_int(_x0);
            int8_128 y2 = __dlc_char_as_int(_x1);
            int8_128 y3 = __dlc_char_as_int(_x2);
            int8_128 y4 = __dlc_char_as_int(_x3);

            y1 = fn(y1);
            y2 = fn(y2);
            y3 = fn(y3);
            y4 = fn(y4);

            y1 = v_s32_sel(v_s32_cmp(EQ, y1, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y2 = v_s32_sel(v_s32_cmp(EQ, y2, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y3 = v_s32_sel(v_s32_cmp(EQ, y3, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y4 = v_s32_sel(v_s32_cmp(EQ, y4, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));

            int8_128 result =
                v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

            v_st_generic(i / 32, dst, 1, 255, result);
        }
    }
    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y1 = __dlc_char_as_int(_x0);
        int8_128 y2 = __dlc_char_as_int(_x1);
        int8_128 y3 = __dlc_char_as_int(_x2);
        int8_128 y4 = __dlc_char_as_int(_x3);

        y1 = fn(y1);
        y2 = fn(y2);
        y3 = fn(y3);
        y4 = fn(y4);

        y1 = v_s32_sel(v_s32_cmp(EQ, y1, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y2 = v_s32_sel(v_s32_cmp(EQ, y2, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y3 = v_s32_sel(v_s32_cmp(EQ, y3, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y4 = v_s32_sel(v_s32_cmp(EQ, y4, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(len1024 / 32, dst, 1, ldst_mask, result);
    }
}

inline void bitwise_boolTobool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int dim0, transform_fn1_t_i fn) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len1024 - 1024; i >= 0; i -= 1024) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);
            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);
            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y1 = __dlc_char_as_int(_x0);
            int8_128 y2 = __dlc_char_as_int(_x1);
            int8_128 y3 = __dlc_char_as_int(_x2);
            int8_128 y4 = __dlc_char_as_int(_x3);

            y1 = fn(y1);
            y2 = fn(y2);
            y3 = fn(y3);
            y4 = fn(y4);

            y1 = v_s32_sel(v_s32_cmp(EQ, y1, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y2 = v_s32_sel(v_s32_cmp(EQ, y2, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y3 = v_s32_sel(v_s32_cmp(EQ, y3, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y4 = v_s32_sel(v_s32_cmp(EQ, y4, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));

            int8_128 result =
                v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

            v_st_generic(i / 32, dst, 1, 255, result);
        }
    }

    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y1 = __dlc_char_as_int(_x0);
        int8_128 y2 = __dlc_char_as_int(_x1);
        int8_128 y3 = __dlc_char_as_int(_x2);
        int8_128 y4 = __dlc_char_as_int(_x3);

        y1 = fn(y1);
        y2 = fn(y2);
        y3 = fn(y3);
        y4 = fn(y4);

        y1 = v_s32_sel(v_s32_cmp(EQ, y1, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y2 = v_s32_sel(v_s32_cmp(EQ, y2, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y3 = v_s32_sel(v_s32_cmp(EQ, y3, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y4 = v_s32_sel(v_s32_cmp(EQ, y4, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(len1024 / 32, dst, 1, ldst_mask, result);
    }
}

inline void bitwise_bool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int d0, transform_fn1_t_i fn) {
    d0 = ALIGN128(d0);
    bitwise_boolTobool(mem, dst, len, d0, fn);
}

inline void bitwise_bool_outloop(SIM_X86::tensor hbm_in, SIM_X86::tensor vmem_tensor, SIM_X86::tensor hbm_out, int len_512,
                                 int _VMEMsize, int dim0, transform_fn1_t_i fn) {
    int l = 0;
    for (; l + _VMEMsize < len_512; l += _VMEMsize) {
        int handle1 = dlc_dma(tensor_slice(hbm_in, l >> 7), D_HBM, tensor_slice(vmem_tensor, 0 / 32), D_VMEM,
                              _VMEMsize / 4, 128, 128, 128, 7);
        dlc_sync(handle1);

        bitwise_bool(vmem_tensor, vmem_tensor, _VMEMsize / 4, dim0, fn);

        int handle2 = dlc_dma(tensor_slice(vmem_tensor, 0 / 32), D_VMEM, tensor_slice(hbm_out, l >> 7), D_HBM,
                              _VMEMsize / 4, 128, 128, 128, 7);
        dlc_sync(handle2);
    }
    if (l < len_512) {
        int VMEMsize = len_512 - l;
        int handle1 = dlc_dma(tensor_slice(hbm_in, l >> 7), D_HBM, tensor_slice(vmem_tensor, 0 / 32), D_VMEM,
                              VMEMsize / 4, 128, 128, 128, 7);
        dlc_sync(handle1);

        bitwise_bool(vmem_tensor, vmem_tensor, VMEMsize / 4, dim0, fn);

        int handle2 = dlc_dma(tensor_slice(vmem_tensor, 0 / 32), D_VMEM, tensor_slice(hbm_out, l >> 7), D_HBM,
                              VMEMsize / 4, 128, 128, 128, 7);
        dlc_sync(handle2);
    }
}

inline void innerLoop_bool_to_bool_bitwise(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int dim0, transform_fn1_t_i fn) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len1024 - 1024; i >= 0; i -= 1024) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);
            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);
            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y1 = __dlc_char_as_int(_x0);
            int8_128 y2 = __dlc_char_as_int(_x1);
            int8_128 y3 = __dlc_char_as_int(_x2);
            int8_128 y4 = __dlc_char_as_int(_x3);

            y1 = fn(y1);
            y2 = fn(y2);
            y3 = fn(y3);
            y4 = fn(y4);

            y1 = v_s32_sel(v_s32_cmp(EQ, y1, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y2 = v_s32_sel(v_s32_cmp(EQ, y2, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y3 = v_s32_sel(v_s32_cmp(EQ, y3, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
            y4 = v_s32_sel(v_s32_cmp(EQ, y4, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));

            int8_128 result =
                v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

            v_st_generic(i / 32, dst, 1, 255, result);
        }
    }

    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y1 = __dlc_char_as_int(_x0);
        int8_128 y2 = __dlc_char_as_int(_x1);
        int8_128 y3 = __dlc_char_as_int(_x2);
        int8_128 y4 = __dlc_char_as_int(_x3);

        y1 = fn(y1);
        y2 = fn(y2);
        y3 = fn(y3);
        y4 = fn(y4);

        y1 = v_s32_sel(v_s32_cmp(EQ, y1, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y2 = v_s32_sel(v_s32_cmp(EQ, y2, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y3 = v_s32_sel(v_s32_cmp(EQ, y3, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));
        y4 = v_s32_sel(v_s32_cmp(EQ, y4, v_u32_move_i(0)), v_u32_move_i(1), v_u32_move_i(0));

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(len1024 / 32, dst, 1, ldst_mask, result);
    }
}

inline void dma_pipeline_bool_to_bool_bitwise(SIM_X86::tensor hbm_in, SIM_X86::tensor hbm_out, SIM_X86::tensor vmem, int len, int _VMEMsize,
                                      int UseVmemSize, int dim0, int dim0_128, transform_fn1_t_i fn) {

    int l = 0;
    // handle the data till (len / _VMEMsize * _VMEMsize), since div is expansive, we use an extra addition
    int half_VMEMsize = _VMEMsize / 2;
    int flagIn0 = DONE, flagIn1 = DONE, flagOut0 = DONE, flagOut1 = DONE;
    // 当_VMEMsize 小于1024的时候，没有必要做dma pipeline
    bool firstForLoop = (l + _VMEMsize < len) && (_VMEMsize > 1024);

    // we pipeline the dma
    if (firstForLoop) {
        HBMtoVMemNoSync(hbm_in, vmem, l, 0, half_VMEMsize, &flagIn0);
    }
    for (; (l + _VMEMsize < len); l += _VMEMsize) {
        // if it's the first time we are in the for loop, we clear done bit,
        // other wise we wait for the previous dma to finish
        dlc_sync(flagOut1);
        HBMtoVMemNoSync(hbm_in, vmem, l + half_VMEMsize, half_VMEMsize, half_VMEMsize, &flagIn1);
        dlc_sync(flagIn0);
        // the input for this loop is guard by flagIn0, output is guard by flagOut0

        innerLoop_bool_to_bool_bitwise(vmem, vmem, half_VMEMsize, dim0, fn);

//         __attribute__((unused)) volatile float wait_1 = vstore_wait(v_f32_ld_tnsr_st_msk(0, vmem, 1, 1));
        VMEMtoHBMNoSync(vmem, hbm_out, 0, l, half_VMEMsize, &flagOut0);
        dlc_sync(flagOut0);
        HBMtoVMemNoSync(hbm_in, vmem, l + 2 * half_VMEMsize, 0, half_VMEMsize, &flagIn0);
        dlc_sync(flagIn1);

        innerLoop_bool_to_bool_bitwise(vmem + half_VMEMsize / 32, vmem + half_VMEMsize / 32, half_VMEMsize, dim0, fn);
//         __attribute__((unused)) volatile float wait_2 = vstore_wait(v_f32_ld_tnsr_st_msk(0, vmem, 1, 1));
        VMEMtoHBMNoSync(vmem, hbm_out, half_VMEMsize, half_VMEMsize, half_VMEMsize, &flagOut1);
    }
    if (firstForLoop) {
        dlc_sync(flagOut1);
    }
    // handle the remaining data
    if (l < len) {
        int VMEMsize_rem = len - l;
        HBMtoVMem(hbm_in, vmem, l, 0, VMEMsize_rem);
        innerLoop_bool_to_bool_bitwise(vmem, vmem, VMEMsize_rem, dim0, fn);
//         __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vmem, 1, 1));
        VMEMtoHBM(vmem, hbm_out, 0, l, VMEMsize_rem);
    }
}

inline void innerLoop_bool_to_bool(SIM_X86::tensor mem, SIM_X86::tensor dst, int len, int dim0) {
    int len1024 = len & 0xfffffc00;
    int len128 = len & 0x3ff;
    if (len1024 != 0) {
        for (int i = len1024 - 1024; i >= 0; i -= 1024) {
            int8_128 x = v_i32_ld_tnsr(i / 32, mem, 1, 255);
            short8_128 x1 = unpack_16b(x, 1);
            short8_128 x0 = unpack_16b(x, 0);
            char8_128 _x0 = unpack_8b(x0, 0);
            char8_128 _x1 = unpack_8b(x0, 1);
            char8_128 _x2 = unpack_8b(x1, 0);
            char8_128 _x3 = unpack_8b(x1, 1);

            int8_128 y1 = __dlc_char_as_int(_x0);
            int8_128 y2 = __dlc_char_as_int(_x1);
            int8_128 y3 = __dlc_char_as_int(_x2);
            int8_128 y4 = __dlc_char_as_int(_x3);

            y1 = v_s32_sel(v_s32_cmp(EQ, y1, v_u32_move_i(0)), v_u32_move_i(0), v_u32_move_i(1));
            y2 = v_s32_sel(v_s32_cmp(EQ, y2, v_u32_move_i(0)), v_u32_move_i(0), v_u32_move_i(1));
            y3 = v_s32_sel(v_s32_cmp(EQ, y3, v_u32_move_i(0)), v_u32_move_i(0), v_u32_move_i(1));
            y4 = v_s32_sel(v_s32_cmp(EQ, y4, v_u32_move_i(0)), v_u32_move_i(0), v_u32_move_i(1));

            int8_128 result =
                v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

            v_st_generic(i / 32, dst, 1, 255, result);
        }
    }

    if (len128 != 0) {
        int ldst_mask = pre_exp2(len128 / 128);
        int8_128 x = v_i32_ld_tnsr(len1024 / 32, mem, 1, ldst_mask);

        short8_128 x1 = unpack_16b(x, 1);
        short8_128 x0 = unpack_16b(x, 0);

        char8_128 _x0 = unpack_8b(x0, 0);
        char8_128 _x1 = unpack_8b(x0, 1);
        char8_128 _x2 = unpack_8b(x1, 0);
        char8_128 _x3 = unpack_8b(x1, 1);

        int8_128 y1 = __dlc_char_as_int(_x0);
        int8_128 y2 = __dlc_char_as_int(_x1);
        int8_128 y3 = __dlc_char_as_int(_x2);
        int8_128 y4 = __dlc_char_as_int(_x3);

        y1 = v_s32_sel(v_s32_cmp(EQ, y1, v_u32_move_i(0)), v_u32_move_i(0), v_u32_move_i(1));
        y2 = v_s32_sel(v_s32_cmp(EQ, y2, v_u32_move_i(0)), v_u32_move_i(0), v_u32_move_i(1));
        y3 = v_s32_sel(v_s32_cmp(EQ, y3, v_u32_move_i(0)), v_u32_move_i(0), v_u32_move_i(1));
        y4 = v_s32_sel(v_s32_cmp(EQ, y4, v_u32_move_i(0)), v_u32_move_i(0), v_u32_move_i(1));

        int8_128 result =
            v_s32_add(v_s32_add(v_s32_add(v_u32_shl(y4, 24), v_u32_shl(y3, 16)), v_u32_shl(y2, 8)), y1);

        v_st_generic(len1024 / 32, dst, 1, ldst_mask, result);
    }
}

inline void dma_pipeline_bool_to_bool(SIM_X86::tensor hbm_in, SIM_X86::tensor hbm_out, SIM_X86::tensor vmem, int len, int _VMEMsize,
                                      int UseVmemSize, int dim0, int dim0_128) {

    int l = 0;
    // handle the data till (len / _VMEMsize * _VMEMsize), since div is expansive, we use an extra addition
    int half_VMEMsize = _VMEMsize / 2;
    int flagIn0 = DONE, flagIn1 = DONE, flagOut0 = DONE, flagOut1 = DONE;
    // 当_VMEMsize 小于1024的时候，没有必要做dma pipeline
    bool firstForLoop = (l + _VMEMsize < len) && (_VMEMsize > 1024);

    // we pipeline the dma
    if (firstForLoop) {
        HBMtoVMemNoSync(hbm_in, vmem, l, 0, half_VMEMsize, &flagIn0);
    }
    for (; (l + _VMEMsize < len); l += _VMEMsize) {
        // if it's the first time we are in the for loop, we clear done bit,
        // other wise we wait for the previous dma to finish
        dlc_sync(flagOut1);
        HBMtoVMemNoSync(hbm_in, vmem, l + half_VMEMsize, half_VMEMsize, half_VMEMsize, &flagIn1);
        dlc_sync(flagIn0);
        // the input for this loop is guard by flagIn0, output is guard by flagOut0

        innerLoop_bool_to_bool(vmem, vmem, half_VMEMsize, dim0);

        VMEMtoHBMNoSync(vmem, hbm_out, 0, l, half_VMEMsize, &flagOut0);
        dlc_sync(flagOut0);
        HBMtoVMemNoSync(hbm_in, vmem, l + 2 * half_VMEMsize, 0, half_VMEMsize, &flagIn0);
        dlc_sync(flagIn1);

        innerLoop_bool_to_bool(vmem + half_VMEMsize / 32, vmem + half_VMEMsize / 32, half_VMEMsize, dim0);
        VMEMtoHBMNoSync(vmem, hbm_out, half_VMEMsize, half_VMEMsize, half_VMEMsize, &flagOut1);
    }
    if (firstForLoop) {
        dlc_sync(flagOut1);
    }
    // handle the remaining data
    if (l < len) {
        int VMEMsize_rem = len - l;
        HBMtoVMem(hbm_in, vmem, l, 0, VMEMsize_rem);
        innerLoop_bool_to_bool(vmem, vmem, VMEMsize_rem, dim0);
        VMEMtoHBM(vmem, hbm_out, 0, l, VMEMsize_rem);
    }
}

inline void bool_to_bool_bitwise(SIM_X86::tensor hbm_in, SIM_X86::tensor hbm_out, SIM_X86::tensor vmem, int len, int _VMEMsize,
                                 int UseVmemSize, int dim0, int dim0_128, transform_fn1_t_i fn) {
    int l = 0;
    for (; (l + _VMEMsize < len); l += _VMEMsize) {
        HBMtoVMem(hbm_in, vmem, l, 0, _VMEMsize);
        innerLoop_bool_to_bool_bitwise(vmem, vmem, _VMEMsize, dim0, fn);
        VMEMtoHBM(vmem, hbm_out, 0, l, _VMEMsize);
    }
    // handle the remaining data
    if (l < len) {
        int VMEMsize_rem = len - l;
        HBMtoVMem(hbm_in, vmem, l, 0, VMEMsize_rem);
        innerLoop_bool_to_bool_bitwise(vmem, vmem, VMEMsize_rem, dim0, fn);
        VMEMtoHBM(vmem, hbm_out, 0, l, VMEMsize_rem);
    }
}

#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"


#include "align.h"
#include "bf16.h"
#include "chunk.h"
#include "constval.h"

#include "libdevice.h"
#include "permute.h"

inline void HBMtoVMem(SIM_X86::tensor hbm_src, SIM_X86::tensor vmem_dst, int src_addr, int dst_addr, int length) {
#ifndef USE_CMEM
    int handle = dlc_dma(tensor_slice(hbm_src, src_addr >> 5), HBM, tensor_slice(vmem_dst, dst_addr >> 5),
                         VMEM, length, 128, 128, 128, 7);
    dlc_sync(handle);
#else
    int handle = dlc_dma(tensor_slice(hbm_src, src_addr >> 5), CMEM, tensor_slice(vmem_dst, dst_addr >> 5),
                         VMEM, length, 128, 128, 128, 7);
    dlc_sync(handle);
#endif
}

inline void VMEMtoHBM(SIM_X86::tensor vmem_src, SIM_X86::tensor hbm_dst, int src_addr, int dst_addr, int length) {
#ifndef USE_CMEM
    int handle = dlc_dma(tensor_slice(vmem_src, src_addr >> 5), VMEM, tensor_slice(hbm_dst, dst_addr >> 5),
                         HBM, length, 128, 128, 128, 7);
    dlc_sync(handle);
#else
    int handle = dlc_dma(tensor_slice(vmem_src, src_addr >> 5), VMEM, tensor_slice(hbm_dst, dst_addr >> 5),
                         CMEM, length, 128, 128, 128, 7);
    dlc_sync(handle);
#endif
}
inline void smallgranule_dma_xys0_s_2_xys1_s(SIM_X86::tensor st_addr) {
    asm volatile("{"
                 "S1@(pr0)	Nah = dma [smem:%[smemaddr]];"
                 "}"
                 :
                 : [smemaddr] "r"(st_addr)
                 :);
}

inline void sync_and_fence_2() {
    asm volatile("{"
                 "pseudo@0	@pseudo imm_2 = 2;"
                 "MISC@(pr0) Nah = wait.done 4, 0;"
                 "}"
                 "{"
                 "S0@(pr0)	Nah = fence;"
                 "}");
}


/* 这里会将value从xys1的xys1_small_granule_address smem地址传到xys0的xys0_small_granule_address
smem地址，temp_smem是用来存中间需要存放的帧， 还需要注意length，这是需要修改的长度，一次传多少*/
inline void smallgranule_dma_2xys_value(float value, int xys0_small_granule_address,
                                  int xys1_small_granule_address, SIM_X86::tensor temp_smem) {

    ((float *)((int)xys1_small_granule_address))[0] = *(float *)(&value);

    // ==============HEADER=================
    // trance_en-1    : 0
    // dest_opcode-2  : 00
    // dst_core_id-3  : 010         xys0
    // dst_mem_id-2   : 01          smem
    // reserved-1     : 0
    // src_opcode-2   : 00          read memory
    // src_core_id-3  : 011         xys1
    // src_mem_id-2   : 01          smem
    // dma_data_type-2: 00
    // reserved-4     : 0000
    // dst_id-10      : 0000000000
    // =======================================
    int header = 0b00001001000011010000000000000000;

    // ==============SRC_SYNC_FLAG=================
    // reserved-16        : 0000000000000000
    // src_sync_core_id-3 : 000
    // src_sync_flag_id-13: 0000000000000
    // =======================================
    int src_sync_flag = 0;

    // ==============DST_SYNC_FLAG=================
    // dst1_sync_core_id-3  : 000
    // dst1_sync_flag_id-13 : 0000000000000
    // dst0_sync_core_id-3  : 010
    // dst0_sync_flag_id-13 : 0000000000010 (2)
    // =======================================
    int dst_sync_flag = 0b00000000000000000100000000000010;

    // =============LENGTH====================
    // granule-1      : 1
    // length-31      : 0000000000000000000000000000010
    // =======================================
    int length = 0b10000000000000000000000000000001;

    // =============SRC_ADDR==================
    // granule-1      : 1
    // mem_offset-31  : 0000000000000000000000000000000 (0)
    // =======================================
    int src_ad = (1 << 31) | (xys1_small_granule_address / 4);

    // =============DST_ADDR==================
    // granule-1      : 1
    // mem_offset-31  : 0000000000000000000000000000000 (0)
    // =======================================
    int dst_ad = (1 << 31) | (xys0_small_granule_address / 4);

    // =============SRC_STRIDE0==================
    // src_stride0-32  : signed(can be negative)
    // =======================================
    int src_stride0 = 1;

    // =============DST_STRIDE0==================
    // dst_stride0-32  : signed(can be negative)
    // =======================================
    int dst_stride0 = 1;

    ((int *)((int)temp_smem))[0] = header;
    ((int *)((int)temp_smem))[1] = src_sync_flag;
    ((int *)((int)temp_smem))[2] = dst_sync_flag;
    ((int *)((int)temp_smem))[3] = length;
    ((int *)((int)temp_smem))[4] = src_ad;
    ((int *)((int)temp_smem))[5] = dst_ad;
    ((int *)((int)temp_smem))[6] = src_stride0;
    ((int *)((int)temp_smem))[7] = dst_stride0;

    smallgranule_dma_xys0_s_2_xys1_s(temp_smem);
}

/* 这里会将idx和min从xys1的xys1_small_granule_address smem地址传到xys0的xys0_small_granule_address
smem地址，temp_smem是用来存中间需要存放的帧， 还需要注意length，这是需要修改的长度，一次传多少*/
inline void smallgranule_dma_2xys(int idx, float value, int xys0_small_granule_address,
                                  int xys1_small_granule_address, SIM_X86::tensor temp_smem) {

    ((float *)((int)xys1_small_granule_address))[0] = *(float *)(&value);
    ((int *)((int)xys1_small_granule_address))[1] = *(int *)(&idx);

    // ==============HEADER=================
    // trance_en-1    : 0
    // dest_opcode-2  : 00
    // dst_core_id-3  : 010         xys0
    // dst_mem_id-2   : 01          smem
    // reserved-1     : 0
    // src_opcode-2   : 00          read memory
    // src_core_id-3  : 011         xys1
    // src_mem_id-2   : 01          smem
    // dma_data_type-2: 00
    // reserved-4     : 0000
    // dst_id-10      : 0000000000
    // =======================================
    int header = 0b00001001000011010000000000000000;

    // ==============SRC_SYNC_FLAG=================
    // reserved-16        : 0000000000000000
    // src_sync_core_id-3 : 000
    // src_sync_flag_id-13: 0000000000000
    // =======================================
    int src_sync_flag = 0;

    // ==============DST_SYNC_FLAG=================
    // dst1_sync_core_id-3  : 000
    // dst1_sync_flag_id-13 : 0000000000000
    // dst0_sync_core_id-3  : 010
    // dst0_sync_flag_id-13 : 0000000000010 (2)
    // =======================================
    int dst_sync_flag = 0b00000000000000000100000000000010;

    // =============LENGTH====================
    // granule-1      : 2
    // length-31      : 0000000000000000000000000000010
    // =======================================
    int length = 0b10000000000000000000000000000010;

    // =============SRC_ADDR==================
    // granule-1      : 1
    // mem_offset-31  : 0000000000000000000000000000000 (0)
    // =======================================
    int src_ad = (1 << 31) | (xys1_small_granule_address / 4);

    // =============DST_ADDR==================
    // granule-1      : 1
    // mem_offset-31  : 0000000000000000000000000000000 (0)
    // =======================================
    int dst_ad = (1 << 31) | (xys0_small_granule_address / 4);

    // =============SRC_STRIDE0==================
    // src_stride0-32  : signed(can be negative)
    // =======================================
    int src_stride0 = 1;

    // =============DST_STRIDE0==================
    // dst_stride0-32  : signed(can be negative)
    // =======================================
    int dst_stride0 = 1;

    ((int *)((int)temp_smem))[0] = header;
    ((int *)((int)temp_smem))[1] = src_sync_flag;
    ((int *)((int)temp_smem))[2] = dst_sync_flag;
    ((int *)((int)temp_smem))[3] = length;
    ((int *)((int)temp_smem))[4] = src_ad;
    ((int *)((int)temp_smem))[5] = dst_ad;
    ((int *)((int)temp_smem))[6] = src_stride0;
    ((int *)((int)temp_smem))[7] = dst_stride0;

    smallgranule_dma_xys0_s_2_xys1_s(temp_smem);
}

inline float8_128 checkVectorContainsNaN(bool8_128 cond, float8_128 one_v, float8_128 zero_v) {
    float8_128 res = v_f32_sel(cond, zero_v, one_v);
    float8_128 a2 = v_row_rotate(res, 1);
    float8_128 b = v_f32_max(res, a2);
    float8_128 a3 = v_row_rotate(res, 0);
    float8_128 a4 = v_row_rotate(a2, 1);
    float8_128 a5 = v_row_rotate(a3, 0);
    float8_128 c = v_f32_max(a3, a5);
    float8_128 a6 = v_row_rotate(a4, 1);
    b = v_f32_max(a4, b);
    float8_128 a7 = v_row_rotate(a5, 0);
    b = v_f32_max(a6, b);
    c = v_f32_max(a7, c);
    float8_128 a8 = v_row_rotate(a6, 1);
    b = v_f32_max(a8, b);
    b = v_f32_max(b, c);
    return v_f32_max_row(b);
}

// 如果超过2^23的上限，会出错，可以不断减去2^23直到可以比较
inline float8_128 max1024(float8_128 a) {
    float8_128 a2 = v_row_rotate(a, 1);
    float8_128 a3 = v_row_rotate(a, 0);
    float8_128 a4 = v_row_rotate(a2, 1);
    float8_128 a5 = v_row_rotate(a3, 0);
    float8_128 a6 = v_row_rotate(a4, 1);
    float8_128 a7 = v_row_rotate(a5, 0);
    float8_128 a8 = v_row_rotate(a6, 1);
    float8_128 b = v_f32_max(a, a2);
    float8_128 c = v_f32_max(a3, a5);
    b = v_f32_max(a4, b);
    b = v_f32_max(a6, b);
    c = v_f32_max(a7, c);
    c = v_f32_max(a8, c);
    b = v_f32_max(c, b);
    b = v_f32_max_row(b);
    return b;
}

// 如果超过2^23的上限，会出错，可以不断减去2^23直到可以比较
inline float8_128 min1024(float8_128 a) {
    float8_128 a2 = v_row_rotate(a, 1);
    float8_128 a3 = v_row_rotate(a, 0);
    float8_128 a4 = v_row_rotate(a2, 1);
    float8_128 a5 = v_row_rotate(a3, 0);
    float8_128 a6 = v_row_rotate(a4, 1);
    float8_128 a7 = v_row_rotate(a5, 0);
    float8_128 a8 = v_row_rotate(a6, 1);
    float8_128 b = v_f32_min(a, a2);
    float8_128 c = v_f32_min(a3, a5);
    b = v_f32_min(a4, b);
    b = v_f32_min(a6, b);
    c = v_f32_min(a7, c);
    c = v_f32_min(a8, c);
    b = v_f32_min(c, b);
    b = v_f32_min_row(b);
    return b;
}

// 如果超过2^23的上限，会出错，可以不断减去2^23直到可以比较
inline int8_128 min1024_i(int8_128 x) {
    float8_128 a = __dlc_int2float_rn(x);

    float8_128 a2 = v_row_rotate(a, 1);
    float8_128 a3 = v_row_rotate(a, 0);
    float8_128 a4 = v_row_rotate(a2, 1);
    float8_128 a5 = v_row_rotate(a3, 0);
    float8_128 a6 = v_row_rotate(a4, 1);
    float8_128 a7 = v_row_rotate(a5, 0);
    float8_128 a8 = v_row_rotate(a6, 1);

    float8_128 b = v_f32_min(a, a2);
    float8_128 c = v_f32_min(a3, a5);

    b = v_f32_min(a4, b);
    c = v_f32_min(a7, c);

    b = v_f32_min(a6, b);
    b = v_f32_min(a8, b);
    c = v_f32_min(b, c);

    b = v_f32_min_row(c);

    int8_128 d = __dlc_float2int_rz(b);
    return d;
}

inline int8_128 get_desired_core_id(int w) {
    int8_128 core_id = get_core_id() >> 7;
    // 前者方法会超过2^23的上限，后者采用scalar的加法
    if (w < 8388608) {
        core_id = __dlc_float2int_rz(v_f32_mul_b(v_cvt_itof(core_id), v_u32_move_f(w)));
    }
    core_id = v_s32_add(core_id, v_u32_and(get_core_id(), v_u32_move_i(0x7f)));
    return core_id;
}

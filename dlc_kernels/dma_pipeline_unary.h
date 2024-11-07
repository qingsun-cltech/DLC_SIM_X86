#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#ifndef _PIPELINE_UNARY_H_X86_
#define _PIPELINE_UNARY_H_X86_

#include "math.h"


typedef float8_128 (*transform_fn1_t)(float8_128, SIM_X86::tensor aux);

inline void element_wise_transform_inplace(SIM_X86::tensor vmem, int len, transform_fn1_t fn, SIM_X86::tensor aux) {
    int len1024 = len / 1024 * 1024;
    float8_128 res;
#pragma clang loop unroll_count(16)
    for (int i = 0; i < len1024; i += 1024) {
        float8_128 val = v_f32_ld_tnsr_b(i / 32, vmem);
        res = fn(val, aux);
        v_f32_st_tnsr_b(i / 32, vmem, res);
    }
    if (len != len1024) {
        int ldmk = (1 << ((len % 1024) / 128)) - 1;
        float8_128 val = v_f32_ld_tnsr_st_msk(len1024 / 32, vmem, 1, ldmk);
        res = fn(val, aux);
        v_f32_st_tnsr_st_msk(len1024 / 32, vmem, 1, ldmk, res);
    }
//     __attribute__((unused))volatile float wait_1 = vstore_wait(res);
}

// 大数据量下有更好表现
inline void element_wise_lanhu_model(SIM_X86::tensor hbmin, SIM_X86::tensor hbmout, int numel, SIM_X86::tensor vmem, int vmemlen,
                                     transform_fn1_t fn, SIM_X86::tensor aux) {
    int l = 0;
    // handle the data till (len / _VMEMsize * _VMEMsize), since div is expansive, we use an extra addition
    int half_vmem = vmemlen / 2;
    int flagIn0 = DONE, flagIn1 = DONE, flagOut0 = DONE, flagOut1 = DONE;
    bool firstForLoop = l + vmemlen <= numel;
    // we pipeline the dma
    if (firstForLoop) {
        flagIn0 = dlc_dma(hbmin + l / 32, HBM, vmem, VMEM, half_vmem, 128, 128, 128, 7);
    }
    for (; l + vmemlen <= numel; l += vmemlen) {
        // if it's the first time we are in the for loop, we clear done bit,
        // other wise we wait for the previous dma to finish
        dlc_sync(flagOut1);
        flagIn1 = dlc_dma(hbmin + (l + half_vmem) / 32, HBM, vmem + half_vmem / 32, VMEM, half_vmem, 128, 128,
                          128, 7);
        dlc_sync(flagIn0);
        // the input for this loop is guard by flagIn0, output is guard by flagOut0
        element_wise_transform_inplace(vmem, half_vmem, fn, aux);
        flagOut0 = dlc_dma(vmem, VMEM, hbmout + l / 32, HBM, half_vmem, 128, 128, 128, 7);
        dlc_sync(flagOut0);

        flagIn0 = dlc_dma(hbmin + (l + 2 * half_vmem) / 32, HBM, vmem, VMEM, half_vmem, 128, 128, 128, 7);
        dlc_sync(flagIn1);
        // the input for this loop is guard by flagIn1, output is guard by flagOut1
        element_wise_transform_inplace(vmem + half_vmem / 32, half_vmem, fn, aux);
        flagOut1 = dlc_dma(vmem + half_vmem / 32, VMEM, hbmout + (l + half_vmem) / 32, HBM, half_vmem, 128,
                           128, 128, 7);
    }
    if (firstForLoop) {
        dlc_sync(flagOut1);
    }
    // handle the remaining data
    if (l < numel) {
        int curvmem = numel - l;
        int flagA = dlc_dma(hbmin + l / 32, HBM, vmem, VMEM, curvmem, 128, 128, 128, 7);
        dlc_sync(flagA);
        element_wise_transform_inplace(vmem, curvmem, fn, aux);
        int flagB = dlc_dma(vmem, VMEM, hbmout + l / 32, HBM, curvmem, 128, 128, 128, 7);
        dlc_sync(flagB);
    }
}

inline void manual_dma_h2v_flag1(SIM_X86::tensor src, SIM_X86::tensor dst, int len) {
    asm volatile("{ pseudo@0 @pseudo imm_5 = 301; "
                 "  MISC@(pr0) 7 = setsync.clear 0; }"
                 "{ pseudo@0 @pseudo imm_2 = 16685; "
                 "  pseudo@0 @pseudo imm_3 = 16384; "
                 "  S0@(pr0) [dest:%[dst]], [sflag:r45] = dmalocal [src:%[src]], %[len], _4352; } "
                 :
                 : [src] "r"((int)src >> 2), [dst] "r"((int)dst >> 2), [len] "r"((uint)len >> 7)
                 :);
}

inline void manual_dma_h2v_flag2(SIM_X86::tensor src, SIM_X86::tensor dst, int len) {
    asm volatile("{ pseudo@0 @pseudo imm_5 = 302; "
                 "  MISC@(pr0) 7 = setsync.clear 0; }"
                 "{ pseudo@0 @pseudo imm_2 = 16686; "
                 "  pseudo@0 @pseudo imm_3 = 16384; "
                 "  S0@(pr0) [dest:%[dst]], [sflag:r45] = dmalocal [src:%[src]], %[len], _4352; } "
                 :
                 : [src] "r"((int)src >> 2), [dst] "r"((int)dst >> 2), [len] "r"((uint)len >> 7)
                 :);
}

inline void manual_dma_v2h_flag1(SIM_X86::tensor src, SIM_X86::tensor dst, int len) {
    asm volatile("{ pseudo@0 @pseudo imm_5 = 301; "
                 "  pseudo@0 @pseudo imm_1 = 7; "
                 "  MISC@(pr0) 7 = setsync.clear 0; }"
                 "{ pseudo@0 @pseudo imm_2 = 16685; "
                 "  pseudo@0 @pseudo imm_3 = 16384; "
                 "  S0@(pr0) [dest:%[dst]], [sflag:r45] = dmalocal [src:%[src]], %[len], _2560; } "
                 :
                 : [src] "r"((int)src >> 2), [dst] "r"((int)dst >> 2), [len] "r"((uint)len >> 7)
                 :);
}

inline void manual_dma_v2h_flag2(SIM_X86::tensor src, SIM_X86::tensor dst, int len) {
    asm volatile("{ pseudo@0 @pseudo imm_5 = 302; "
                 "  pseudo@0 @pseudo imm_1 = 7; "
                 "  MISC@(pr0) 7 = setsync.clear 0; }"
                 "{ pseudo@0 @pseudo imm_2 = 16686; "
                 "  pseudo@0 @pseudo imm_3 = 16384; "
                 "  S0@(pr0) [dest:%[dst]], [sflag:r45] = dmalocal [src:%[src]], %[len], _2560; } "
                 :
                 : [src] "r"((int)src >> 2), [dst] "r"((int)dst >> 2), [len] "r"((uint)len >> 7)
                 :);
}

inline void manual_dma_done_flag1() {
    asm volatile("{ pseudo@0 @pseudo imm_5 = 301; MISC@(pr0) 7 = setsync.done 0; }" : : :);
}

inline void manual_dma_done_flag2() {
    asm volatile("{ pseudo@0 @pseudo imm_5 = 302; MISC@(pr0) 7 = setsync.done 0; }" : : :);
}

inline void manual_dma_sync_flag1() {
    asm volatile("{ pseudo@0 @pseudo imm_5 = 301; MISC@(pr0) Nah = wait.done 7, 0; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 :
                 :);
}

inline void manual_dma_sync_flag2() {
    asm volatile("{ pseudo@0 @pseudo imm_5 = 302; MISC@(pr0) Nah = wait.done 7, 0; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 :
                 :);
}

// 小数据量下有更好表现
inline void element_wise_qingshan_model(SIM_X86::tensor in_hbm, SIM_X86::tensor out_hbm, int numel, SIM_X86::tensor vmem, int vmemlen,
                                        transform_fn1_t fn, SIM_X86::tensor aux) {
    int half_vmem = vmemlen / 2;
    SIM_X86::tensor A = vmem;
    SIM_X86::tensor B = vmem + half_vmem / 32;
    int hbm_len = numel;

    int loff = 0;

    manual_dma_done_flag1();
    manual_dma_done_flag2();

    manual_dma_h2v_flag1(in_hbm + loff / 32, A, min(hbm_len - loff, half_vmem));

    int idx = 0;
    int soff = 0;
    for (; soff < hbm_len; soff += half_vmem, idx++) {
        loff += half_vmem;
        int tllen = min(hbm_len - loff, half_vmem);
        int tslen = min(hbm_len - soff, half_vmem);
        if (idx % 2 == 0) {
            manual_dma_sync_flag1();
            // already stopped by fence, so jmp only has a little cost
            if (loff < hbm_len) {
                manual_dma_sync_flag2();
                manual_dma_h2v_flag2(in_hbm + loff / 32, B, tllen);
            }
            element_wise_transform_inplace(A, tslen, fn, aux);
            manual_dma_v2h_flag1(A, out_hbm + soff / 32, tslen);
        } else {
            manual_dma_sync_flag2();
            if (loff < hbm_len) {
                manual_dma_sync_flag1();
                manual_dma_h2v_flag1(in_hbm + loff / 32, A, tllen);
            }
            element_wise_transform_inplace(B, tslen, fn, aux);
            manual_dma_v2h_flag2(B, out_hbm + soff / 32, tslen);
        }
    }

    manual_dma_sync_flag1();
    manual_dma_sync_flag2();
}

#endif

#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// #include "typehint.h"
#include "align.h"

inline int VMEMtoHBM2_nosync(SIM_X86::tensor vmem, SIM_X86::tensor hbm, int H, int W, int dst_stride) {
    int len = H * 128;
    for (int i = 0; i < ALIGN128(W); i += 128) {
#ifdef USE_CMEM
        dlc_dma(vmem + i * H / 32, VMEM, hbm + i / 32, CMEM, len, 128, dst_stride, 128, 7);
#else
        dlc_dma(vmem + i * H / 32, VMEM, hbm + i / 32, HBM, len, 128, dst_stride, 128, 7);
#endif
    }
#ifdef USE_CMEM 
    return dlc_dma(vmem, VMEM, hbm, CMEM, 0, 128, dst_stride, 128, 7);
    // dlc_sync(sync);
#else
    return dlc_dma(vmem, VMEM, hbm, HBM, 0, 128, dst_stride, 128, 7);
    // dlc_sync(sync);
#endif 
}

inline int HBMtoVMEM2_nosync(SIM_X86::tensor hbm, SIM_X86::tensor vmem, int H, int W, int src_stride){
    int len = H * 128;
    for(int i = 0; i < ALIGN128(W); i += 128){
        #ifdef USE_CMEM
            dlc_dma(hbm + i / 32, CMEM, vmem + i * H / 32, VMEM, len, src_stride, 128, 128, 7);
        #else
            dlc_dma(hbm + i / 32, HBM, vmem + i * H / 32, VMEM, len, src_stride, 128, 128, 7);
        #endif
        // dlc_sync(sync);
    }   
    #ifdef USE_CMEM  
        return dlc_dma(hbm, CMEM, vmem, VMEM, 0, src_stride, 128, 128, 7);
        // dlc_sync(sync);
    #else
        return dlc_dma(hbm, HBM, vmem, VMEM, 0, src_stride, 128, 128, 7);
        // dlc_sync(sync);
    #endif  
}

inline int VMEMtoHBM2_cmem_nosync(SIM_X86::tensor vmem, SIM_X86::tensor hbm, SIM_X86::tensor cmem, int H, int W, int dst_stride) {
    int len = H * 128;
    for (int i = 0; i < ALIGN128(W); i += 128) {
        int sync = dlc_dma(vmem + i * H / 32, VMEM, cmem + i / 32, CMEM, len, 128, dst_stride, 128, 7);
        dlc_sync(sync);
    }
    return dlc_dma(cmem, CMEM, hbm, HBM, H * W, 128, 128, 128, 7);
}

inline int CMEMtoVMEM_nosync(SIM_X86::tensor cmem, SIM_X86::tensor vmem, int H, int W, int src_stride){
    int len = H * 128;
    for(int i = 0; i < ALIGN128(W); i += 128){
        dlc_dma(cmem + i / 32, CMEM, vmem + i * H / 32, VMEM, len, src_stride, 128, 128, 7);
        // dlc_sync(sync);
    }   
    return dlc_dma(cmem, CMEM, vmem, VMEM, 0, src_stride, 128, 128, 7);
}

inline void swap_tensor(SIM_X86::tensor* a, SIM_X86::tensor* b){
    SIM_X86::tensor tmp = *a;
    *a = *b;
    *b = tmp;
}
//这里计算下一个地址，需要遵守外面的循环的逻辑
inline int next_posa(int H, int W, int h,  int w, int ibw, int num_bw, int steph, int stepw,
                     int* process_ah, int* process_aw){
    w += stepw;
    if(w >= W){
        w = 0;
        if((ibw == num_bw -1) || (stepw >= W)){
            h = h + steph;
        }
    }
    if(h >= H){
        *process_ah = min(H, steph);
        *process_aw = min(W, stepw);
        return 0;
    }
    *process_ah = min(H - h, steph);
    *process_aw = min(W - w, stepw);
    return (h * ALIGN128(W) + w) / 32;
}

inline int next_posa_bf16(int H, int W, int h,  int w, int ibw, int num_bw, int steph, int stepw,
                     int* process_ah, int* process_aw){
    w += stepw;
    if(w >= W){
        w = 0;
        if((ibw == num_bw -1) || (stepw >= W)){
            h = h + steph;
        }
    }
    if(h >= H){
        *process_ah = min(H, steph);
        *process_aw = min(W, stepw);
        return 0;
    }
    *process_ah = min(H - h, steph);
    *process_aw = min(W - w, stepw);
    return (h * ALIGN256(W) + w) / 64;
}

inline int next_posa_cmem(int H, int W, int h,  int w, int ibw, int num_bw, int steph, int stepw,
                     int* process_ah, int* process_aw, bool* is_cmem, int* cmem_address){
    int pre_h = h;
    w += stepw;
    if(w >= W){
        w = 0;
        if((ibw == num_bw -1) || (stepw >= W)){
            h = h + steph;
        }
    }
    if(h >= H){
        *process_ah = min(H, steph);
        *process_aw = min(W, stepw);
        if(pre_h != 0){
            *is_cmem = true;
            *cmem_address = 0;
        }
        return 0;
    }
    if(pre_h != h){
        *is_cmem = true;
        *cmem_address = (h * ALIGN128(W) + w) / 32;
    }
    *process_ah = min(H - h, steph);
    *process_aw = min(W - w, stepw);
    return w / 32;
}

inline int next_posb(int H, int W, int h, int w, int steph, int stepw, int* process_bw){
    h += steph;
    if(h >= H){
        h = 0;
        w = w + stepw;
    }
    if(w >= W){
        *process_bw = min(W, stepw);
        return 0;
    }
    *process_bw = min(W - w, stepw);
    return (h * ALIGN128(W) + w) / 32;
}

inline int next_posb_bf16(int H, int W, int h, int w, int steph, int stepw, int* process_bw){
    h += steph;
    if(h >= H){
        h = 0;
        w = w + stepw;
    }
    if(w >= W){
        *process_bw = min(W, stepw);
        return 0;
    }
    *process_bw = min(W - w, stepw);
    return (h * ALIGN256(W) + w) / 64;
}

inline void HBM2VMEM_pingpong(SIM_X86::tensor input_hbm, SIM_X86::tensor* input_cur, SIM_X86::tensor* input_next,
                              SIM_X86::tensor mat2_hbm, SIM_X86::tensor* mat2_cur, SIM_X86::tensor* mat2_next,
                              int idxa, int idxb, int* prea, int* preb,
                              int a_stride, int b_stride, int input_next_Offset, int mat2_next_offset,
                              int process_ah, int process_aw, int process_bw, int* sync0, int* sync1){
    if(idxa != *prea){
        *prea = idxa;
        swap_tensor(input_cur, input_next);
        dlc_sync(*sync0);
        *sync0 = HBMtoVMEM2_nosync(input_hbm + input_next_Offset, *input_next, process_ah, process_aw, a_stride);
    }
    if(idxb != *preb){
        *preb = idxb;
        swap_tensor(mat2_cur, mat2_next);
        dlc_sync(*sync1);
        *sync1 = HBMtoVMEM2_nosync(mat2_hbm + mat2_next_offset, *mat2_next, process_aw, process_bw, b_stride); 
    }
}

inline void HBM2VMEM_pingpong_bf16(SIM_X86::tensor input_hbm, SIM_X86::tensor* input_cur, SIM_X86::tensor* input_next,
                              SIM_X86::tensor mat2_hbm, SIM_X86::tensor* mat2_cur, SIM_X86::tensor* mat2_next,
                              int idxa, int idxb, int* prea, int* preb,
                              int a_stride, int b_stride, int input_next_Offset, int mat2_next_offset,
                              int process_ah, int process_aw, int process_bw, int* sync0, int* sync1){
    if(idxa != *prea){
        *prea = idxa;
        swap_tensor(input_cur, input_next);
        dlc_sync(*sync0);
        *sync0 = HBMtoVMEM2_nosync(input_hbm + input_next_Offset, *input_next, process_ah, ALIGN256(process_aw)/2, a_stride);
    }
    if(idxb != *preb){
        *preb = idxb;
        swap_tensor(mat2_cur, mat2_next);
        dlc_sync(*sync1);
        *sync1 = HBMtoVMEM2_nosync(mat2_hbm + mat2_next_offset, *mat2_next, process_aw, ALIGN256(process_bw)/2, b_stride); 
    }
}

//lhs: cmem
inline void HBM2VMEM_pingpong_cmem(SIM_X86::tensor input_hbm, SIM_X86::tensor cmem, SIM_X86::tensor* input_cur, SIM_X86::tensor* input_next,
                              SIM_X86::tensor mat2_hbm, SIM_X86::tensor* mat2_cur, SIM_X86::tensor* mat2_next,
                              int idxa, int idxb, int* prea, int* preb,
                              int AW128, int b_stride, int input_next_Offset, int mat2_next_offset,
                              int process_ah, int process_aw, int process_bw, int* sync0, int* sync1,
                              bool is_cmem, int cmem_addr){
    if(is_cmem){
        int sync5 = dlc_dma(input_hbm + cmem_addr, HBM, cmem, CMEM, process_ah * AW128, 128, 128, 128, 7);
        dlc_sync(sync5);
    }
    if(idxa != *prea){
        *prea = idxa;
        swap_tensor(input_cur, input_next);
        dlc_sync(*sync0);
        *sync0 = CMEMtoVMEM_nosync(cmem + input_next_Offset, *input_next, process_ah, process_aw, AW128);
    }
    if(idxb != *preb){
        *preb = idxb;
        swap_tensor(mat2_cur, mat2_next);
        dlc_sync(*sync1);
        *sync1 = HBMtoVMEM2_nosync(mat2_hbm + mat2_next_offset, *mat2_next, process_aw, process_bw, b_stride); 
    }
}
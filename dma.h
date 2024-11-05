#ifndef _DMA_H_X86_
#define _DMA_H_X86_

#include "libdevice.h"
#include "typehint.h"

#ifdef USE_CMEM
const int D_HBM = 3;
#else
const int D_HBM = 1;
#endif
const int D_VMEM = 2;
// src      : [ dim0, dim1, dim2, dim3 ]
// src_index: [ idx0, idx1, idx2, idx3 ]
// dst      : [ vmemH, vmemW ]
// unit     :  4B
//  inline void load_mat_level2(SIM_X86::tensor src, SIM_X86::tensor dst, const int mh, const int mw, const int b1h, const int
//  b1w,
//                              const int i1h, const int i1w, const int b2h, const int b2w, const int i2h,
//                              const int i2w) {
/*
inline void load_mat_level2(SIM_X86::tensor src,
                            SIM_X86::tensor dst,
                            const int mh,
                            const int mw,
                            const int b1h,
                            const int b1w,
                            const int i1h,
                            const int i1w,
                            const int b2h,
                            const int b2w,
                            const int i2h,
                            const int i2w) 函数说明：

       |<-------------dim1------------>|
       |               |<-----dim3---->|
       |               |       |<-vmW->|
idx0 > +---------------+-------+-------+ ---------------
       |               |       |       |        ^      ^
       |               |       |       |        |      |
       |               |       |       |        |      |
       |        idx2 > |-------+-------+ --     + dim2 |
       |               |       |░░░░░░░| ^      |      |
       |               |       |░░░░░░░| + vmH  |      |
       |               |       |░░░░░░░| v      v      |
       |---------------+---------------+ --------      + dim0
       |               |       ^       |               |
       |               |       idx3    |               |
       |               |               |               |
       |               |               |               |
       |               |               |               |
       |               |               |               v
       +-------------------------------+ ---------------
                       ^
                       idx1

load ░:
(b2h) times load offset (i1h * b1h * mw + i1w * b1w + i2h * b2h * mw + i2w * b2w) stride (mw)
*/
inline void load_mat_level2(SIM_X86::tensor src, SIM_X86::tensor dst, const int dim0, const int dim1, const int dim2,
                            const int dim3, const int idx0, const int idx1, const int idx2, const int idx3,
                            const int vmemH, const int vmemW, const int src_stride) {
    // const int offset = i1h * b1h * mw + i1w * b1w + i2h * b2h * mw + i2w * b2w;
    const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    const int unitLen = vmemW / 128;
    if (vmemH < unitLen) {
        for (int i = 0; i < vmemH; ++i) {
            int h = dlc_dma(tensor_slice(src, offset / 32 + src_stride * i / 32), HBM, tensor_slice(dst, vmemW * i / 32), VMEM,
                            vmemW, 128, 128, 128, 7);
            dlc_sync(h);
        }
    } else {
        for (int i = 0; i < unitLen; i += 1) {
            int h = dlc_dma(tensor_slice(src, offset / 32 + i * 4), HBM, tensor_slice(dst, i * 4), VMEM,
                            vmemH * 128, src_stride, vmemW, 128, 7);
            dlc_sync(h);
        }
    }
}

// src      : [ vmemH, vmemW ]
// dst      : [ dim0, dim1, dim2, dim3 ]
// dst_index: [ idx0, idx1, idx2, idx3 ]
// unit     :  4B
inline void store_mat_level2(SIM_X86::tensor src, SIM_X86::tensor dst, const int dim0, const int dim1, const int dim2,
                             const int dim3, const int idx0, const int idx1, const int idx2, const int idx3,
                             const int vmemH, const int vmemW, const int dst_stride) {
    const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    const int unitLen = vmemW / 128;
    __attribute((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, src, 1, 1));
    if (vmemH < unitLen) {
        for (int i = 0; i < vmemH; ++i) {
            int h = dlc_dma(tensor_slice(src, vmemW * i / 32), VMEM, tensor_slice(dst, offset / 32 + dst_stride * i / 32), HBM,
                            vmemW, 128, 128, 128, 7);
            dlc_sync(h);
        }
    } else {
        for (int i = 0; i < unitLen; i += 1) {
            int h = dlc_dma(tensor_slice(src, i * 4), VMEM, tensor_slice(dst, offset / 32 + i * 4), HBM,
                            vmemH * 128, vmemW, dst_stride, 128, 7);
            dlc_sync(h);
        }
    }
}

// store_mat_0123(vmem_m, hbm_logsumexp, batch, num_head, seq_len, 1, b, nh, i, 0, q_chunk_h, 128);
// src_stride: dim3
inline void load_mat_0123(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
                          int idx2, int idx3, int vmemH, int vmemW) {
    load_mat_level2(src, dst, dim0, dim1, dim2, dim3, idx0, idx1, idx2, idx3, vmemH, vmemW, dim3);
}

// dst_stride: dim3
inline void store_mat_0123(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
                           int idx2, int idx3, int vmemH, int vmemW) {
    store_mat_level2(src, dst, dim0, dim1, dim2, dim3, idx0, idx1, idx2, idx3, vmemH, vmemW, dim3);
}

inline void HBM2VMem(SIM_X86::tensor hbm_address, SIM_X86::tensor vmem_address, int length){
    int handle = dlc_dma(tensor_slice(hbm_address, 0 / 32), HBM,
                        tensor_slice(vmem_address, 0 / 32), VMEM, length, 128,
                        128, 128, 7);
    dlc_sync(handle);
}

inline void Vmem2HBM(SIM_X86::tensor vmem_address, SIM_X86::tensor hbm_address, int length){
    int handle = dlc_dma(tensor_slice(vmem_address, 0 / 32), VMEM,
                        tensor_slice(hbm_address, 0 / 32), HBM, length, 128,
                        128, 128, 7);
    dlc_sync(handle);
}

//dst_stride为128
inline void HBM2VMEMstride(SIM_X86::tensor hbm, SIM_X86::tensor vmem, int H, int _W, int src_stride, int dst_stride){
    int len = H * 128;
    int W = (_W + 127) & 0xffffff80;
    for(int i = 0; i < W; i += 128){
        #ifdef USE_CMEM
            dlc_dma(hbm + i / 32, CMEM, vmem + i * H / 32, VMEM, len, src_stride, dst_stride, 128, 7);
        #else
            dlc_dma(hbm + i / 32, HBM, vmem + i * H / 32, VMEM, len, src_stride, dst_stride, 128, 7);
        #endif
    }
    #ifdef USE_CMEM
        int sync = dlc_dma(hbm, CMEM, vmem, VMEM, 0, src_stride, 128, 128, 7);
        dlc_sync(sync);
    #else
        int sync = dlc_dma(hbm, HBM, vmem, VMEM, 0, src_stride, 128, 128, 7);
        dlc_sync(sync);
    #endif  
}

#endif
#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#ifndef __MM_T_H_X86__
#define __MM_T_H_X86__

#include "ldst.h"

#include "nn.h"
#include "bf16.h"
#include "align.h"
//matmul_f32
inline int CalcVMemBlockSizeMatrixW(int MatrixW, int h, int VmemSize){
    int l = 128, r = MatrixW;
    while(l < r){
        int mid = (l + r + 1) >> 1;
        if((h * ALIGN128(mid)) > VmemSize) r = mid - 128;
        else l = mid; 
    }
    return l;
}

inline void tile_trans_transfer2(SIM_X86::tensor src, SIM_X86::tensor dst, int src_h, int src_w) {
    int paddingH = ALIGN128(src_h);
    int paddingW = ALIGN128(src_w);
    int i = 0;

    for (; i + 128 <= src_h; i += 128) {
        int j = 0;
        for (; j + 128 <= src_w; j += 256) {
            int addr0 = j * src_h + i * 128;
            int addr1 = (j + 128) * src_h + i * 128;
            int st = 1;
            SIM_X86::tensor t0 = tensor_slice(src, addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, addr1 / 32);
            float8_128 data_a[16];
            float8_128 data_b[16];

            data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_start(data_a[0], 128, 0);
            data_b[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_start(data_b[0], 128, 1);

            for (int ii = 1; ii < 15; ii++) {
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
                m_transpose_mid(data_a[i], 0);
                data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), 255);
                m_transpose_mid(data_b[i], 1);
            }

            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
            m_transpose_end(data_a[15], 0);
            data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 120 * st * 128 / 32), 255);
            m_transpose_end(data_b[15], 1);

            int store_addr0 = j * 128 + i * src_w;
            int store_addr1 = (j + 128) * 128 + i * src_w;
            int store_st = 1;

            int cur_h = min(src_w - j - 128, 128);
            int kS = (cur_h + 7) / 8;
            // #pragma unroll
            for (int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                store8_128_stride_stmk(i * 8 * store_st * 128 / 32, store_st,
                                       tensor_slice(dst, (store_addr0) / 32), x0, 255);
                float8_128 x1 = m_pop_trf(1);
                store8_128_stride_stmk(i * 8 * store_st * 128 / 32, store_st,
                                       tensor_slice(dst, (store_addr1) / 32), x1, (1 << cur_sth) - 1);
            }
            // #pragma unroll
            for (int i = kS; i < 16; i++) {
                __attribute__((unused)) float8_128 x1 = m_pop_trf(1);
                float8_128 x = m_pop_trf(0);
                store8_128_stride_stmk(i * 8 * store_st * 128 / 32, store_st,
                                       tensor_slice(dst, (store_addr0) / 32), x, 255);
            }
        }
        int tail_j = src_w - j;
        if (tail_j <= 0)
            continue;
        int addr0 = j * src_h + i * 128;
        int st = 1;
        SIM_X86::tensor t0 = tensor_slice(src, addr0 / 32);
        float8_128 data_a[16];

        data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
        m_transpose_start(data_a[0], 128, 0);

        for (int ii = 1; ii < 15; ii++) {
            int _i = ii * 8;
            data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
            m_transpose_mid(data_a[i], 0);
        }

        data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
        m_transpose_end(data_a[15], 0);

        int store_addr0 = j * 128 + i * src_w;
        int store_st = 1;

        int cur_h = min(src_w - j, 128);
        int kS = (cur_h + 7) / 8;
        // #pragma unroll
        for (int i = 0; i < kS; i++) {
            int cur_sth = min(cur_h - i * 8, 8);
            float8_128 x0 = m_pop_trf(0);
            store8_128_stride_stmk(i * 8 * store_st * 128 / 32, store_st,
                                   tensor_slice(dst, (store_addr0) / 32), x0, (1 << cur_sth) - 1);
        }
        // #pragma unroll
        for (int i = kS; i < 16; i++) {
            __attribute__((unused)) float8_128 x = m_pop_trf(0);
        }
    }
    if (src_h - i <= 0)
        return;
    for (int j = 0; j < src_w; j += 128) {
        float128_128 v = loadh_k_T2(src, paddingH, src_h, paddingW, src_w, 0, i / 128, j / 128);
        store128_128_ex2(dst, src_w, src_h, j, i, v);
    }
}

//1.算出最大aw
//2.根据aw算出ah和bw
//3.根据ah和bw、输出的vmemsize, 来调整ah和bw
//如果aw != AW, bw != BW,则aw和bw就必须为128的倍数; 否则,不行
// vmemA ≥ 128 * 128 && vmemB ≥ 128 * 128 && vmemC ≥ 128 * 128 
//****较为粗糙，需要优化****
inline void CalVmemSizeBlock(int AH, int AW, int BW, int* ah, int* aw, int* bw, int vmemA, int vmemB, int vmemC, int flag){
    AW = ALIGN128(AW);
    AH = ALIGN128(AH);
    BW = ALIGN128(BW);
    //1.算出最大aw,优先尽可能让aw大
    int maxc = vmemB / 128; //rhs至少为128列
    int maxa = vmemA / 128; //lhs至少为128行
    *aw = min(min(maxc, AW), maxa);
    //不能用向上取成128的倍数，因为有可能会让ah和bw小于128
    int aw128 = (*aw) & 0xffffff80; 
    //2.根据aw算出ah和bw,其次是bw
    int maxah = CalcVMemBlockSizeMatrix(AH, aw128, vmemA);
    *ah = min(AH, maxah);
    //对于需要rhs转置的matmul_Transpose来说，aw需要是128的倍数
    int maxbw = CalcVMemBlockSizeMatrixW(BW, aw128, vmemB);
    *bw = min(BW, maxbw);
    int bw128 = ALIGN128(*bw); 
    //到这，ah和bw都是≥128,这里会先调整ah，如有需要会再调整bw
    if((*ah) * bw128 > vmemC){
        maxbw = (vmemC / 128) & 0xffffff80;
        (*bw) = min(*bw, maxbw);
        bw128 = ALIGN128(*bw); 
        (*ah) = min(soft_sdiv(vmemC, bw128), *ah);
    }

    *bw = (*bw) & 0xffffff80;
    *aw = (*aw) & 0xffffff80;
    *ah = (*ah) & 0xffffff80;
}

inline void HBMtoVMEM(SIM_X86::tensor hbm, SIM_X86::tensor vmem, int H, int W, int src_stride, int dst_stride){
    int len = H * 128;
    for(int i = 0; i < ALIGN128(W); i += 128){
        #ifdef USE_CMEM
            int sync = dlc_dma(hbm + i / 32, CMEM, vmem + i / 32, VMEM, len, src_stride, dst_stride, 128, 7);
        #else
            int sync = dlc_dma(hbm + i / 32, HBM, vmem + i / 32, VMEM, len, src_stride, dst_stride, 128, 7);
        #endif
        dlc_sync(sync);
    } 
    // #ifdef USE_CMEM  
    //     int sync = dlc_dma(hbm, CMEM, vmem, VMEM, 0, src_stride, 128, 128, 7);
    //     dlc_sync(sync);
    // #else
    //     int sync = dlc_dma(hbm, HBM, vmem, VMEM, 0, src_stride, 128, 128, 7);
    //     dlc_sync(sync);
    // #endif  
}

inline void CMEMtoVMEM(SIM_X86::tensor hbm, SIM_X86::tensor vmem, int H, int W, int src_stride, int dst_stride){
    int len = H * 128;
    for(int i = 0; i < ALIGN128(W); i += 128){
        int sync = dlc_dma(hbm + i / 32, CMEM, vmem + i / 32, VMEM, len, src_stride, dst_stride, 128, 7);
        dlc_sync(sync);
    }  
}

inline void VMEMtoHBM2(SIM_X86::tensor vmem, SIM_X86::tensor hbm, int H, int W, int dst_stride) {
    int len = H * 128;
    for (int i = 0; i < ALIGN128(W); i += 128) {
#ifdef USE_CMEM
        int sync = dlc_dma(vmem + i * H / 32, VMEM, hbm + i / 32, CMEM, len, 128, dst_stride, 128, 7);
#else
        int sync = dlc_dma(vmem + i * H / 32, VMEM, hbm + i / 32, HBM, len, 128, dst_stride, 128, 7);
#endif
        dlc_sync(sync);
    }
// #ifdef USE_CMEM 
//     int sync = dlc_dma(vmem, VMEM, hbm, CMEM, 0, 128, dst_stride, 128, 7);
//     dlc_sync(sync);
// #else
//     int sync = dlc_dma(vmem, VMEM, hbm, HBM, 0, 128, dst_stride, 128, 7);
//     dlc_sync(sync);
// #endif 
}

inline void VMEMtoCMEM2(SIM_X86::tensor vmem, SIM_X86::tensor hbm, int H, int W, int dst_stride) {
    int len = H * 128;
    for (int i = 0; i < ALIGN128(W); i += 128) {
        int sync = dlc_dma(vmem + i * H / 32, VMEM, hbm + i / 32, CMEM, len, 128, dst_stride, 128, 7);
        dlc_sync(sync);
    }
}

inline void HBMtoVMEM2(SIM_X86::tensor hbm, SIM_X86::tensor vmem, int H, int W, int src_stride){
    int len = H * 128;
    for(int i = 0; i < ALIGN128(W); i += 128){
        #ifdef USE_CMEM
            int sync = dlc_dma(hbm + i / 32, CMEM, vmem + i * H / 32, VMEM, len, src_stride, 128, 128, 7);
        #else
            int sync = dlc_dma(hbm + i / 32, HBM, vmem + i * H / 32, VMEM, len, src_stride, 128, 128, 7);
        #endif
        dlc_sync(sync);
    }    
}

inline void VMEMtoHBM(SIM_X86::tensor vmem, SIM_X86::tensor hbm, int H, int W, int src_stride, int dst_stride){
    int len = H * 128;
    for(int i = 0; i < ALIGN128(W); i += 128){
        #ifdef USE_CMEM
            int sync = dlc_dma(vmem + i / 32, VMEM, hbm + i / 32, CMEM, len, src_stride, dst_stride, 128, 7);
        #else
            int sync = dlc_dma(vmem + i / 32, VMEM, hbm + i / 32, HBM, len, src_stride, dst_stride, 128, 7);
        #endif
        dlc_sync(sync);
    }   
    // #ifdef USE_CMEM 
    //     int sync = dlc_dma(vmem, VMEM, hbm, CMEM, 0, src_stride, dst_stride, 128, 7);
    //     dlc_sync(sync);
    // #else
    //     int sync = dlc_dma(vmem, VMEM, hbm, HBM, 0, src_stride, dst_stride, 128, 7);
    //     dlc_sync(sync);
    // #endif 
}

inline int Calculate_length(unsigned* shape){
    return shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
}

inline void load_mat_level2_h(SIM_X86::tensor src, SIM_X86::tensor dst, const int dim0, const int dim1, const int dim2,
    const int dim3, const int idx0, const int idx1, const int idx2, const int idx3,
    const int vmemH, const int vmemW, const int src_stride) {
    // const int offset = i1h * b1h * mw + i1w * b1w + i2h * b2h * mw + i2w * b2w;
    const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    for (int i = 0; i < vmemW; i += 128) {
        dlc_sync(dlc_dma(tensor_slice(src, (offset + i) / 32), HBM, tensor_slice(dst, i * vmemH / 32), VMEM,
            vmemH * 128, src_stride, 128, 128, 7));
    }
}

inline void load_mat_0123_h(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
    int idx2, int idx3, int vmemH, int vmemW) {
    load_mat_level2_h(src, dst, dim0, dim1, dim2, dim3, idx0, idx1, idx2, idx3, vmemH, vmemW, dim3);
}

inline void store_mat_level2_h(SIM_X86::tensor src, SIM_X86::tensor dst, const int dim0, const int dim1, const int dim2,
    const int dim3, const int idx0, const int idx1, const int idx2, const int idx3,
    const int vmemH, const int vmemW, const int dst_stride) {
    const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    int len = vmemH * 128;
    for (int i = 0; i < ALIGN128(vmemW); i += 128) {
        dlc_sync(dlc_dma(src + i * vmemH / 32, VMEM, dst + (i + offset) / 32, HBM, len, 128, dst_stride, 128, 7));
    }
}

inline void store_mat_0123_h(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
    int idx2, int idx3, int vmemH, int vmemW) {
    store_mat_level2_h(src, dst, dim0, dim1, dim2, dim3, idx0, idx1, idx2, idx3, vmemH, vmemW, dim3);
}

inline void VMEMtoVMEM2(SIM_X86::tensor vmem, SIM_X86::tensor hbm, int H, int W, int dst_stride) {
    int len = H * 128;
    for (int i = 0; i < ALIGN128(W); i += 128) {
        dlc_dma(vmem + i * H / 32, VMEM, hbm + i / 32, VMEM, len, 128, dst_stride, 128, 7);
    }
    int sync = dlc_dma(vmem, VMEM, hbm, VMEM, 0, 128, dst_stride, 128, 7);
    dlc_sync(sync);
}

//matmul_rhsT_bf16

//1.算出最大aw
//2.根据aw算出ah和bw
//3.根据ah和bw、输出的vmemsize, 来调整ah和bw
//如果aw != AW, bw != BW,则aw和bw就必须为128的倍数; 否则,不行
inline void CalVmemSizeBlock_bf16(int AH, int AW, int BW, int* ah, int* aw, int* bw, int vmemA, int vmemB, int vmemC){
   int maxc = vmemB / 256; //rhs至少为128列
   *aw = (maxc > AW) ? AW : maxc;
   int aw128 = ALIGN128(*aw);
   int maxah = CalcVMemBlockSizeMatrix(AH, aw128, vmemA);
   *ah = (maxah > AH) ? AH : maxah;
   int maxbw = CalcVMemBlockSizeMatrixW(BW, aw128, vmemB);
   *bw = (maxbw > BW) ? BW : maxbw;
   int bw256 = ALIGN256(*bw);
   maxah = CalcVMemBlockSizeMatrix(*ah, bw256, vmemC);
   *ah = min(maxah, *ah);
   maxbw = CalcVMemBlockSizeMatrixW(*bw, *ah, vmemC);
   *bw = min(maxbw, *bw);
   if((*bw) != BW ){
     *bw = (*bw) & 0xffffff00;
   }
   if((*aw) != AW ){
    *aw = (*aw) & 0xffffff80;
   }
}

// matmul lhsT 2pgx bf16
inline void Vmem2CMEM(SIM_X86::tensor src, SIM_X86::tensor dst, const int H, const int W){
    int sync = dlc_dma(src, VMEM, dst, CMEM, H * W, 128, 128, 128, 7);
    dlc_sync(sync);
}


inline void load_mat_level2_h_cmem(SIM_X86::tensor src, SIM_X86::tensor dst, const int dim0, const int dim1, const int dim2,
    const int dim3, const int idx0, const int idx1, const int idx2, const int idx3,
    const int vmemH, const int vmemW, const int src_stride) {
    // const int offset = i1h * b1h * mw + i1w * b1w + i2h * b2h * mw + i2w * b2w;
    const int offset = idx0 * dim1 * dim2 * dim3 + idx1 * dim2 * dim3 + idx2 * dim3 + idx3;
    for (int i = 0; i < vmemW; i += 128) {
        dlc_dma(tensor_slice(src, (offset + i) / 32), CMEM, tensor_slice(dst, i * vmemH / 32), VMEM,
            vmemH * 128, src_stride, 128, 128, 7);
    }
}

inline void load_mat_0123_h_cmem(SIM_X86::tensor src, SIM_X86::tensor dst, int dim0, int dim1, int dim2, int dim3, int idx0, int idx1,
    int idx2, int idx3, int vmemH, int vmemW) {
    load_mat_level2_h_cmem(src, dst, dim0, dim1, dim2, dim3, idx0, idx1, idx2, idx3, vmemH, vmemW, dim3);
}

inline void set_permute_for_tsps_bf16() {
    int8_128 coreid = get_core_id();
    int8_128 permute_odd = coreid * 2;
    int8_128 permute_even = permute_odd + 1;
    m_set_permute(permute_odd, 0);
    m_set_permute(permute_even, 1);
}

// 在调用该函数前面set permute
// int8_128 coreid = get_core_id();
// int8_128 permute_odd = coreid * 2;
// int8_128 permute_even = permute_odd + 1;
// m_set_permute(permute_odd, 0);
// m_set_permute(permute_even, 1);
inline void tile_trans_transfer_bf16(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr, int stride_src, int stride_dst) {
    int i = 0;
    for (; i + 256 <= src_h; i += 256) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr1 = (i + 128) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            data1[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_packed_start(data1[0], 128, 1);
            // Print("The first vector loaded 0:%h\n", data0[0]);
            // Print("The first vector loaded 1:%h\n", data1[0]);
            for (int index = 1; index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data1[index], 1);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data1[7], 1);
            // Print("The last vector loaded 0:%h\n", data0[7]);
            // Print("The last vector loaded 1:%h\n", data1[7]);

            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int addr2 = (i + 64) * stride_src + j / 2;
            int addr3 = (i + 192) * stride_src + j / 2;
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);
            SIM_X86::tensor t3 = tensor_slice(src, srcaddr / 32 + addr3 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            float8_128 __attribute__((address_space(2))) data3[8];
            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);
            data3[0] = load8_128_stride_ldmk(0, st, t3, 255);
            m_transpose_packed_start(data3[0], 128, 1);
            // Print("The first vector loaded 0:%h\n", data2[0]);
            // Print("The first vector loaded 1:%h\n", data3[0]);
            for(int index = 1; index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data2[index], 0);
                data3[index] = load8_128_stride_ldmk(0, st, tensor_slice(t3, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data3[index], 1);
            }

            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data2[7], 0);
            data3[7] = load8_128_stride_ldmk(0, st, tensor_slice(t3, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data3[7], 1);

            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);

                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res23[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);

                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
    int rest_h = src_h - i;
    if(rest_h == 0) return ;
    if(rest_h <= 64) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int st = stride_src / 128;
            int push_num = (rest_h + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            int cur_sth = min(rest_h, 8);
            data0[0] = load8_128_stride_ldmk(0, st, t0, (1 << cur_sth) - 1);

            m_transpose_packed_start(data0[0], 128, 0);
        
            for(int index = 1; index < push_num - 1; index++) {
                cur_sth = min(rest_h - index * 8, 8);
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data0[index], 0);
            }

            cur_sth = min(rest_h - (push_num - 1) * 8, 8);
            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data0[7], 0);
            
            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                x0 = __$F(__$S(x0) << 16);
                float8_128 zero = v_u32_move_b(0);
                res01[index] = __$F(float_to_bfloat16(zero, x0));
            }

            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            
            for(int index = 0; index < store_num; index++) {
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                int cur_sth = min(cur_h - index * 8, 8);
                Print("i:%d\n", i);
                Print("j:%d\n", j);
                Print("index:%d\n", index);
                Print("addr:%d\n", index * 1024 + dstaddr + store_addr);
                Print("up:%h\n", up);
                
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth2) - 1);
            }
        }
    } else if(rest_h <= 128) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr2 = (i + 64) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data2[8];
            int push_num = (rest_h - 64 + 7) / 8;
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            int cur_sth = min(rest_h - 64, 8);
            data2[0] = load8_128_stride_ldmk(0, st, t2, (1 << cur_sth) - 1);
            m_transpose_packed_start(data2[0], 128, 1);

            for(int index = 1; index < push_num - 1; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                cur_sth = min(rest_h - 64 - index * 8, 8);
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data2[index], 1);
            }

            for(int index = max(push_num - 1, 1); index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            cur_sth = min(rest_h - 64 - (push_num - 1) * 8, 8);
            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data2[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                float8_128 zero = v_u32_move_b(0);
                res01[index] = __$F(float_to_bfloat16(zero, x0));
                res23[index] = __$F(float_to_bfloat16(zero, x1));
            }
            
            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {

                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    } else if(rest_h <= 192) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr1 = (i + 128) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            int cur_sth = min(rest_h - 128, 8);
            data1[0] = load8_128_stride_ldmk(0, st, t1, (1 << cur_sth) - 1);
            int push_num = (rest_h - 128 + 7) / 8;

            m_transpose_packed_start(data1[0], 128, 1);

            for(int index = 1; index < push_num - 1; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                cur_sth = min(rest_h - 128 - index * 8, 8);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data1[index], 1);
            }

            for(int index = max(push_num - 1, 1); index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
            }
            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            cur_sth = min(rest_h - 128 - (push_num - 1) * 8, 8);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data1[7], 1);
            
            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int addr2 = (i + 64) * stride_src + j / 2;
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);

            for(int index = 1; index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data2[index], 0);
            }

            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data2[7], 0);

            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                x0 = __$F(__$S(x0) << 16);
                float8_128 zero = v_u32_move_b(0);
                res23[index] = __$F(float_to_bfloat16(zero, x0));
            }
            
            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    } else if(rest_h < 256) {
        for(int j = 0; j < src_w; j += 256) {
            int addr0 = i * stride_src + j / 2;
            int addr1 = (i + 128) * stride_src + j / 2;
            int st = stride_src / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            data1[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_packed_start(data1[0], 128, 1);

            for(int index = 1; index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data1[index], 1);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data0[7], 0);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data1[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int addr2 = (i + 64) * stride_src + j / 2;
            int addr3 = (i + 192) * stride_src + j / 2;
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);
            SIM_X86::tensor t3 = tensor_slice(src, srcaddr / 32 + addr3 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            float8_128 __attribute__((address_space(2))) data3[8];
            int push_num = (rest_h - 192 + 7) / 8;

            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);
            int cur_sth = min(rest_h - 192, 8);
            data3[0] = load8_128_stride_ldmk(0, st, t3, (1 << cur_sth) - 1);
            m_transpose_packed_start(data3[0], 128, 1);

            for(int index = 1; index < push_num - 1; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data2[index], 0);
                cur_sth = min(rest_h - 192 - index * 8, 8);
                data3[index] = load8_128_stride_ldmk(0, st, tensor_slice(t3, index * 8 * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data3[index], 1);
            }

            for(int index = max(push_num - 1, 1); index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 8 * st * 128 / 32), 255);
                m_transpose_packed_mid(data2[index], 0);
            }
            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 8 * st * 128 / 32), 255);
            m_transpose_packed_end(data2[7], 0);
            cur_sth = min(rest_h - 192 - (push_num - 1) * 8, 8);
            data3[7] = load8_128_stride_ldmk(0, st, tensor_slice(t3, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data3[7], 1);            

            float8_128 __attribute__((address_space(2))) res23[16];
            for(int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);

                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res23[index] = __$F(float_to_bfloat16(x1, x0));
            }
            
            int store_addr = j * stride_dst + i / 2;
            int store_st = stride_dst / 128;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for(int index = 0; index < store_num; index++) {
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(8 * index * stride_dst / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((8 * index * stride_dst + 128 * stride_dst) / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
}

inline void tile_trans_transfer_f32_to_bf16(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr) {
    int paddingH = ALIGN256(src_h) / 2;
    int paddingW = ALIGN128(src_w);
    int i = 0;
    for (; i + 256 <= src_h; i += 256) {
        for (int j = 0; j < src_w; j += 128) {
            int addr0 = j + i * paddingW;
            int addr1 = j + (i + 128) * paddingW;
            int st = paddingW / 128;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);
            float8_128 data_a[16];
            float8_128 data_b[16];
            
            data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_start(data_a[0], 128, 0);
            data_b[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_start(data_b[0], 128, 1);
            
            for(int ii = 1; ii < 15; ii++) {
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
                m_transpose_mid(data_a[i], 0);
                data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), 255);
                m_transpose_mid(data_b[i], 1);
            }
        
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
            m_transpose_end(data_a[15], 0);
            data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 120 * st * 128 / 32), 255);
            m_transpose_end(data_b[15], 1);
            
            int store_addr = j * paddingH + i / 2;
            int store_st = paddingH / 128;
            
            int cur_h = min(src_w - j, 128);
            int kS = (cur_h + 7) / 8;
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                float8_128 res = __$F(float_to_bfloat16(x1, x0));
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), res, (1 << cur_sth) - 1);
            }
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x0 = m_pop_trf(0);
                __attribute__((unused))float8_128 x1 = m_pop_trf(1);
            }
        }
    }
    int rest_h = src_h - i;
    if(rest_h == 0) return ;
    if(rest_h <= 128) {
        for (int j = 0; j < src_w; j += 128) {
            int addr0 = j + i * paddingW;
            int st = paddingW / 128;
            int push_num = (rest_h + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            float8_128 data_a[16];
            
            int cur_sth = min(rest_h, 8);
            data_a[0] = load8_128_stride_ldmk(0, st, t0, (1 << cur_sth) - 1);
            m_transpose_start(data_a[0], 128, 0);
            
            for(int ii = 1; ii < push_num - 1; ii++) {
                cur_sth = min(rest_h - ii * 8, 8);
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_mid(data_a[i], 0);
            }
        
            cur_sth = min(rest_h - (push_num - 1) * 8, 8);
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 8 * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_end(data_a[15], 0);
            
            int store_addr = j * paddingH + i / 2;
            int store_st = paddingH / 128;
            
            int cur_h = min(src_w - j, 128);
            int kS = (cur_h + 7) / 8;
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = v_u32_move_b(0);
                float8_128 res = __$F(float_to_bfloat16(x1, x0));
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), res, (1 << cur_sth) - 1);
            }
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x0 = m_pop_trf(0);
            }
        }
    } else {
        for (int j = 0; j < src_w; j += 128) {
            int addr0 = j + i * paddingW;
            int addr1 = j + (i + 128) * paddingW;
            int st = paddingW / 128;
            int push_num = (rest_h - 128 + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);
            float8_128 data_a[16];
            float8_128 data_b[16];
            
            data_a[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_start(data_a[0], 128, 0);
            int cur_sth = min(rest_h - 128, 8);
            data_b[0] = load8_128_stride_ldmk(0, st, t1, (1 << cur_sth) - 1);
            m_transpose_start(data_b[0], 128, 1);
            
            for(int ii = 1; ii < 15; ii++) {
                int _i = ii * 8;
                data_a[i] = load8_128_stride_ldmk(0, st, tensor_slice(t0, _i * st * 128 / 32), 255);
                m_transpose_mid(data_a[i], 0);
                cur_sth = min(rest_h - 128 - ii * 8, 8);
                data_b[i] = load8_128_stride_ldmk(0, st, tensor_slice(t1, _i * st * 128 / 32), (1 << cur_sth) - 1);
                m_transpose_mid(data_b[i], 1);
            }
            
            data_a[15] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 120 * st * 128 / 32), 255);
            m_transpose_end(data_a[15], 0);
            cur_sth = min(rest_h - 128 - (push_num - 1) * 8, 8);
            data_b[15] = load8_128_stride_ldmk(0, st, tensor_slice(t1, (push_num - 1) * st * 128 / 32), (1 << cur_sth) - 1);
            m_transpose_end(data_b[15], 1);
            
            int store_addr = j * paddingH + i / 2;
            int store_st = paddingH / 128;
            
            int cur_h = min(src_w - j, 128);
            int kS = (cur_h + 7) / 8;
            for(int i = 0; i < kS; i++) {
                int cur_sth = min(cur_h - i * 8, 8);
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                float8_128 res = __$F(float_to_bfloat16(x1, x0));
                store8_128_stride_stmk(i * 8 * paddingH / 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), res, (1 << cur_sth) - 1);
            }
            for(int i = kS; i < 16; i++) {
                __attribute__((unused))float8_128 x0 = m_pop_trf(0);
                __attribute__((unused))float8_128 x1 = m_pop_trf(1);
            }
        }
    }
}
#endif
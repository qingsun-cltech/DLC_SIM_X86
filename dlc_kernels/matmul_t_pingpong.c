// #include "ldst.h"
// #include "typehint.h"
// #include "nn.h"
// #include "kernel_arg_types.h"
// #include "chunk.h"
// #include "pingpong.h"

#include "../x86.h"

enum { UNDONE = 0, DONE = 1 };

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

inline void VMEMtoHBM2(SIM_X86::tensor vmem, SIM_X86::tensor hbm, int H, int W, int dst_stride) {
    int len = H * 128;
    for (int i = 0; i < ALIGN128(W); i += 128) {
#ifdef USE_CMEM
        dlc_dma(vmem + i * H / 32, VMEM, hbm + i / 32, CMEM, len, 128, dst_stride, 128, 7);
#else
        dlc_dma(vmem + i * H / 32, VMEM, hbm + i / 32, HBM, len, 128, dst_stride, 128, 7);
#endif
        // dlc_sync(sync);
    }
#ifdef USE_CMEM 
    int sync = dlc_dma(vmem, VMEM, hbm, CMEM, 0, 128, dst_stride, 128, 7);
    dlc_sync(sync);
#else
    int sync = dlc_dma(vmem, VMEM, hbm, HBM, 0, 128, dst_stride, 128, 7);
    dlc_sync(sync);
#endif 
}

inline void HBMtoVMEM2(SIM_X86::tensor hbm, SIM_X86::tensor vmem, int H, int W, int src_stride){
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
        int sync = dlc_dma(hbm, CMEM, vmem, VMEM, 0, src_stride, 128, 128, 7);
        dlc_sync(sync);
    #else
        int sync = dlc_dma(hbm, HBM, vmem, VMEM, 0, src_stride, 128, 128, 7);
        dlc_sync(sync);
    #endif  
}

inline void CMEMtoVMEM2(SIM_X86::tensor cmem, SIM_X86::tensor vmem, int H, int W, int src_stride){
    int len = H * 128;
    for(int i = 0; i < ALIGN128(W); i += 128){
        dlc_dma(cmem + i / 32, CMEM, vmem + i * H / 32, VMEM, len, src_stride, 128, 128, 7);
    }   
    int sync = dlc_dma(cmem, CMEM, vmem, VMEM, 0, src_stride, 128, 128, 7);
    dlc_sync(sync);
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
//input、weight、out:hbm
//分批会有问题，因为input需要转置，但是out和rhs的数据需要放在vmem中
inline void matmul_lhsT_2xys(SIM_X86::tensor input_hbm, SIM_X86::tensor mat2_hbm, SIM_X86::tensor out_hbm, SIM_X86::tensor vmem,
                     int AH, int AW, int BW, int ah, int aw, int bw)
{
    //通过get_device_id()可以得到当前使用的xys的下标，一块芯片有两个xys，所以下标为0,1
    int device_id = get_device_id();

    SIM_X86::tensor input = vmem;
    SIM_X86::tensor mat2 = input + ah * ALIGN128(aw) / 32;
    SIM_X86::tensor out = mat2 + aw * ALIGN128(bw) / 32;
    SIM_X86::tensor tmp = out + ah * ALIGN128(bw) / 32;

    //参数
    int BW128 = ALIGN128(BW), AH128 = ALIGN128(AH);

    int process_ah, process_aw, process_bw;
    int i0_offset = 0, i1_offset = 0, o_offset = 0;
    int curi0_offset = -1, curi1_offset = -1, curo_offset = -1;
    int AH_XYS = AH;
    int AH_offset = 0;
    //双xys拆分AH
    if(AH > 128){
        AH_XYS = ALIGN128(AH / 2);
        AH_offset = 0;
    }
    if(device_id == 1){
        AH_offset = AH_XYS;
        AH_XYS = AH - AH_XYS; 
    }
    // if (get_device_id()) {
        // printf("AW128 = %d\n", AW128);
        // printf("BW128 = %d\n", BW128);
    // }
    // printf("[XYS%d]: AH_XYS = %d\n", get_device_id(), AH_XYS);
    // printf("[XYS%d]: BW = %d\n", get_device_id(), BW);
    // printf("[XYS%d]: AW = %d\n", get_device_id(), AW);

    input_hbm = input_hbm + AH_offset / 32;
    out_hbm = out_hbm + AH_offset * BW128 / 32;
    for(int i=0; i < AH_XYS; i+= ah){
        for(int j = 0; j < BW; j += bw){
            o_offset = i * BW128 + j;
            for(int k = 0; k < AW; k+= aw){
    // for(int i=0; i < 1; i+= ah){
    //     for(int j = 0; j < 1; j += bw){
    //         o_offset = i * BW128 + j;
    //         for(int k = 0; k < 1; k+= aw){
                process_ah = min(AH_XYS - i, ah);
                process_aw = min(AW - k, aw);
                process_bw = min(BW - j, bw);

                i0_offset = k * AH128 + i;
                i1_offset = k * BW128 + j;
                if(curi0_offset != i0_offset){
                    HBMtoVMEM2(input_hbm + i0_offset / 32, tmp, process_aw, process_ah, AH128);
                    if (get_device_id()) {
                        // Print("input_hbm = ", input_hbm + i0_offset / 32, 1024, PrintType::FLOAT);
                        // Print("tmp = ", tmp, 1024, PrintType::FLOAT);
                    }
                    tile_trans_transfer2(tmp, input, process_aw, process_ah);
                    curi0_offset = i0_offset;
                }
                if(curi1_offset != i1_offset){
                    HBMtoVMEM2(mat2_hbm + i1_offset / 32, mat2, process_aw, process_bw, BW128);
                    if (get_device_id()) {
                        // Print("mat2_hbm = ", mat2_hbm + i1_offset / 32, 1024, PrintType::FLOAT);
                        // Print("mat2 = ", mat2, 1024, PrintType::FLOAT);
                    }
                    curi1_offset = i1_offset;
                }
                
                matmul_all_f32(input, mat2, out, process_ah, process_aw, process_bw, k, 1.0, 1.0);
            }
            if(curo_offset != o_offset){
                VMEMtoHBM2(out, out_hbm + o_offset / 32, process_ah, process_bw, BW128);
                if (get_device_id()) {
                    // Print("out = ", out, 128, PrintType::FLOAT);
                    // Print("out_hbm = ", out_hbm + o_offset / 32, 1024, PrintType::FLOAT);
                }
                curo_offset = o_offset;
            }
        }
    }
    //等待两个xys都结束任务后退出
    sync_device();
}

//input、weight、out:hbm
inline void matmul_rhsT_2xys(SIM_X86::tensor input_hbm, SIM_X86::tensor mat2_hbm, SIM_X86::tensor out_hbm, SIM_X86::tensor vmem,
                     int AH, int AW, int BW, int ah, int aw, int bw)
{
    //通过get_device_id()可以得到当前使用的xys的下标，一块芯片有两个xys，所以下标为0,1
    int device_id = get_device_id();

    SIM_X86::tensor input = vmem;
    SIM_X86::tensor mat2 = input + ah * ALIGN128(aw) / 32;
    SIM_X86::tensor out = mat2 + bw * ALIGN128(aw) / 32;

    //参数
    int AW128 = ALIGN128(AW), BW128 = ALIGN128(BW);
 
    int process_ah, process_aw, process_bw;
    int i0_offset = 0, i1_offset = 0, o_offset = 0;
    int curi0_offset = -1, curi1_offset = -1, curo_offset = -1;
    //双xys拆分AH
    int AH_XYS = AH / 2;
    int AH_offset = 0;
    if(device_id == 1){
        AH_offset = AH_XYS;
        AH_XYS = AH - AH_XYS; 
    }
    input_hbm = input_hbm + AH_offset * AW128 / 32;
    out_hbm = out_hbm + AH_offset * BW128 / 32;
    for(int i=0; i < AH_XYS; i+= ah){
        for(int j = 0; j < BW; j += bw){
            o_offset = i * BW128 + j;
            for(int k = 0; k < AW; k+= aw){
                process_ah = min(AH_XYS - i, ah);
                process_aw = min(AW - k, aw);
                process_bw = min(BW - j, bw);

                i0_offset = i * AW128 + k;
                i1_offset = j * AW128 + k;
                if(curi0_offset != i0_offset){
                    HBMtoVMEM2(input_hbm + i0_offset / 32, input, process_ah, process_aw, AW128);
                    curi0_offset = i0_offset;
                }
                if(curi1_offset != i1_offset){
                    HBMtoVMEM2(mat2_hbm + i1_offset / 32, mat2, process_bw, process_aw, AW128);
                    curi1_offset = i1_offset;
                }
                matmul_all_f32_RHST(input, mat2, out, process_ah, process_aw, process_bw, k, 1.0, 1.0);
            }
            if(curo_offset != o_offset){
                // VMEMtoHBM(out, out_hbm + o_offset / 32, process_ah, process_bw, ALIGN128(process_bw), BW128);
                VMEMtoHBM2(out, out_hbm + o_offset / 32, process_ah, process_bw, BW128);
                curo_offset = o_offset;
            }
        }
    }
    //等待两个xys都结束任务后退出
    sync_device();
}

//input、weight、out:hbm
inline void matmul_resT_2xys(SIM_X86::tensor input_hbm, SIM_X86::tensor mat2_hbm, SIM_X86::tensor out_hbm, SIM_X86::tensor vmem,
                     int AH, int AW, int BW, int ah, int aw, int bw)
{
    //通过get_device_id()可以得到当前使用的xys的下标，一块芯片有两个xys，所以下标为0,1
    int device_id = get_device_id();

    SIM_X86::tensor input = vmem;
    SIM_X86::tensor mat2 = input + ah * ALIGN128(aw) / 32;
    SIM_X86::tensor out = mat2 + aw * ALIGN128(bw) / 32;
    SIM_X86::tensor tmp = out + ah * ALIGN128(bw) / 32;

    //参数
    int AW128 = ALIGN128(AW), BW128 = ALIGN128(BW), AH128 = ALIGN128(AH);

    int process_ah, process_aw, process_bw;
    int i0_offset = 0, i1_offset = 0, o_offset = 0;
    int curi0_offset = -1, curi1_offset = -1, curo_offset = -1;
    int AH_XYS = AH;
    int AH_offset = 0;
    //双xys拆分AH
    if(AH > 128){
        AH_XYS = ALIGN128(AH / 2);
        AH_offset = 0;
    }
    if(device_id == 1){
        AH_offset = AH_XYS;
        AH_XYS = AH - AH_XYS; 
    }
    input_hbm = input_hbm + AH_offset * AW128 / 32;
    out_hbm = out_hbm + AH_offset / 32;
    for(int i=0; i < AH_XYS; i+= ah){
        for(int j = 0; j < BW; j += bw){
            o_offset = j * AH128 + i;
            for(int k = 0; k < AW; k+= aw){
                process_ah = min(AH_XYS - i, ah);
                process_aw = min(AW - k, aw);
                process_bw = min(BW - j, bw);
            
                i0_offset = i * AW128 + k;
                i1_offset = k * BW128 + j;
                if(curi0_offset != i0_offset){
                    HBMtoVMEM2(input_hbm + i0_offset / 32, input, process_ah, process_aw, AW128);
                    curi0_offset = i0_offset;
                }
                
                if(curi1_offset != i1_offset){
                    HBMtoVMEM2(mat2_hbm + i1_offset / 32, mat2, process_aw, process_bw, BW128);
                    curi1_offset = i1_offset;
                }
                
                matmul_all_f32(input, mat2, out, process_ah, process_aw, process_bw, k, 1.0, 1.0);
            }
            if(curo_offset != o_offset){
                tile_trans_transfer2(out, tmp, process_ah, process_bw);
                VMEMtoHBM2(tmp, out_hbm + o_offset / 32, process_bw, process_ah, AH128);
                curo_offset = o_offset;
            }
        }
    }
    //等待两个xys都结束任务后退出
    sync_device();
}

//因为ping pong buffer需要在做上一行最后一次add时，把下一次的数据load到cmem，如果要将res也存到cmem，然后用dma传的话
//如果数据量太大，会导致cmem不够用，如果cmem够用的话，可以先用dma stride传回到cmem，再从cmem传到hbm
inline void matmul_2xys_cmem(SIM_X86::tensor input_hbm, SIM_X86::tensor mat2_hbm, SIM_X86::tensor out_hbm, SIM_X86::tensor vmem, SIM_X86::tensor cmem,
                     int AH, int AW, int BW, int ah, int aw, int bw){
    //通过get_device_id()可以得到当前使用的xys的下标，一块芯片有两个xys，所以下标为0,1
    int device_id = get_device_id();

    SIM_X86::tensor input_cur = vmem;
    SIM_X86::tensor input_next = input_cur + ah * ALIGN128(aw) / 32;
    SIM_X86::tensor mat2_cur = input_next + ah * ALIGN128(aw) / 32;
    SIM_X86::tensor mat2_next = mat2_cur + aw * ALIGN128(bw) / 32;
    SIM_X86::tensor out_cur = mat2_next + aw * ALIGN128(bw) / 32;
    //参数
    int AW128 = ALIGN128(AW), BW128 = ALIGN128(BW);

    int process_ah, process_aw, process_bw;

    //双xys拆分AH
    int AH_XYS = AH / 2;
    int AH_offset = 0;
    if(device_id == 1){
        AH_offset = AH_XYS;
        AH_XYS = AH - AH_XYS; 
    }
    input_hbm = input_hbm + AH_offset * AW128 / 32;
    out_hbm = out_hbm + AH_offset * BW128 / 32;

    int num_blocks_AH = soft_sdiv(AH_XYS + ah -1, ah);
    int num_blocks_BW = soft_sdiv(BW + bw - 1, bw);
    int num_blocks_AW = soft_sdiv(AW + aw - 1, aw);
    int prea = -1, preb = -1;
    int sync0 = DONE, sync1 = DONE, sync2 = DONE;
    cmem = cmem + device_id * 1020 * 4096 / 32;
    int next_process_ah = 0, next_process_aw = 0, next_process_bw = 0; 
    for(int i = 0; i < num_blocks_AH; i ++){
        for(int j = 0; j < num_blocks_BW; j ++){
            int Coffset = i * ah * BW128 + j * bw;
            for(int k = 0; k < num_blocks_AW; k ++){
                int idxa = i * num_blocks_AW + k;
                int idxb = k * num_blocks_BW + j;
                
                process_ah = min(AH_XYS - i * ah, ah);
                process_aw = min(AW - k * aw, aw);
                process_bw = min(BW - j * bw, bw);
                bool is_cmem = false;
                int cmem_addr = 0;
                int Aoffset = next_posa_cmem(AH_XYS, AW, i * ah, k * aw, j, num_blocks_BW, ah, aw, &next_process_ah, &next_process_aw,
                              &is_cmem, &cmem_addr);
                int Boffset = next_posb(AW, BW128, k * aw, j * bw, aw, bw, &next_process_bw);                    
                if(idxa == 0 && idxb == 0){
                    int sync5 = dlc_dma(input_hbm, HBM, cmem, CMEM, process_ah * AW128, 128, 128, 128, 7);
                    dlc_sync(sync5);
                    CMEMtoVMEM2(cmem, input_cur, process_ah, process_aw, AW128);
                    HBMtoVMEM2(mat2_hbm, mat2_cur, process_aw, process_bw, BW128);
                    sync1 = HBMtoVMEM2_nosync(mat2_hbm + Boffset, mat2_next, next_process_aw, next_process_bw, BW128);
                    if(cmem_addr != 0){
                        sync5 = dlc_dma(input_hbm + cmem_addr, HBM, cmem, CMEM, process_ah * AW128, 128, 128, 128, 7);
                        dlc_sync(sync5);
                    }
                    sync0 = CMEMtoVMEM_nosync(cmem + Aoffset, input_next, next_process_ah, next_process_aw, AW128);
                    prea = idxa, preb = idxb;
                }else{
                    HBM2VMEM_pingpong_cmem(input_hbm, cmem, &input_cur, &input_next,
                                    mat2_hbm, &mat2_cur, &mat2_next,
                                    idxa, idxb, &prea, &preb, 
                                    AW128, BW128, Aoffset, Boffset,
                                    next_process_ah, next_process_aw, next_process_bw,
                                    &sync0, &sync1, is_cmem, cmem_addr);   
                }
                matmul_all_f32(input_cur, mat2_cur, out_cur, process_ah, process_aw, process_bw, k, 1.0, 1.0);
            }
            VMEMtoHBM2(out_cur, out_hbm + Coffset / 32, process_ah, process_bw, BW128);
        }
    }
    dlc_sync(sync1);
    dlc_sync(sync0);

    //等待两个xys都结束任务后退出
    sync_device();
}


//input、weight、out:hbm
inline void matmul_2xys(SIM_X86::tensor input_hbm, SIM_X86::tensor mat2_hbm, SIM_X86::tensor out_hbm, SIM_X86::tensor vmem,
                     int AH, int AW, int BW, int ah, int aw, int bw)
{
    //通过get_device_id()可以得到当前使用的xys的下标，一块芯片有两个xys，所以下标为0,1
    int device_id = get_device_id();

    SIM_X86::tensor input_cur = vmem;
    SIM_X86::tensor input_next = input_cur + ah * ALIGN128(aw) / 32;
    SIM_X86::tensor mat2_cur = input_next + ah * ALIGN128(aw) / 32;
    SIM_X86::tensor mat2_next = mat2_cur + aw * ALIGN128(bw) / 32;
    SIM_X86::tensor out_cur = mat2_next + aw * ALIGN128(bw) / 32;
    //参数
    int AW128 = ALIGN128(AW), BW128 = ALIGN128(BW);

    int process_ah, process_aw, process_bw;

    //双xys拆分AH
    int AH_XYS = AH / 2;
    int AH_offset = 0;
    if(device_id == 1){
        AH_offset = AH_XYS;
        AH_XYS = AH - AH_XYS; 
    }
    input_hbm = input_hbm + AH_offset * AW128 / 32;
    out_hbm = out_hbm + AH_offset * BW128 / 32;

    int num_blocks_AH = soft_sdiv(AH_XYS + ah -1, ah);
    int num_blocks_BW = soft_sdiv(BW + bw - 1, bw);
    int num_blocks_AW = soft_sdiv(AW + aw - 1, aw);
    int prea = -1, preb = -1;
    int sync0 = DONE, sync1 = DONE;
    int next_process_ah = 0, next_process_aw = 0, next_process_bw = 0; 

    for(int i = 0; i < num_blocks_AH; i ++){
        for(int j = 0; j < num_blocks_BW; j ++){
            int Coffset = i * ah * BW128 + j * bw;
            process_ah = min(AH_XYS - i * ah, ah);
            for(int k = 0; k < num_blocks_AW; k ++){
                int idxa = i * num_blocks_AW + k;
                int idxb = k * num_blocks_BW + j;
                
                process_aw = min(AW - k * aw, aw);
                process_bw = min(BW - j * bw, bw);
                int Aoffset = next_posa(AH_XYS, AW, i * ah, k * aw, j, num_blocks_BW, ah, aw, &next_process_ah, &next_process_aw);
                int Boffset = next_posb(AW, BW128, k * aw, j * bw, aw, bw, &next_process_bw);                    

                if(idxa == 0 && idxb == 0){
                    HBMtoVMEM2(input_hbm, input_cur, process_ah, process_aw, AW128);  
                    HBMtoVMEM2(mat2_hbm, mat2_cur, process_aw, process_bw, BW128);
                    sync1 = HBMtoVMEM2_nosync(mat2_hbm + Boffset, mat2_next, next_process_aw, next_process_bw, BW128);
                    sync0 = HBMtoVMEM2_nosync(input_hbm + Aoffset, input_next, next_process_ah, next_process_aw, AW128);
                    prea = idxa, preb = idxb;
                }else{
                    HBM2VMEM_pingpong(input_hbm, &input_cur, &input_next,
                                    mat2_hbm, &mat2_cur, &mat2_next,
                                    idxa, idxb, &prea, &preb, 
                                    AW128, BW128, 
                                    Aoffset, Boffset,
                                    next_process_ah, next_process_aw, next_process_bw,
                                    &sync0, &sync1);   
                }
                matmul_all_f32(input_cur, mat2_cur, out_cur, process_ah, process_aw, process_bw, k, 1.0, 1.0);
            }
            VMEMtoHBM2(out_cur, out_hbm + Coffset / 32, process_ah, process_bw, BW128);
        }
    }
    dlc_sync(sync1);
    dlc_sync(sync0);


    //等待两个xys都结束任务后退出
    sync_device();
}

void main_x86(SIM_X86::DLCMem* mem,
              SIM_X86::DLCTensor* input_hbm_, SIM_X86::DLCTensor* mat2_hbm_,
              SIM_X86::DLCTensor* out_hbm_,
              int* ChunkBlock, bool* Is_Trans){
/*##AUTO-GEN##*/TensorFixDims(input_hbm_);TensorFixDims(mat2_hbm_);TensorFixDims(out_hbm_);/*##END-GEN##*/
    // bool input_T = Is_Trans[0].value;
    // bool mat2_T = Is_Trans[1].value;
    // bool out_T = Is_Trans[2].value;   
    bool input_T = Is_Trans[0];
    bool mat2_T = Is_Trans[1];
    bool out_T = Is_Trans[2];
    int flag = 0;
    if(input_T) flag |= 4;
    if(mat2_T) flag |= 2;
    if(out_T) flag |= 1;
    // printf("[XYS%d]: flag = %d\n", get_device_id(), flag);
    // return;
    SIM_X86::tensor lhs = *(SIM_X86::tensor*)input_hbm_->address;
    SIM_X86::tensor rhs = *(SIM_X86::tensor*)mat2_hbm_->address;
    SIM_X86::tensor out = *(SIM_X86::tensor*)out_hbm_->address;
    // int AH = input_hbm_->shape[1], AW = input_hbm_->shape[0], BW = mat2_hbm_->shape[0];
    // int ah = ChunkBlock[0].value, aw = ChunkBlock[1].value, bw = ChunkBlock[2].value; 
    int AH = input_hbm_->shape[1], AW = input_hbm_->shape[0], BW = mat2_hbm_->shape[0];
    int ah = ChunkBlock[0], aw = ChunkBlock[1], bw = ChunkBlock[2];
    // int AW128 = ALIGN128(AW), BW128 = ALIGN128(BW);
    if(flag == 0 || flag == 7){
        // bool lhs_is_cmem = ((AW128 % 256 == 0) && ((ah * AW128) <= 4096 * 1020));
        if(flag == 7){
            lhs = *(SIM_X86::tensor*)mat2_hbm_->address;
            rhs = *(SIM_X86::tensor*)input_hbm_->address;
            AH = mat2_hbm_->shape[1];
            AW = mat2_hbm_->shape[0];
            BW = input_hbm_->shape[0];
        }
        // if(lhs_is_cmem){
            // matmul_2xys_cmem(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, *(SIM_X86::tensor*)mem->cmem_addr, AH, AW, BW, ah, aw, bw);
        // }else{
            matmul_2xys(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, AH, AW, BW, ah, aw, bw);
        // }
    }else if(flag == 1 || flag == 6){
        if(flag == 6){
            lhs = *(SIM_X86::tensor*)mat2_hbm_->address;
            rhs = *(SIM_X86::tensor*)input_hbm_->address;
            AH = mat2_hbm_->shape[1];
            AW = mat2_hbm_->shape[0];
            BW = input_hbm_->shape[0];
        }
        matmul_resT_2xys(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, AH, AW, BW, ah, aw, bw);
    }else if(flag == 2 || flag == 3){
        if(flag == 3){
            lhs = *(SIM_X86::tensor*)mat2_hbm_->address;
            rhs = *(SIM_X86::tensor*)input_hbm_->address;
            AH = mat2_hbm_->shape[1];
            AW = mat2_hbm_->shape[0];
            BW = input_hbm_->shape[1];
        }else{
            BW = mat2_hbm_->shape[1];
        }
        matmul_rhsT_2xys(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, AH, AW, BW, ah, aw, bw);
    }else{
        if(flag == 5){
            lhs = *(SIM_X86::tensor*)mat2_hbm_->address;
            rhs = *(SIM_X86::tensor*)input_hbm_->address;
            AH = mat2_hbm_->shape[0];
            AW = mat2_hbm_->shape[1];
            BW = input_hbm_->shape[0];
        }else{
            AH = input_hbm_->shape[0];
            AW = input_hbm_->shape[1];
        }
        // printf("[XYS%d]: AH = %d, AW = %d, BW = %d\n", get_device_id(), AH, AW, BW);
        matmul_lhsT_2xys(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, AH, AW, BW, ah, aw, bw);
    }
    sync_device();
}
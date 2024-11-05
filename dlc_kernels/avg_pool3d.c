// #include "align.h"
// #include "matmul_t.h"
// #include "typehint.h"

// #include "kernel_arg_types.h"
// #include "libdevice.h"
// #include "permute.h"

#include "../x86.h"

inline void HBMToVMEM(SIM_X86::tensor hbm_in, SIM_X86::tensor vmem_out, const int mh, /*4B*/
                      const int mw,                                 /*4B*/
                      const int block_h,                            /*4B*/
                      const int block_w,                            /*4B*/
                      const int row, const int col) {
    const int rw = min(mw - col * block_w, block_w);
    const int rh = min(mh - row * block_h, block_h);
    for (int i = 0; i < rw; i += 128) {
        int handle = dlc_dma(tensor_slice(hbm_in, (i + mw * block_h * row + block_w * col) / 32), D_HBM,
                             tensor_slice(vmem_out, i / 32), D_VMEM, rh * 128, mw, rw, 128, 7);
        dlc_sync(handle);
    }
}

inline void VMEMToHBM(SIM_X86::tensor vmem_in, SIM_X86::tensor hbm_out, const int mh, /*4B*/
                      const int mw,                                 /*4B*/
                      const int block_h,                            /*4B*/
                      const int block_w,                            /*4B*/
                      const int row, const int col) {
    const int rw = min(mw - col * block_w, block_w);
    const int rh = min(mh - row * block_h, block_h);
    for (int i = 0; i < rw; i += 128) {
        int handle = dlc_dma(tensor_slice(vmem_in, i / 32), D_VMEM,
                             tensor_slice(hbm_out, (i + mw * block_h * row + block_w * col) / 32), D_HBM,
                             rh * 128, rw, mw, 128, 7);
        dlc_sync(handle);
    }
}

inline void padding(SIM_X86::tensor tensor_trans, SIM_X86::tensor tensor_pad, int D, int D_pad, int H, int H_pad, int W,
                    int W_pad, int C_al, int pad_d, int pad_h, int pad_w) {

    for (int d = 0; d < D_pad; d++) {
        for (int h = 0; h < H_pad; h++) {
            for (int w = 0; w < W_pad; w++) {
                for (int c = 0; c < C_al; c += 1024) {
                    int size = min(1024, C_al - c);
                    int mask = pre_exp2(size / 128);
                    float8_128 a = v_u32_move_b(0);
                    v_f32_st_tnsr_st_msk(((d * H_pad * W_pad + h * W_pad + w) * C_al + c) / 32, tensor_pad, 1,
                                         mask, a);
                }
            }
        }
    }
    for (int d = 0; d < D; d++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C_al; c += 1024) {
                    int size = min(1024, C_al - c);
                    int mask = pre_exp2(size / 128);
                    float8_128 a = v_f32_ld_tnsr_st_msk(((d * H * W + h * W + w) * C_al + c) / 32,
                                                        tensor_trans, 1, mask);
                    v_f32_st_tnsr_st_msk(
                        (((d + pad_d) * H_pad * W_pad + (h + pad_h) * W_pad + (w + pad_w)) * C_al + c) / 32,
                        tensor_pad, 1, mask, a);
                }
            }
        }
    }
}

inline void avg_pool_3d_kernel(SIM_X86::tensor tensor_pad, SIM_X86::tensor tensor_avg, int D_out, int H, int H_out, int W_out,
                               int C_al, int D_pad, int H_pad, int W_pad, int kernel_d, int kernel_h,
                               int kernel_w, int stride_d, int stride_h, int stride_w, int pad_d, int pad_h,
                               int pad_w, int W, int ceil_mode, int count_include_pad, int divisor_override) {

int PrintFlag = 0;
    if (get_device_id()) {
        // printf("D_out = %d\n", D_out);
        // printf("H_out = %d\n", H_out);
        // printf("W_out = %d\n", W_out);
        // printf("C_al = %d\n", C_al);
        // printf("CCC_al = %d\n", C_al);
    }
    for (int d = 0; d < D_out; d++) {
        int kernel_h_temp = kernel_h;
        int kernel_w_temp = kernel_w;
        int kernel_d_temp = kernel_d;
        for (int h = 0; h < H_out; h++) {
            kernel_h_temp = kernel_h;
            kernel_w_temp = kernel_w;
            kernel_d_temp = kernel_d;
            for (int w = 0; w < W_out; w++) {
                kernel_h_temp = kernel_h;
                kernel_w_temp = kernel_w;
                kernel_d_temp = kernel_d;
                for (int c = 0; c < C_al; c += 1024) {
                    int size = min(1024, C_al - c);
                    int mask = pre_exp2(size / 128);
                    // If the ceil_mode is 1 and the last added part is not done by paading
                    if (ceil_mode) {
                        if (w == (W_out - 1)) {
                            kernel_w_temp = W_pad - w * stride_w;
                            if (kernel_w_temp > kernel_w) {
                                kernel_w_temp = kernel_w;
                            }
                        }
                        if (h == (H_out - 1)) {
                            kernel_h_temp = H_pad - h * stride_h;
                            if (kernel_h_temp > kernel_h) {
                                kernel_h_temp = kernel_h;
                            }
                        }
                        if (d == (D_out - 1)) {
                            kernel_d_temp = D_pad - d * stride_d;
                            if (kernel_d_temp > kernel_d) {
                                kernel_d_temp = kernel_d;
                            }
                        }
                    }
                    float8_128 sum = v_u32_move_f(0);
                    int count = 0;
                    for (unsigned id = 0; id < kernel_d_temp; id += 1) {
                        for (unsigned ih = 0; ih < kernel_h_temp; ih += 1) {
                            for (unsigned iw = 0; iw < kernel_w_temp; iw += 1) {
                                int offset = iw + ih * W_pad + w * stride_w + h * W_pad * stride_h +
                                             id * H_pad * W_pad + d * W_pad * H_pad * stride_d;
                                float8_128 x =
                                    v_f32_ld_tnsr_st_msk((offset * C_al + c) / 32, tensor_pad, 1, mask);
                                bool valid = false;
                                if (!count_include_pad) {
                                    unsigned int d_offset = id + d * stride_d;
                                    unsigned int h_offset = ih + h * stride_h;
                                    unsigned int w_offset = iw + w * stride_w;
                                    if ((pad_h != 0) && (pad_w == 0) && (pad_d == 0)) {
                                        valid = !(h_offset < pad_h || h_offset >= (H_pad - pad_h));
                                    } else if ((pad_h == 0) && (pad_w != 0) && (pad_d == 0)) {
                                        valid = !(w_offset < pad_w || w_offset >= (W_pad - pad_w));
                                    } else if ((pad_h == 0) && (pad_w == 0) && (pad_d != 0)) {
                                        valid = !(d_offset < pad_d || d_offset >= (D_pad - pad_d));
                                    } else if ((pad_h != 0) && (pad_w != 0) && (pad_d == 0)) {
                                        if (!(h_offset < pad_h || h_offset >= (H_pad - pad_h))) {
                                            valid = !(w_offset < pad_w || w_offset >= (W_pad - pad_w));
                                        }
                                    } else if ((pad_h != 0) && (pad_w == 0) && (pad_d != 0)) {
                                        if (!(d_offset < pad_d || d_offset >= (D_pad - pad_d))) {
                                            valid = !(h_offset < pad_h || h_offset >= (H_pad - pad_h));
                                        }
                                    } else if ((pad_h == 0) && (pad_w != 0) && (pad_d != 0)) {
                                        if (!(d_offset < pad_d || d_offset >= (D_pad - pad_d))) {
                                            valid = !(w_offset < pad_w || w_offset >= (W_pad - pad_w));
                                        }
                                    } else if (pad_h != 0 && pad_w != 0 && pad_d != 0) {
                                        if (!(d_offset < pad_d || d_offset >= (D_pad - pad_d))) {
                                            if (!(h_offset < pad_h || h_offset >= (H_pad - pad_h))) {
                                                valid = !(w_offset < pad_w || w_offset >= (W_pad - pad_w));
                                            }
                                        }
                                    } else {
                                        valid = true;
                                    }
                                } else {
                                    valid = true;
                                }
                                if (valid) {
                                    sum = v_f32_add_b(sum, x);
                                    count += 1;
                                }
                                if (d == D_out - 1 && h == 0 && w == 0) {
                                    // printf("valid = %d, x = %f, sum = %f, count = %d, offset * C_al + c = %d\n", valid, x[0], sum[0], count, offset * C_al + c);
                                    if (offset * C_al + c == 219008) {
                                        // Print("tensor1 = ", tensor_slice(tensor_pad, 206208 / 32), 128, PrintType::FLOAT);
                                        // printf("ptr = %d, size = %d\n", tensor_pad.data_ptr, tensor_pad.data_size);
                                        // printf("ptr = %d, size = %d\n", tensor_slice(tensor_pad, 206208 / 32).data_ptr, tensor_slice(tensor_pad, 206208 / 32).data_size);
                                        // printf("ptr = %f\n", tensor_slice(tensor_pad, 206208 / 32).data_ptr[0]);
                                        // Print("tensor2 = ", tensor_slice(tensor_pad, (offset * C_al + c) / 32), 128, PrintType::FLOAT);
                                        // printf("ptr2 = %d, size2 = %d\n", tensor_pad.data_ptr, tensor_pad.data_size);
                                        // printf("ptr2 = %d, size2 = %d\n", tensor_slice(tensor_pad, 219008 / 32).data_ptr, tensor_slice(tensor_pad, 219008 / 32).data_size);
                                        // printf("ptr = %f\n", tensor_slice(tensor_pad, 219008 / 32).data_ptr[0]);
                                        // Print("tensor3 = ", tensor_pad, 300000, PrintType::FLOAT);
                                    }
                                }
                                if (PrintFlag == 0) {
                                    // printf("id = %d, valid = %d, x[0] = %f, x[1] = %f\n", id * kernel_h_temp * kernel_w_temp + ih * kernel_w_temp + iw, valid, x[0], x[1]);
                                    if (id * kernel_h_temp * kernel_w_temp + ih * kernel_w_temp + iw == 7) {
                                        // printf("offset * C_al + c = %d\n", offset * C_al + c);
                                        // printf("mask = %d\n", mask);
                                        // Print("tensor =", tensor_slice(tensor_pad, (offset * C_al + c) / 32), 1024, PrintType::FLOAT);
                                    }
                                }
                            }
                        }
                    }
                    if (divisor_override) {
                        count = divisor_override;
                    }
                    float8_128 count_num = count;
                    count_num = __dlc_frcp_rd_without_unary(count_num);
                    float8_128 avg = v_f32_mul_b(count_num, sum);
                    // printf("count_num = %f, count = %d\n", count_num[0], count);
                    // printf("sum = %f\n", sum[0]);
                    // printf("avg = %f\n", avg[0]);
                    if (get_device_id() && PrintFlag == 0) {
                        // Print("count_num = ", count_num, PrintType::FLOAT);
                        // Print("sum = ", sum, PrintType::FLOAT);
                        // Print("avg = ", avg, PrintType::FLOAT);
                        PrintFlag = 1;
                    }
                    int offset_trans = d * H_out * W_out + h * W_out + w;
                    v_f32_st_tnsr_st_msk((offset_trans * C_al + c) / 32, tensor_avg, 1, mask, avg);
                }
            }
        }
    }
}

inline void permute_(SIM_X86::tensor in, SIM_X86::tensor out, int d3, int d2, int d1, int d0, int p4, int p3, int p2, int p1,
                     int p0) {
    int oridim[5] = {ALIGN128(d0), d1, d2, d3, 1};
    int perm[5] = {p0, p1, p2, p3, p4};
    load_tran_trans(oridim, perm, in, out, d0);
    // printf("use load_tran_trans\n");
}
void main_x86(SIM_X86::DLCMem *mem_info, SIM_X86::DLCTensor *hbm_in_, SIM_X86::DLCTensor *hbm_out_,
            //   SIM_X86::DLCScalar *Parameters) {
              int *Parameters) {
/*##AUTO-GEN##*/TensorFixDims(hbm_in_);TensorFixDims(hbm_out_);/*##END-GEN##*/
    SIM_X86::tensor hbm_out = *(SIM_X86::tensor*)hbm_out_->address;
    SIM_X86::tensor hbm_in = *(SIM_X86::tensor*)hbm_in_->address;

    int UseVmemSize = min(mem_info->vmem_size / 4, 1024 * 3072);

    unsigned *InputSize = hbm_in_->shape;
    unsigned *OutputSize = hbm_out_->shape;

    // const int kernel_d = Parameters[0].value;
    // const int kernel_h = Parameters[1].value;
    // const int kernel_w = Parameters[2].value;

    // const int stride_d = Parameters[3].value;
    // const int stride_h = Parameters[4].value;
    // const int stride_w = Parameters[5].value;

    // const int pad_d = Parameters[6].value;
    // const int pad_h = Parameters[7].value;
    // const int pad_w = Parameters[8].value;

    // const int ceil_mode = Parameters[9].value;
    // const int count_include_pad = Parameters[10].value;
    // const int divisor_override = Parameters[11].value;
    const int kernel_d = Parameters[0];
    const int kernel_h = Parameters[1];
    const int kernel_w = Parameters[2];

    const int stride_d = Parameters[3];
    const int stride_h = Parameters[4];
    const int stride_w = Parameters[5];

    const int pad_d = Parameters[6];
    const int pad_h = Parameters[7];
    const int pad_w = Parameters[8];

    const int ceil_mode = Parameters[9];
    const int count_include_pad = Parameters[10];
    const int divisor_override = Parameters[11];

    const int N = InputSize[4];
    const int C = InputSize[3];
    const int D = InputSize[2];
    const int H = InputSize[1];
    const int W = InputSize[0];

    int W_al = ALIGN128(W);
    int C_al = ALIGN128(C);

    int H_pad = H + 2 * pad_h;
    int W_pad = W + 2 * pad_w;
    int D_pad = D + 2 * pad_d;

    int W_out = OutputSize[0];
    int H_out = OutputSize[1];
    int D_out = OutputSize[2];

    int W_out_al = ALIGN128(W_out);

    int kernel_h_c = kernel_h, kernel_w_c = kernel_w, kernel_d_c = kernel_d;

    int vmemb = UseVmemSize / 5;
    vmemb = vmemb / 128 * 128;

    SIM_X86::tensor vmem_in = *(SIM_X86::tensor*)mem_info->vmem_addr;
    SIM_X86::tensor tensor_trans = (vmem_in + vmemb / 32);
    SIM_X86::tensor tensor_pad = (tensor_trans + vmemb / 32);
    SIM_X86::tensor tensor_avg = (tensor_pad + (vmemb + 1024 * 256) / 32);
    SIM_X86::tensor vmem_out = (tensor_avg + vmemb / 32);

    if (get_device_id() == 1) {
        std::cout << "vmemb = " << vmemb << std::endl;
        std::cout << "Address of tensor_pad: " << static_cast<void*>(tensor_pad.data_ptr) << std::endl;
        std::cout << "Address of tensor_avg: " << static_cast<void*>(tensor_avg.data_ptr) << std::endl;
    }

    int N_half = N / 2;
    int N_batch = N_half;
    if (get_device_id() == 1) {
        N_batch = N - N_half;
        hbm_out = *(SIM_X86::tensor*)hbm_out_->address + N_half * W_out_al * H_out * D_out * C / 32;
        hbm_in = *(SIM_X86::tensor*)hbm_in_->address + N_half * W_al * H * D * C / 32;
    }
    // int N_batch = N;
    // if (get_device_id() == 0) { N_batch = 0; }
    
    // printf("[XYS%d]: N_batch = %d, id = %d\n", get_device_id(), N_batch);

    for (int n = 0; n < N_batch; n++) {
        HBMToVMEM(hbm_in, vmem_in, N * C, D * H * W_al, C, D * H * W_al, n, 0);
        // Print("vmem_in = ", vmem_in, 3072, PrintType::FLOAT);
        // transpose(vmem_in, tensor_trans, C, D * H * W_al);
        // Convert_W_avg_pool_3d_kernelal_to_W(tensor_trans, tensor_trans, D, H, W, W_al, C_al);
        // Print("vmem_in = ", vmem_in, C * D * H * ALIGN128(W), PrintType::FLOAT);
        if (W % 128 == 0)
            tile_trans_transfer(vmem_in, tensor_trans, 0, C, D * H * W, 0);
        else
            permute_(vmem_in, tensor_trans, C, D, H, W, 4, 2, 1, 0, 3);
        // Print("tensor_trans = ", tensor_trans, D * H * W * ALIGN128(C), PrintType::FLOAT);
        padding(tensor_trans, tensor_pad, D, D_pad, H, H_pad, W, W_pad, C_al, pad_d, pad_h, pad_w);
        // Print("tensor_pad = ", tensor_pad, 300000, PrintType::FLOAT);

        // Print("tensor_pad = ", tensor_pad, 3072, PrintType::FLOAT);

        avg_pool_3d_kernel(tensor_pad, tensor_avg, D_out, H, H_out, W_out, C_al, D_pad, H_pad, W_pad,
                           kernel_d_c, kernel_h_c, kernel_w_c, stride_d, stride_h, stride_w, pad_d, pad_h,
                           pad_w, W, ceil_mode, count_include_pad, divisor_override);
        // Print("tensor_avg = ", tensor_avg, 128, PrintType::FLOAT);

        // Print("tensor_avg = ", tensor_avg, D_out * H_out * W_out * ALIGN128(C), PrintType::FLOAT);
        // printf("h = %d, w = %d\n", D_out * H_out * W_out, C); // 225 1
        if (W_out % 128 == 0)
            tile_trans_transfer(tensor_avg, vmem_out, 0, D_out * H_out * W_out, C, 0);
        else
            permute_(tensor_avg, vmem_out, D_out, H_out, W_out, C, 4, 0, 3, 2, 1);
        // Print("vmem_out = ", vmem_out, 5760, PrintType::FLOAT);
        // Convert_W_to_W_al(tensor_avg, tensor_trans, D_out, H_out, W_out, W_out_al, C_al);
        // transpose(tensor_trans, vmem_out, D_out * H_out * W_out_al, C);
        VMEMToHBM(vmem_out, hbm_out, N * C, D_out * H_out * W_out_al, C, D_out * H_out * W_out_al, n, 0);
    } 
// std::this_thread::sleep_for(std::chrono::seconds(10)); // 暂停10秒
    sync_device();
}

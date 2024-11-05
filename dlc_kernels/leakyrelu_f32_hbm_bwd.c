// #include "ldst.h"
// #include "align.h"
// #include "typehint.h"

// #include "kernel_arg_types.h"

#include "../x86.h"

void main_x86(SIM_X86::DLCMem *mem_info, SIM_X86::DLCTensor *output_hbm_, SIM_X86::DLCTensor *output_grad_hbm_, SIM_X86::DLCTensor *input_grad_hbm_,
          float *relu_number_) {
/*##AUTO-GEN##*/TensorFixDims(output_hbm_);TensorFixDims(output_grad_hbm_);TensorFixDims(input_grad_hbm_);/*##END-GEN##*/

    SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;
    SIM_X86::tensor output_grad_hbm = *(SIM_X86::tensor*)output_grad_hbm_->address;
    SIM_X86::tensor input_grad_hbm = *(SIM_X86::tensor*)input_grad_hbm_->address;

    int VMEMsize_ = min(((int)mem_info->vmem_size) / (4 * 2) , 3072 * 512);

    SIM_X86::tensor output_vmem = *(SIM_X86::tensor*)mem_info->vmem_addr;
    SIM_X86::tensor output_grad_vmem = (output_vmem +  VMEMsize_ / 32);

    float relu_number = *(float *)relu_number_;
//     unsigned *InputSize = output_hbm_->shape;

    const int D_HBM = 1;
    const int D_VMEM = 2;

    /* input0 : leakyrelu的输出结果   ， input1 ：输出结果的梯度  ，  output ： 输入值的梯度*/

    /*输入Tensor的总长度*/

    unsigned* InputSize = output_hbm_->shape;

    const unsigned dim0 = InputSize[0];
    const unsigned dim1 = InputSize[1];
    const unsigned dim2 = InputSize[2];
    const unsigned dim3 = InputSize[3];
    const unsigned dim4 = InputSize[4];
    const unsigned dim0_pad = ALIGN128(dim0);

    int hbm_len = dim0_pad * dim1 * dim2 * dim3 * dim4;

    int device_id = get_device_id();
      //使用双xys
    int len_xys = hbm_len / 2;
    len_xys = ALIGN128(len_xys);
    int xys_offset = 0;
    if(device_id == 1){
        xys_offset = len_xys;
        len_xys = hbm_len - len_xys;
    }


    for (int l = 0; l <= len_xys; l+= VMEMsize_) {

        /*DMA传输需要进行计算的数据量*/
        int math_size = min(len_xys - l , VMEMsize_);
        /*使用双xys进行控制dma计算*/
        int handle1 = dlc_dma(output_hbm + (l + xys_offset) / 32, D_HBM, output_vmem,
                              D_VMEM, math_size, 128, 128, 128, 7);
        dlc_sync(handle1);
        int handle2 = dlc_dma(output_grad_hbm + (l + xys_offset) / 32, D_HBM,
                              output_grad_vmem, D_VMEM, math_size, 128, 128, 128, 7);
        dlc_sync(handle2);

        /*将Vmem的数据load并进行leakyrelubackword操作，当leakyrelu结果小于0时y=x(梯度)*y(relu_number)，当其大于0时y=x(梯度)*/
        int d = 0;
        for (; d + 1024 <= math_size ; d += 1024) {

            float8_128 x = v_f32_ld_tnsr_st_msk(d / 32, output_vmem, 1, 255);
            float8_128 y = v_f32_ld_tnsr_st_msk(d / 32, output_grad_vmem, 1, 255);
            float8_128 z = relu_number;
            // bool1024 cond = v_f32_cmp_grt_b(x, 0.0f);
            // y = v_f32_mul_vb(y, z, y, cond);
            bool8_128 cond = v_f32_cmp(GT, x, 0.0f);
            y = v_f32_mul_vb(y, z, y, cond);
            v_f32_st_tnsr_st_msk(d / 32, output_vmem, 1, 255, y);
        }

        int len = math_size % 1024;
        int ldst_vmask = pre_exp2(len / 128);
        float8_128 x = v_f32_ld_tnsr_st_msk((math_size - len) / 32, output_vmem, 1, ldst_vmask);
        float8_128 y = v_f32_ld_tnsr_st_msk((math_size - len) / 32, output_grad_vmem, 1, ldst_vmask);
        float8_128 z = relu_number;
        // bool1024 cond = v_f32_cmp_grt_b(x, 0.0f);
        // y = v_f32_mul_vb(y, z, y, cond);
        bool8_128 cond = v_f32_cmp(GT, x, 0.0f);
        y = v_f32_mul_vb(y, z, y, cond);
        v_f32_st_tnsr_st_msk((math_size - len) / 32, output_vmem, 1, ldst_vmask, y);

//          __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output_vmem, 1, 1));

        /*将计算好的值放回HBM*/
        int handle3 =
            dlc_dma(output_vmem, D_VMEM, input_grad_hbm + (l + xys_offset) / 32, D_HBM,
                    math_size, 128, 128, 128, 7);
        dlc_sync(handle3);

    }

    sync_device();
}

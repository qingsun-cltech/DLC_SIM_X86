// #include "bf16.h"
// #include "ldst.h"
// #include "align.h"
// #include "typehint.h"

// #include "kernel_arg_types.h"

// const int D_HBM = 1;
// const int D_VMEM = 2;
// inline void HBM2VMem(SIM_X86::tensor hbm_address, SIM_X86::tensor vmem_address, int length) {
//     int handle = dlc_dma(tensor_slice(hbm_address, 0 / 32), D_HBM, tensor_slice(vmem_address, 0 / 32), D_VMEM,
//                          length, 128, 128, 128, 7);
//     dlc_sync(handle);
// }
// inline void Vmem2HBM(SIM_X86::tensor vmem_address, SIM_X86::tensor hbm_address, int length) {
//     int handle = dlc_dma(tensor_slice(vmem_address, 0 / 32), D_VMEM, tensor_slice(hbm_address, 0 / 32), D_HBM,
//                          length, 128, 128, 128, 7);
//     dlc_sync(handle);
// }

#include "../x86.h"

inline void leakyrelu_bwd(SIM_X86::tensor vmem_output, SIM_X86::tensor vmem_output_grad, float relu_value, int _len) {
    int d = 0;
    for (; d + 1024 <= _len ; d += 1024) {

        float8_128 x = v_f32_ld_tnsr_st_msk(d / 32, vmem_output, 1, 255);

        float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
        float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));

        float8_128 y = v_f32_ld_tnsr_st_msk(d / 32, vmem_output_grad, 1, 255);

        float8_128 y2 = bfloat16_to_float(unpack_16b(__$S(y), 1));
        float8_128 y1 = bfloat16_to_float(unpack_16b(__$S(y), 0));

        float8_128 z = relu_value;
        // bool1024 cond2 = v_f32_cmp_grt_b(x2, 0.0f);
        // bool1024 cond1 = v_f32_cmp_grt_b(x1, 0.0f);

        // y2 = v_f32_mul_vb(y2, z, y2, cond2);
        // y1 = v_f32_mul_vb(y1, z, y1, cond1);
        
        bool8_128 cond2 = v_f32_cmp(GT, x2, 0.0f);
        bool8_128 cond1 = v_f32_cmp(GT, x1, 0.0f);
        y2 = v_f32_mul_vb(y2, z, y2, cond2);
        y1 = v_f32_mul_vb(y1, z, y1, cond1);

        float8_128 res = __$F(float_to_bfloat16(y2, y1));

        v_f32_st_tnsr_st_msk(d / 32, vmem_output, 1, 255, res);
    }
    if (_len % 1024 != 0) {
        int len = _len % 1024;
        int ldst_vmask = pre_exp2(len / 128);
        float8_128 x = v_f32_ld_tnsr_st_msk((_len - len) / 32, vmem_output, 1, ldst_vmask);

        float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
        float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));

        float8_128 y = v_f32_ld_tnsr_st_msk((_len - len) / 32, vmem_output_grad, 1, ldst_vmask);

        float8_128 y2 = bfloat16_to_float(unpack_16b(__$S(y), 1));
        float8_128 y1 = bfloat16_to_float(unpack_16b(__$S(y), 0));

        float8_128 z = relu_value;
        // bool1024 cond2 = v_f32_cmp_grt_b(x2, 0.0f);
        // bool1024 cond1 = v_f32_cmp_grt_b(x1, 0.0f);

        // y2 = v_f32_mul_vb(y2, z, y2, cond2);
        // y1 = v_f32_mul_vb(y1, z, y1, cond1);

        bool8_128 cond2 = v_f32_cmp(GT, x2, 0.0f);
        bool8_128 cond1 = v_f32_cmp(GT, x1, 0.0f);
        y2 = v_f32_mul_vb(y2, z, y2, cond2);
        y1 = v_f32_mul_vb(y1, z, y1, cond1);

        float8_128 res = __$F(float_to_bfloat16(y2, y1));

        v_f32_st_tnsr_st_msk((_len - len) / 32, vmem_output, 1, ldst_vmask, res);
    }
}

// void main(DLCMem *mem_info, DLCTensor *output_hbm_, DLCTensor *output_grad_hbm_, DLCTensor *input_grad_hbm_,
//           DLCScalar *relu_number_) {
void main_x86(SIM_X86::DLCMem *mem_info, SIM_X86::DLCTensor *output_hbm_, SIM_X86::DLCTensor *output_grad_hbm_, SIM_X86::DLCTensor *input_grad_hbm_,
          float *relu_number_) {
/*##AUTO-GEN##*/TensorFixDims(output_hbm_);TensorFixDims(output_grad_hbm_);TensorFixDims(input_grad_hbm_);/*##END-GEN##*/

    SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;
    SIM_X86::tensor output_grad_hbm = *(SIM_X86::tensor*)output_grad_hbm_->address;
    SIM_X86::tensor input_grad_hbm = *(SIM_X86::tensor*)input_grad_hbm_->address;

    int VMEMsize_ = min(((int)mem_info->vmem_size) / (4 * 2), 512 * 3072);

    SIM_X86::tensor output_vmem = *(SIM_X86::tensor*)mem_info->vmem_addr;
    SIM_X86::tensor output_grad_vmem = (output_vmem +  VMEMsize_ / 32);

    float relu_number = *(float *)relu_number_;
    unsigned *InputSize = output_hbm_->shape;

    const unsigned dim0 = InputSize[0];
    const unsigned dim1 = InputSize[1];
    const unsigned dim2 = InputSize[2];
    const unsigned dim3 = InputSize[3];
    const unsigned dim4 = InputSize[4];
    const unsigned dim0_dma = ALIGN256(dim0) / 2;

    int hbm_len = dim0_dma * dim1 * dim2 * dim3 * dim4;

    int device_id = get_device_id();
      //使用双xys
    int len_xys = hbm_len / 2;
    len_xys = ALIGN128(len_xys);
    int xys_offset = 0;
    if(device_id == 1){
        xys_offset = len_xys;
        len_xys = hbm_len - len_xys;
    }


    for (int l = 0; l < len_xys; l += VMEMsize_) {
        int VMemsize = min(len_xys - l, VMEMsize_);
        /*将数据dma传输到vmem上*/
        HBM2VMem(output_hbm + (l + xys_offset) / 32, output_vmem, VMemsize);
        HBM2VMem(output_grad_hbm + (l + xys_offset) / 32, output_grad_vmem, VMemsize);

        /*进行leakyrelu_bwd操作*/
        leakyrelu_bwd(output_vmem, output_grad_vmem, relu_number, VMemsize);

//          __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output_vmem, 1, 1));

        /*将转回的16dma传回到hbm中*/
        Vmem2HBM(output_vmem, input_grad_hbm + (l + xys_offset) / 32, VMemsize);
    }

    sync_device();
}

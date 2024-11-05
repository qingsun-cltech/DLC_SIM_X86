// #include "bf16.h"
// #include "ldst.h"
// #include "align.h"
// #include "typehint.h"

// #include "kernel_arg_types.h"

#include "../x86.h"

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

inline void leakyrelu(SIM_X86::tensor vmem_input0, float relu_value, int _len) {
    int l = 0;
    for (; l + 1024 <= _len; l += 1024) {
        float8_128 input_value = load8_128_stride_with_ldmask(l / 32, 1, 255, vmem_input0);

        float8_128 input_value2 = bfloat16_to_float(unpack_16b(__$S(input_value), 1));
        float8_128 input_value1 = bfloat16_to_float(unpack_16b(__$S(input_value), 0));

        float8_128 relu_number = relu_value;

        // bool1024 cond2 = v_f32_cmp_grt_b(input_value2, 0.0f);
        // bool1024 cond1 = v_f32_cmp_grt_b(input_value1, 0.0f);

        // float8_128 output_value2 = v_f32_mul_vb(input_value2, relu_number, input_value2, cond2);
        // float8_128 output_value1 = v_f32_mul_vb(input_value1, relu_number, input_value1, cond1);
        
        bool8_128 cond2 = v_f32_cmp(GT, input_value2, 0.0f);
        bool8_128 cond1 = v_f32_cmp(GT, input_value1, 0.0f);
        float8_128 output_value2 = v_f32_mul_vb(input_value2, relu_number, input_value2, cond2);
        float8_128 output_value1 = v_f32_mul_vb(input_value1, relu_number, input_value1, cond1);

        float8_128 res = __$F(float_to_bfloat16(output_value2, output_value1));

        store8_128_stride8_with_stmask(l / 32, 1, 255, vmem_input0, res);
    }
    if (_len % 1024 != 0) {
        int size = _len % 1024;
        int ldst_maskt = pre_exp2(size / 128);
        float8_128 input_value = load8_128_stride_with_ldmask((_len - size) / 32, 1, ldst_maskt, vmem_input0);

        float8_128 input_value2 = bfloat16_to_float(unpack_16b(__$S(input_value), 1));
        float8_128 input_value1 = bfloat16_to_float(unpack_16b(__$S(input_value), 0));

        float8_128 relu_number = relu_value;
        // bool1024 cond2 = v_f32_cmp_grt_b(input_value2, 0.0f);
        // bool1024 cond1 = v_f32_cmp_grt_b(input_value1, 0.0f);

        // float8_128 output_value2 = v_f32_mul_vb(input_value2, relu_number, input_value2, cond2);
        // float8_128 output_value1 = v_f32_mul_vb(input_value1, relu_number, input_value1, cond1);

        bool8_128 cond2 = v_f32_cmp(GT, input_value2, 0.0f);
        bool8_128 cond1 = v_f32_cmp(GT, input_value1, 0.0f);
        float8_128 output_value2 = v_f32_mul_vb(input_value2, relu_number, input_value2, cond2);
        float8_128 output_value1 = v_f32_mul_vb(input_value1, relu_number, input_value1, cond1);

        float8_128 res = __$F(float_to_bfloat16(output_value2, output_value1));

        store8_128_stride8_with_stmask((_len - size) / 32, 1, ldst_maskt, vmem_input0, res);
    }
}

// void main(SIM_X86::tensor hbm_input0, SIM_X86::tensor vmem_input0, SIM_X86::tensor hbm_output, TensorInfo* tensorInfo , float
// relu_value){
// void main(DLCMem *mem_info, DLCTensor *hbm_input0_, DLCTensor *hbm_output_, DLCScalar *relu_value_) {
void main_x86(SIM_X86::DLCMem* mem_info, SIM_X86::DLCTensor* hbm_input0_, SIM_X86::DLCTensor* hbm_output_, float* relu_value_){
/*##AUTO-GEN##*/TensorFixDims(hbm_input0_);TensorFixDims(hbm_output_);/*##END-GEN##*/

    unsigned *InputSize = hbm_input0_->shape;
    SIM_X86::tensor hbm_input0 = *(SIM_X86::tensor*)hbm_input0_->address;
    SIM_X86::tensor hbm_output = *(SIM_X86::tensor*)hbm_output_->address;
    SIM_X86::tensor vmem_input0 = *(SIM_X86::tensor*)mem_info->vmem_addr;
    float relu_value = *(float *)relu_value_;

    int VMEMsize_ = min(((int)mem_info->vmem_size) / 4, 1024 * 3072);

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

        HBM2VMem(hbm_input0 + (l + xys_offset) / 32, vmem_input0, VMemsize);

        leakyrelu(vmem_input0, relu_value, VMemsize);

//          __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, vmem_input0, 1, 1));

        Vmem2HBM(vmem_input0, hbm_output + (l + xys_offset) / 32, VMemsize);
    }

    sync_device();
}
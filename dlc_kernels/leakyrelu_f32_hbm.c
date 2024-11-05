// #include "typehint.h"
// #include "align.h"

// #include "kernel_arg_types.h"
// #include "ldst.h"

#include "../x86.h"

// void main( tensor input0 , tensor input0_vmem  , tensor  output,  tensor output_hbm  ,TensorInfo* tensorInfo , float relu_number) {
// void main(DLCMem* mem_info, DLCTensor* input0_, DLCTensor* output_hbm_, DLCScalar* relu_number_){
void main_x86(SIM_X86::DLCMem* mem_info, SIM_X86::DLCTensor* input0_, SIM_X86::DLCTensor* output_hbm_, float* relu_number_){
/*##AUTO-GEN##*/TensorFixDims(input0_);TensorFixDims(output_hbm_);/*##END-GEN##*/


    SIM_X86::tensor input0 = *(SIM_X86::tensor*)input0_->address;
    SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;
    int VMEMsize_ = ((int)mem_info->vmem_size) / 4;

    int vmem_offset = min(VMEMsize_,3072 * 1024);
    SIM_X86::tensor input0_vmem = *(SIM_X86::tensor*)mem_info->vmem_addr;

    float relu_number = *(float*)relu_number_;

    const int D_HBM = 1;
    const int D_VMEM = 2;

    unsigned* InputSize = input0_->shape;

    const unsigned dim0 = InputSize[0];
    const unsigned dim1 = InputSize[1];
    const unsigned dim2 = InputSize[2];
    const unsigned dim3 = InputSize[3];
    const unsigned dim4 = InputSize[4];
    const unsigned dim0_pad = ALIGN128(dim0);
    const int sync_size = 3 * 32 * 128;


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

    float8_128 res;
    float8_128 relu_value = relu_number;

    /*设定一批数据传输的大小，若小于VMEM就一批传输完成，若大于VMEM采用VMEME极限大小作为一批数据传输*/
    for(int l = 0 ; l <= len_xys ; l += vmem_offset)
        {
            int math_size = min(len_xys - l , vmem_offset);


            /*DMA传输需要进行计算的数据量*/
            int handle1 = dlc_dma(tensor_slice(input0 , ( l + xys_offset) / 32) ,  D_HBM  , input0_vmem, D_VMEM, math_size , 128, 128, 128, 7);
                // dlc_sync(handle1);
            int sync_count = 0;
            for(;sync_count + sync_size <= math_size; sync_count += sync_size ){
                dlc_sync_gte(handle1,  (sync_count + sync_size) /  128);
                int d = 0;
                for (; d  < sync_size; d += 1024) {
                        float8_128 x = v_f32_ld_tnsr_st_msk( (d + sync_count)/ 32 , input0_vmem ,1 , 255);
                        // bool1024 cond = v_f32_cmp_grt_b(x, 0.0f);
                        // res = v_f32_mul_vb(x, relu_value, x, cond);
                        bool8_128 cond = v_f32_cmp(GT, x, 0.0f);
                        res = v_f32_mul_vb(x, relu_value, x, cond);
                        v_f32_st_tnsr_st_msk((d + sync_count) / 32, input0_vmem ,1 , 255 , res);
                    }
            }
            /*将Vmem的数据load并进行leakyrelu操作，当其小于0时y=x*y，当其大于0时y=x*/
            if(sync_count < math_size){
                int last_size = math_size - sync_count;
                dlc_sync_gte(handle1,  math_size / 128);
                int d = 0;
                for (; d < last_size; d += 1024) {
                        int len = min(last_size - d,1024);
                        int ldst_mask = pre_exp2(len / 128);
                        float8_128 x = v_f32_ld_tnsr_st_msk( (d + sync_count)/ 32 , input0_vmem ,1 , ldst_mask);
                        // bool1024 cond = v_f32_cmp_grt_b(x, 0.0f);
                        // res = v_f32_mul_vb(x, relu_value, x, cond);
                        bool8_128 cond = v_f32_cmp(GT, x, 0.0f);
                        res = v_f32_mul_vb(x, relu_value, x, cond);
                        v_f32_st_tnsr_st_msk((d + sync_count) / 32, input0_vmem ,1 , ldst_mask , res);
                    }

            }
            dlc_sync(handle1);
            // __attribute__((unused))volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, input0_vmem, 1, 1));
            /*将计算好的值放回HBM*/
            int handle2 = dlc_dma(input0_vmem ,  D_VMEM  ,tensor_slice(output_hbm , (l + xys_offset) / 32), D_HBM, math_size , 128, 128, 128, 7);
                dlc_sync(handle2);             
        }
    sync_device();
}




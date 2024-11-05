// #include "ldst.h"
// #include "typehint.h"
// #include "convert_element_type.h"

// #include "kernel_arg_types.h"
// #include "libdevice.h"

// const int D_HBM = 1;
// const int D_VMEM = 2;

// inline void HBM2VMem(tensor hbm_address, tensor vmem_address, int length){
//     int handle = dlc_dma(tensor_slice(hbm_address, 0 / 32), D_HBM,
//                         tensor_slice(vmem_address, 0 / 32), D_VMEM, length, 128,
//                         128, 128, 7);
//     dlc_sync(handle);    
// }
// inline void Vmem2HBM(tensor vmem_address, tensor hbm_address, int length){
//     int handle = dlc_dma(tensor_slice(vmem_address, 0 / 32), D_VMEM,
//                         tensor_slice(hbm_address, 0 / 32), D_HBM, length, 128,
//                         128, 128, 7);
//     dlc_sync(handle);    
// }


// inline void vmem_isinf_bf16(tensor in_out, int len) {
//     for (int i = 0; i < len / 1024; i += 1) {
//         float8_128 x = v_f32_ld_tnsr_st_msk(i * 32, in_out, 1, 255);
//         float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
//         float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));  

//         int8_128 res_int2 = __dlc_isinff(x2);
//         int8_128 res_int1 = __dlc_isinff(x1);

//         bool8_128 isinf_bool2 = v_s32_cmp(EQ, res_int2, 0);
//         bool8_128 isinf_bool1 = v_s32_cmp(EQ, res_int1, 0);

//         res_int2 = v_s32_sel(isinf_bool2, 1, 0);
//         res_int1 = v_s32_sel(isinf_bool1, 1, 0);


//         v_st_generic(i * 32, in_out, 1, 255, int_to_int16(res_int2, res_int1));
//     }
//     if (len % 1024 != 0) {
//         int actual_len = len % 1024;
//         int ldst_vmask = pre_exp2(actual_len / 128);
//         float8_128 x = v_f32_ld_tnsr_st_msk((len - actual_len) / 32, in_out, 1, ldst_vmask);
//         float8_128 x2 = bfloat16_to_float(unpack_16b(__$S(x), 1));
//         float8_128 x1 = bfloat16_to_float(unpack_16b(__$S(x), 0));  

//         int8_128 res_int2 = __dlc_isinff(x2);
//         int8_128 res_int1 = __dlc_isinff(x1);

//         bool8_128 isinf_bool2 = v_s32_cmp(EQ, res_int2, 0);
//         bool8_128 isinf_bool1 = v_s32_cmp(EQ, res_int1, 0);

//         res_int2 = v_s32_sel(isinf_bool2, 1, 0);
//         res_int1 = v_s32_sel(isinf_bool1, 1, 0);

//         v_st_generic((len - actual_len) / 32, in_out, 1, ldst_vmask, int_to_int16(res_int2, res_int1));
//     }
// }
// inline int calcVMemBlockSize(int height, int width, int vmemSize)
// {
//     int k = height * width;
//     if(k > vmemSize){
//         return (vmemSize / width);
//     }
//     return height;
// }

// void main(DLCMem* mem_info,DLCTensor* hbm_input_, DLCTensor* hbm_output_){
// /*##AUTO-GEN##*/TensorFixDims(hbm_input_);TensorFixDims(hbm_output_);/*##END-GEN##*/
    

//     void* hbm_input = hbm_input_->address;
//     void* hbm_output = hbm_output_->address;
//     unsigned* InputSize = hbm_input_->shape; 

//     int VMEMsize_ = ((int)mem_info->vmem_size) / 4;

//     int vmem_offset = min(VMEMsize_,3072 * 1024);

//     void* vmem_input = (void*)mem_info->vmem_addr;


//     int total_size = ((InputSize[0] + 255) & 0xffffff00) * hbm_output_->dim1 / 2;

//     int xys0_size = ((total_size / 2) + 255) & 0xffffff00;

//     int size = xys0_size;

//     int device_id = get_device_id();

//     if (device_id == 1) {
//         size = total_size - xys0_size;
//     }
    
//     for(int l = 0; l < size; l += vmem_offset){
//         int VMemsize = min(size - l, vmem_offset);
//         HBM2VMem(hbm_input + device_id * xys0_size / 32 + l / 32, vmem_input, VMemsize);    

//         vmem_isinf_bf16(vmem_input, VMemsize);
        
//         Vmem2HBM(vmem_input, hbm_output + device_id * xys0_size / 32 + l / 32, VMemsize);
//     }

//     sync_device();
// } 

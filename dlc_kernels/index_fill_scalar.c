// #include "bf16.h"
// #include "ldst.h"
// #include "typehint.h"
// #include "permute.h"

// #include "kernel_arg_types.h"
// #include "align.h"

#include "../x86.h"

const int D_SMEM = 0;

inline void HBM2SMem(SIM_X86::tensor hbm_address, SIM_X86::tensor smem_address, int length){
    int handle = dlc_dma(hbm_address, D_HBM, smem_address, D_SMEM, length, 128,
                        128, 128, 7);
    dlc_sync(handle);    
}

inline void HBMtoVMem(SIM_X86::tensor hbm_src, SIM_X86::tensor vmem_dst, int src_addr, int dst_addr, int length){
    int handle = dlc_dma(tensor_slice(hbm_src, src_addr / 32), D_HBM,
                         tensor_slice(vmem_dst, dst_addr / 32), D_VMEM, length, 128,
                         128, 128, 7);
    dlc_sync(handle);	
}
inline void VMEMtoHBM(SIM_X86::tensor vmem_src, SIM_X86::tensor hbm_dst, int src_addr, int dst_addr, int length){
    int handle = dlc_dma(tensor_slice(vmem_src, src_addr / 32), D_VMEM,
                         tensor_slice(hbm_dst, dst_addr / 32), D_HBM, length, 128,
                         128, 128, 7);
    dlc_sync(handle);	
}



inline void vmask_fill(SIM_X86::tensor vmem_input, int math_h , int dim0_pad , SIM_X86::tensor smem_index, int index_size ,float value  ){

  bool8_128 cmp_temp_core;
  int vs = 0;
  //每个1024进行一次sel
  for(; vs + 1024 <= dim0_pad ; vs += 1024){
    int8_128 core_id = get_core_id();
    core_id = core_id + vs;
    bool8_128 cmp_core = 0;
    for(int j = 0 ; j < index_size ; j++){
      float SS = smem_index[j];
      cmp_temp_core =  v_s32_cmp(EQ, core_id, *reinterpret_cast<int*>(&smem_index[j]));
      // cmp_temp_core =  v_s32_cmp(EQ, core_id, smem_index[j]);
      cmp_core = cmp_core | cmp_temp_core;
    }
    //后续所有的共用同一个cmp_core
    for(int dim0_count = 0; dim0_count < math_h ; dim0_count += 1 ){
      float8_128 value_x = v_f32_ld_tnsr_st_msk((vs + dim0_count * dim0_pad ) / 32 , vmem_input, 1 ,255);
      float8_128 res = v_f32_sel(cmp_core,value_x,value);
      v_f32_st_tnsr_st_msk((vs + dim0_count * dim0_pad ) / 32, vmem_input , 1, 255 , res);
    }
  }
  // 不足1024的处理
  if(vs < dim0_pad){
    int8_128 core_id = get_core_id();
    core_id = core_id + vs;
    bool8_128 cmp_core = 0;
    for(int j = 0 ; j < index_size ; j++){
      float SS = smem_index[j];
      cmp_temp_core =  v_s32_cmp(EQ, core_id, *reinterpret_cast<int*>(&smem_index[j]));
      // cmp_temp_core =  v_s32_cmp(EQ, core_id, smem_index[j]);
      cmp_core = cmp_core | cmp_temp_core;
    }
    for(int dim0_count = 0; dim0_count < math_h ; dim0_count += 1 ){
      int len = dim0_pad - vs;
      int ldst_msk = pre_exp2(len / 128);
      float8_128 value_x = v_f32_ld_tnsr_st_msk((vs + dim0_count * dim0_pad ) / 32 , vmem_input, 1 ,ldst_msk);
      float8_128 res = v_f32_sel(cmp_core,value_x,value);
      v_f32_st_tnsr_st_msk((vs + dim0_count * dim0_pad ) / 32, vmem_input , 1, ldst_msk , res);
    }
  }
}


void main_x86(SIM_X86::DLCMem* mem_info, SIM_X86::DLCTensor* input_hbm0_, 
              SIM_X86::DLCTensor* index_, SIM_X86::DLCTensor* output_hbm_, 
              int* _index_size, 
              int* _dim, float* _value) {
/*##AUTO-GEN##*/TensorFixDims(input_hbm0_);TensorFixDims(index_);TensorFixDims(output_hbm_);/*##END-GEN##*/

	int device_id = get_device_id();

  SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;
  SIM_X86::tensor input_hbm0 = *(SIM_X86::tensor*)input_hbm0_->address;
  SIM_X86::tensor index = *(SIM_X86::tensor*)index_->address;

  unsigned* InputSize = input_hbm0_->shape;

  const unsigned dim0 = InputSize[0];
  const unsigned dim1 = InputSize[1];
  const unsigned dim2 = InputSize[2];
  const unsigned dim3 = InputSize[3];
  const unsigned dim4 = InputSize[4];
  const unsigned dim0_pad = ALIGN128(dim0);

  // int dim = *(int*)_dim;
  // int index_size = *(int*)_index_size;
  // float value = *(float*)_value;

  int h = dim1 * dim2 * dim3 * dim4;
  int UseVMEMsize = 3072*1024; //默认index_size < dim0

  SIM_X86::tensor vmem_input = *(SIM_X86::tensor*)mem_info->vmem_addr;

  SIM_X86::tensor smem_index = *(SIM_X86::tensor*)mem_info->smem_addr;

  int hbm_len = dim0_pad * dim1 * dim2 * dim3 * dim4 ;

  int index_pad = ALIGN128(index_size);
  const int sync_size = 32 * 128;

  HBM2SMem(index, smem_index, index_pad);

/*对最低维度进行select操作*/
  if(dim == 0){ 
    //使用双xys
    int len_xys = h / 2;
    int xys_offset = 0;
    if(device_id == 1){
        xys_offset = len_xys;
        len_xys = h - len_xys;
    }
    int dma_stride_count = dim0_pad / 128;
    int dma_normal_count = soft_sdiv(h,len_xys);
    if(dma_stride_count <= dma_normal_count || h > 300){
      // int dma_stride = dim0_pad;
      bool8_128 cmp_temp_core;
      for(int dim0_count = 0 ; dim0_count < dim0_pad; dim0_count += 128){
        //每128计算一组mask
        int8_128 core_id = get_core_id();
        core_id = core_id % 128;
        core_id = core_id + dim0_count;
        bool8_128 cmp_core = 0;
        for(int j = 0 ; j < index_size ; j++){
          // float SS = smem_index[j];
          cmp_temp_core =  v_s32_cmp(EQ, core_id, *reinterpret_cast<int*>(&smem_index[j]));
          // cmp_temp_core =  v_s32_cmp(EQ, core_id, smem_index[j]);
          cmp_core = cmp_core | cmp_temp_core;
        }
        //dma_stride传输需要的数据
        for(int l = 0 ; l < len_xys * 128 ; l += UseVMEMsize){
          int VMEMsize = min(len_xys * 128 - l, UseVMEMsize);
          int handle = dlc_dma(tensor_slice(input_hbm0, ((l / 128 * dim0_pad) + (xys_offset * dim0_pad) + dim0_count) / 32), D_HBM,
                        tensor_slice(vmem_input, 0 / 32), D_VMEM, VMEMsize, dim0_pad,
                        128, 128, 7); 
          int ldst_count = 0;
          //sync_size分批vmem
          for(; ldst_count + sync_size <= VMEMsize; ldst_count += sync_size){
            dlc_sync_gte(handle,((ldst_count + sync_size) / 128));
            for(int vs = 0; vs < sync_size; vs += 1024 ){
              float8_128 value_x = v_f32_ld_tnsr_st_msk((vs + ldst_count) / 32 , vmem_input, 1 ,255);
              float8_128 res = v_f32_sel(cmp_core,value_x,value);
              v_f32_st_tnsr_st_msk((vs + ldst_count) / 32, vmem_input , 1, 255 , res);
            }
          }
          if(ldst_count < VMEMsize){
            int last_ldst_size = VMEMsize - ldst_count;
            dlc_sync_gte(handle,(VMEMsize / 128));
            for(int vs = 0; vs < last_ldst_size; vs += 1024 ){
              int len = min(last_ldst_size - vs,1024);
              int ldst_mask = pre_exp2(len / 128);
              float8_128 value_x = v_f32_ld_tnsr_st_msk((vs + ldst_count) / 32 , vmem_input, 1 ,ldst_mask);
              float8_128 res = v_f32_sel(cmp_core,value_x,value);
              v_f32_st_tnsr_st_msk((vs + ldst_count ) / 32, vmem_input , 1, ldst_mask , res);
            }
          }
          //将做好的数据dma_stride放回HBM
          dlc_sync(handle);
          int handle2 = dlc_dma(tensor_slice(vmem_input, 0 / 32), D_VMEM,
                  tensor_slice(output_hbm, ((l / 128 * dim0_pad ) + (xys_offset * dim0_pad) + dim0_count) / 32), D_HBM, VMEMsize, 128,
                  dim0_pad, 128, 7); 
          dlc_sync(handle2);
        }
      }
      sync_device();
    } else {

      //使用vmask实现
      UseVMEMsize = soft_sdiv(UseVMEMsize , dim0_pad) * dim0_pad;
      for(int l = 0 ; l < len_xys * dim0_pad ; l+= UseVMEMsize)
      {
        int VMEMsize = min(len_xys * dim0_pad - l, UseVMEMsize);

        int math_h = soft_sdiv(VMEMsize , dim0_pad);

        HBMtoVMem(input_hbm0, vmem_input, l + (xys_offset * dim0_pad), 0, math_h * dim0_pad);

        vmask_fill(vmem_input, math_h,dim0_pad, smem_index, index_size, value);

        VMEMtoHBM(vmem_input , output_hbm ,  0 , l + (xys_offset * dim0_pad), math_h * dim0_pad);
      }
      sync_device();

    }

  }
/*除了最低维度的其他维度*/
  else{

      int step[6] = {1, dim0_pad, dim0_pad*dim1, dim0_pad*dim1*dim2, dim0_pad*dim1*dim2*dim3, dim0_pad*dim1*dim2*dim3*dim4};
      int len_xys = ALIGN128(hbm_len / 2);
      int xys_offset = 0;
      if(device_id == 1){
          xys_offset = len_xys;
          len_xys = hbm_len - len_xys;
      }
      int dma_start = 0;       
      //将输入tensor全部搬运到输出tensor上
      for(;dma_start < len_xys ; dma_start += UseVMEMsize){

        int VMEMsize = min(len_xys - dma_start, UseVMEMsize);

        HBMtoVMem(input_hbm0 , vmem_input, dma_start + xys_offset, 0, VMEMsize);

        VMEMtoHBM(vmem_input, output_hbm, 0, dma_start + xys_offset, VMEMsize);
        
      }
      sync_device();
      float8_128 fill_value = value;
      v_f32_st_tnsr_st_msk(0,vmem_input,1,1,fill_value);
      //       __attribute__((unused)) volatile float wait = vstore_wait(fill_value);
      //外部循环的dma的offset
      int dma_normal_count = soft_sdiv(step[5] , step[dim + 1]);
      int dma_stride_count = soft_sdiv(step[dim] , 128);
      //根据不同的dma数据量选用不同的dma方式
      if(dma_stride_count < dma_normal_count){

        int xys_count = dma_stride_count / 2;
        int xys_count_offset = 0;
        if(device_id == 1){
            xys_count_offset = xys_count;
            xys_count = dma_stride_count - xys_count;
        }
        //dma的外部循环
        for(int j = 0 ; j < xys_count; j ++) {

          //index的内部循环offset
          for(int i = 0 ; i < index_size ; i++){
            float SS = smem_index[i];
            int index_offset = *reinterpret_cast<int*>(&smem_index[i]);
            // int index_offset = smem_index[i];

            //dma传输需要fill的值到输出Hbm里
            dlc_dma(tensor_slice(vmem_input, 0 / 32), D_VMEM,
                                tensor_slice(output_hbm, ((j + xys_count_offset) * 128 + index_offset * step[dim] ) / 32), D_HBM, 128 * dma_normal_count, 0,
                                step[dim + 1], 128, 7);

          }
        }

        int handle = dlc_dma(tensor_slice(vmem_input, 0 / 32), D_VMEM,
                                tensor_slice(output_hbm, 0), D_HBM, 0, 128,
                                128, 128, 7);
        dlc_sync(handle);	
      } else {

        int xys_count = dma_normal_count / 2;
        int xys_count_offset = 0;
        if(device_id == 1){
            xys_count_offset = xys_count;
            xys_count = dma_normal_count - xys_count;
        }
              

        for(int j = 0 ; j < xys_count; j ++) {

          //index的内部循环offset
          for(int i = 0 ; i < index_size ; i++){
            float SS = smem_index[i];
            int index_offset = *reinterpret_cast<int*>(&smem_index[i]);
            // int index_offset = smem_index[i];

            // dma传输需要fill的值到输出Hbm里
            dlc_dma(tensor_slice(vmem_input, 0 / 32), D_VMEM,
                                tensor_slice(output_hbm, (index_offset * step[dim] + (j + xys_count_offset) * step[dim + 1] ) / 32), D_HBM, step[dim], 0,
                                128, 128, 7);
            // dlc_sync(handle);	

          }
        }
        int handle = dlc_dma(tensor_slice(vmem_input, 0 / 32), D_VMEM,
                                tensor_slice(output_hbm, 0), D_HBM, 0, 128,
                                128, 128, 7);
        dlc_sync(handle);	

      }
  }
  sync_device();
}
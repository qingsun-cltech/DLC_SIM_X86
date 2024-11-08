#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "convert_element_type.h"

#include "ldst.h"
const int D_SMEM = 0;
#ifdef USE_CMEM
const int D_HBM = 3;
#else
const int D_HBM = 1;
#endif
const int D_VMEM = 2;
inline float8_128 $F(int8_128 a) {
    float8_128 result0 = *(float8_128*)(&a);
    return result0;
}
inline void HBM2VMem(SIM_X86::tensor hbm_address, SIM_X86::tensor vmem_address, int length) {
    int handle = dlc_dma(hbm_address, D_HBM, vmem_address, D_VMEM, length, 128, 128, 128, 7);
    dlc_sync(handle);
}
inline void Vmem2HBM(SIM_X86::tensor vmem_address, SIM_X86::tensor hbm_address, int length) {
    int handle = dlc_dma(vmem_address, D_VMEM, hbm_address, D_HBM, length, 128, 128, 128, 7);
    dlc_sync(handle);
}
inline void HBM2SMem(SIM_X86::tensor hbm_address, SIM_X86::tensor smem_address, int length) {
    int handle = dlc_dma(hbm_address, D_HBM, (int*)((unsigned)(smem_address) / 128), D_SMEM, length, 128, 128, 128, 7);
    dlc_sync(handle);
}
inline void Vmem2SMem(SIM_X86::tensor vmem_address, SIM_X86::tensor smem_address, int length) {
    int handle = dlc_dma(vmem_address, D_VMEM, (int*)((unsigned)(smem_address) / 128), D_SMEM, length, 128, 128, 128, 7);
    dlc_sync(handle);
}
inline void SMem2HBM(SIM_X86::tensor smem_address, SIM_X86::tensor hbm_address, int length){
    int handle = dlc_dma((int*)((unsigned)(smem_address) / 128), D_SMEM, hbm_address, D_HBM, length, 128, 128, 128, 7);
    dlc_sync(handle);    
}


inline int8_128 v_s32_max_row(int8_128 a) { return v_cvt_ftoi(v_f32_max_row(v_cvt_itof(a)), 5); }
inline int8_128 v_s32_min_row(int8_128 a) { return v_cvt_ftoi(v_f32_min_row(v_cvt_itof(a)), 5); }

inline int8_128 idx_padded_1024(int8_128 idx, int dim0, int dim0_padded){  
  return idx % dim0_padded + idx / dim0_padded * dim0;
}
inline float8_128 get_data_from_offset(SIM_X86::tensor vmem_in, int8_128 offset, int hbm_address_min, int hbm_address_max){
    float8_128 result = 0;
    hbm_address_min = hbm_address_min / 128 * 128;
    hbm_address_max = ALIGN128(hbm_address_max + 1);
    for(int m = hbm_address_min; m < hbm_address_max; m += 128){
        int8_128 input_index1 = offset - m;   //对input_index进行处理，
        float8_128 input_data = v_f32_ld_tnsr_st_msk(m / 32, vmem_in, 1, 1);  //从vmem_input中取出有效的128
        float8_128 nws = m_f32_perm(input_data, input_index1, 0, 0);

        int8_128 zero = 0;
        int8_128 tmp128 = 127; 
        bool8_128 cmp_zero = v_s32_cmp(LS, input_index1, zero);
        bool8_128 cmp_128 = v_s32_cmp(GT, input_index1, tmp128); 

        nws = v_f32_sel(cmp_zero, nws, 0.0f);  //out[i] = input01[i] ? input03[i] : input02[i];
        nws = v_f32_sel(cmp_128, nws, 0.0f);   //如果为true，则返回0
        result =  v_f32_add_b(result, nws);
    }

    return result;
}





inline int8_128 idx_to_offset_1024(int8_128 idx, int* stride, int* shape, int storage_offset){
  int8_128 strides_contiguous[5];
  int8_128 idx_multi_dim[5];
  strides_contiguous[0] = 1;
  strides_contiguous[1] = strides_contiguous[0] * shape[0];
  strides_contiguous[2] = strides_contiguous[1] * shape[1];
  strides_contiguous[3] = strides_contiguous[2] * shape[2];
  strides_contiguous[4] = strides_contiguous[3] * shape[3]; 

  int8_128 remainder4 = 0;
  int8_128 remainder3 = 0;
  int8_128 remainder2 = 0;
  int8_128 remainder1 = 0;
//   int8_128 remainder0 = 0;

  idx_multi_dim[4] = soft_sdiv_remainder_1024(idx, strides_contiguous[4], &remainder4);
  idx_multi_dim[3] = soft_sdiv_remainder_1024(remainder4, strides_contiguous[3], &remainder3);
  idx_multi_dim[2] = soft_sdiv_remainder_1024(remainder3, strides_contiguous[2], &remainder2);
  idx_multi_dim[1] = soft_sdiv_remainder_1024(remainder2, strides_contiguous[1], &remainder1);
  idx_multi_dim[0] = remainder1; 

  int8_128 offset = storage_offset;
  offset += idx_multi_dim[0] * stride[0];
  offset += idx_multi_dim[1] * stride[1];
  offset += idx_multi_dim[2] * stride[2];
  offset += idx_multi_dim[3] * stride[3];
  offset += idx_multi_dim[4] * stride[4];
  return offset;
}

inline int8_128 to_padded_idx_1024(int8_128 idx, int dim0, int dim0_pad){  
    int8_128 dim0_1024 = v_u32_move_i(dim0);
    int8_128 dim0_padded = v_u32_move_i(dim0_pad);
    int8_128 remainder = 0;
    int8_128 quotient = soft_sdiv_remainder_1024(idx, dim0_1024, &remainder);
    return remainder + quotient * dim0_padded;
}

inline int8_128 long_to_padded_idx_1024(int8_128 idx, int dim0, int dim0_pad){  
    int8_128 dim0_1024 = v_u32_move_i(dim0);
    int8_128 dim0_padded = v_u32_move_i(dim0_pad);
    int8_128 remainder = 0;
    int8_128 quotient = soft_sdiv_remainder_1024(idx, dim0_1024, &remainder);
    return remainder + quotient * dim0_padded;
}

inline int8_128 from_padded_idx_1024(int8_128 idx, int dim0, int dim0_pad){
//   int8_128 dim0_1024 = v_u32_move_i(dim0);
  int8_128 dim0_padded = v_u32_move_i(dim0_pad);
  int8_128 remainder = 0;
  int8_128 quotient = soft_sdiv_remainder_1024(idx, dim0_padded, &remainder);
  return remainder + quotient * dim0;
}


inline void strided_get(SIM_X86::DLCTensor *input_, SIM_X86::DLCTensor *output_, SIM_X86::tensor vmem, int vmem_max){
    SIM_X86::tensor hbm_out = *(SIM_X86::tensor*)output_->address;
    SIM_X86::tensor hbm_in = *(SIM_X86::tensor*)input_->address;
    
    int dim1_i = input_->dim1;
    int dim0_i = input_->dim0;
    int dim0_o = output_->dim0;
    // int dim1_o = output_->dim1;

    int dim0_padded_o = output_->dim0_padded;
    int dim0_padded_i = input_->dim0_padded;
    int src_shape[5];
    int dst_shape[5];
    int src_stride[5];

    for(int i = 0; i < 5; i++) {
        src_shape[i] = input_->shape[i];
        src_stride[i] = input_->stride[i];
        dst_shape[i] = output_->shape[i];
    }
    int src_storage_offset = input_->storage_offset;
    int dst_storage_offset = output_->storage_offset;

    // int out_hbm_length = dim0_padded_o * dim1_o;
    int out_hbm_length = dst_shape[0] * dst_shape[1] * dst_shape[2] * dst_shape[3] * dst_shape[4];
    // int numel = dim0_i * dim1_i;

    SIM_X86::tensor vmem_in = vmem;
    SIM_X86::tensor vmem_out = (SIM_X86::tensor)((int)vmem + vmem_max / 32);
    // 方便分批，保证vmem的大小是输出最后一维
    // int vmem_out_max = soft_sdiv(vmem_max, dim0_padded_o) * dim0_padded_o;
    int vmem_out_max = vmem_max / 128 * 128;
    int vmem_in_max = soft_sdiv(vmem_max, dim0_padded_i) * dim0_padded_i;

    int input_hbm_start = 0;
    int input_hbm_end = 0;

    int input_hbm_max = 0;
    int input_hbm_min = 0;
    int input_hbm_max_next = 0;
    int input_hbm_min_next = 0;
    int8_128 temp_max = 0;
    int8_128 temp_min = 0;

    int8_128 idx_next_128 = 0;
    int8_128 valid_idx_128 = 0;
    int8_128 offset_128 = 0;
    int8_128 offset_128_next = 0;
    int dst_storage_offset_rem = 0;
    int dst_storage_offset_res = soft_sdiv_remainder(dst_storage_offset, dim0_o, &dst_storage_offset_rem);
    int dst_storage_offset_align = dst_storage_offset_res * dim0_padded_o;
    // output在dst_malloc中的起始地址，向下对齐到dim0_padded_o
    int output_hbm_start = dst_storage_offset_align;
    // output在dst_malloc中的结束地址，向上对齐到dim0_padded_o
    int output_hbm_end = soft_sdiv(dst_storage_offset + out_hbm_length + dim0_o - 1, dim0_o) * dim0_padded_o;
    int input_hbm_max_end = dim1_i * dim0_padded_i;
    for(int i = output_hbm_start; i < output_hbm_end; i += vmem_out_max) {
        int handle_size_vmem = min(vmem_out_max, output_hbm_end - i);
        idx_next_128 = v_u32_move_i(i - output_hbm_start) + get_core_id();
        // 计算有效idx排布
        valid_idx_128 = from_padded_idx_1024(idx_next_128, dim0_o, dim0_padded_o) - dst_storage_offset_rem;
        valid_idx_128 = v_s32_sel(v_s32_cmp(GTEQ, valid_idx_128, 0), 0, valid_idx_128);
        // 将idx pad部分的值设置为pad_value
        int8_128 idx_pad_value = soft_sdiv_1024(idx_next_128 + dim0_padded_o, dim0_padded_o) * dim0_o - 1;
        valid_idx_128 = v_s32_sel(v_s32_cmp(LS, valid_idx_128, idx_pad_value), idx_pad_value, valid_idx_128);
        offset_128_next = to_padded_idx_1024(idx_to_offset_1024(valid_idx_128, src_stride, src_shape, src_storage_offset), dim0_i, dim0_padded_i);
        
        // 计算出第一批数据的hbm地址范围
        temp_min = v_s32_min_row(offset_128_next);
        input_hbm_min_next = temp_min[0];
        temp_max = v_s32_max_row(offset_128_next);
        input_hbm_max_next = temp_max[0];
        input_hbm_start = input_hbm_min_next / 128 * 128;
        int vmem_in_size_handle = min(vmem_in_max, input_hbm_max_end - input_hbm_start);
        HBM2VMem(hbm_in + input_hbm_start / 32, vmem_in, vmem_in_size_handle);

        input_hbm_end = input_hbm_start + vmem_in_max;

        for(int j = 0; j < handle_size_vmem; j += 128) {
          // 计算出当前处理数据在vmem中的偏移、max、min
          input_hbm_min = input_hbm_min_next - input_hbm_start;
          input_hbm_max = input_hbm_max_next - input_hbm_start;
          offset_128 = offset_128_next - input_hbm_start;

          // 使用permute从vmem中取出数据
          float8_128 result_128 = get_data_from_offset(vmem_in, offset_128, input_hbm_min, input_hbm_max);
          v_f32_st_tnsr_st_msk(j / 32, vmem_out, 1 , 1 , result_128);

          // 计算下一批数据的hbm地址范围
          idx_next_128 += 128;
          valid_idx_128 = from_padded_idx_1024(idx_next_128, dim0_o, dim0_padded_o) - dst_storage_offset_rem;
          int8_128 idx_pad_value = soft_sdiv_1024(idx_next_128 + dim0_padded_o, dim0_padded_o) * dim0_o - 1;

          valid_idx_128 = v_s32_sel(v_s32_cmp(LS, valid_idx_128, idx_pad_value), idx_pad_value, valid_idx_128);
          offset_128_next = to_padded_idx_1024(idx_to_offset_1024(valid_idx_128, src_stride, src_shape, src_storage_offset), dim0_i, dim0_padded_i);
          temp_min = v_s32_min_row(offset_128_next);
          input_hbm_min_next = temp_min[0];
          temp_max = v_s32_max_row(offset_128_next);
          input_hbm_max_next = temp_max[0];

          // 如果下一批数据的hbm地址范围超出当前vmem的范围，则重新加载vmem
          if(ALIGN128(input_hbm_max_next) > input_hbm_end || input_hbm_min_next < input_hbm_start){
            input_hbm_start = input_hbm_min_next / 128 * 128;
            int vmem_in_size_handle_this = min(vmem_in_max, input_hbm_max_end - input_hbm_start);
            HBM2VMem(hbm_in + input_hbm_start / 32, vmem_in, vmem_in_size_handle_this);
            input_hbm_end = input_hbm_start + vmem_in_size_handle_this;
          }
        }
        Vmem2HBM(vmem_out, hbm_out + i / 32, handle_size_vmem);
    }
}

inline void strided_get_bf16(SIM_X86::DLCTensor *input_, SIM_X86::DLCTensor *output_, SIM_X86::tensor vmem, int vmem_max){
    SIM_X86::tensor hbm_out = *(SIM_X86::tensor*)output_->address;
    SIM_X86::tensor hbm_in = *(SIM_X86::tensor*)input_->address;
    
    int dim0_i = input_->dim0;
    // int dim0_i_padded = input_->dim0_padded;
    int dim1_i = input_->dim1;

    int dim0_o = output_->dim0;
    // int dim1_o = output_->dim1;

    int dim0_padded_o = output_->dim0_padded;
    int dim0_padded_i = input_->dim0_padded;
    int src_shape[5];
    int dst_shape[5];
    int src_stride[5];

    for(int i = 0; i < 5; i++) {
        src_shape[i] = input_->shape[i];
        src_stride[i] = input_->stride[i];
        dst_shape[i] = output_->shape[i];
    }
    int src_storage_offset = input_->storage_offset;
    int dst_storage_offset = output_->storage_offset;

    // int out_hbm_length = dim0_padded_o * dim1_o;
    int out_hbm_length = dst_shape[0] * dst_shape[1] * dst_shape[2] * dst_shape[3] * dst_shape[4];
    // int numel = dim0_i * dim1_i;
    int input_hbm_max_end = dim1_i * dim0_padded_i;

    // vmem_max /= 2;
    SIM_X86::tensor vmem_in = vmem;
    SIM_X86::tensor vmem_out = (SIM_X86::tensor)((int)vmem + vmem_max / 32);
    // 方便分批，保证vmem的大小是输出最后一维
    // int vmem_out_max = soft_sdiv(vmem_max, dim0_padded_o) * dim0_padded_o;
    int vmem_out_max = vmem_max / 256 * 256;
    int vmem_in_max = soft_sdiv(vmem_max, dim0_padded_i) * dim0_padded_i;

    int input_hbm_start = 0;
    int input_hbm_end = 0;

    int input_hbm_max = 0;
    int input_hbm_min = 0;
    int input_hbm_max_next = 0;
    int input_hbm_min_next = 0;
    int8_128 temp_max = 0;
    int8_128 temp_min = 0;

    int8_128 idx_next_128 = 0;
    int8_128 valid_idx_128 = 0;
    int8_128 offset_128 = 0;
    int8_128 offset_128_next = 0;
    int dst_storage_offset_rem = 0;
    int dst_storage_offset_res = soft_sdiv_remainder(dst_storage_offset, dim0_o, &dst_storage_offset_rem);
    int dst_storage_offset_align = dst_storage_offset_res * dim0_padded_o;
    // int dst_storage_offset_bf16 = 
    // output在dst_malloc中的起始地址，向下对齐到dim0_padded_o（第output_hbm_start个bf16）
    int output_hbm_start = dst_storage_offset_align;
    // 在hbm中的起始地址
    output_hbm_start /= 2;
    // output在dst_malloc中的结束地址，向上对齐到dim0_padded_o（第output_hbm_end个bf16）
    int output_hbm_end = soft_sdiv(dst_storage_offset + out_hbm_length + dim0_o - 1, dim0_o) * dim0_padded_o;

    // 在hbm中的结束地址
    output_hbm_end /= 2;
    // int idx_start = dst_storage_offset_align;
    for(int i = output_hbm_start; i < output_hbm_end; i += vmem_out_max) {
        idx_next_128 = v_u32_move_i(i - output_hbm_start) * 2 + get_core_id();
        // 计算有效idx排布
        valid_idx_128 = from_padded_idx_1024(idx_next_128, dim0_o, dim0_padded_o) - dst_storage_offset_rem;
        valid_idx_128 = v_s32_sel(v_s32_cmp(GTEQ, valid_idx_128, 0), 0, valid_idx_128);
        // 将idx pad部分的值设置为pad_value
        int8_128 idx_pad_value = soft_sdiv_1024(idx_next_128 + dim0_padded_o, dim0_padded_o) * dim0_o - 1;
        valid_idx_128 = v_s32_sel(v_s32_cmp(LS, valid_idx_128, idx_pad_value), idx_pad_value, valid_idx_128);
        offset_128_next = to_padded_idx_1024(idx_to_offset_1024(valid_idx_128, src_stride, src_shape, src_storage_offset), dim0_i, dim0_padded_i);
        
        // 计算出第一批数据的hbm地址范围
        temp_min = v_s32_min_row(offset_128_next);
        input_hbm_min_next = temp_min[0];
        temp_max = v_s32_max_row(offset_128_next);
        input_hbm_max_next = temp_max[0];
        input_hbm_start = input_hbm_min_next / 256 * 256;
        int vmem_in_size_handle = min(vmem_in_max, input_hbm_max_end - input_hbm_start);

        HBM2VMem(hbm_in + input_hbm_start / 64, vmem_in, vmem_in_size_handle);
        // __f32ToBf16_256(vmem_in, vmem_in, vmem_in_max / 2, false);
        __bf16ToF32_256(vmem_in, vmem_in, vmem_in_size_handle / 2);
        
        input_hbm_end = input_hbm_start + vmem_in_size_handle;

        int handle_size_vmem = min(vmem_out_max, (output_hbm_end - i) * 2);
        for(int j = 0; j < handle_size_vmem; j += 128) {
          // 计算出当前处理数据在vmem中的偏移、max、min
          input_hbm_min = input_hbm_min_next - input_hbm_start;
          input_hbm_max = input_hbm_max_next - input_hbm_start;
          offset_128 = offset_128_next - input_hbm_start;

          // 使用permute从vmem中取出数据
          float8_128 result_128 = get_data_from_offset(vmem_in, offset_128, input_hbm_min, input_hbm_max);
          v_f32_st_tnsr_st_msk(j / 32, vmem_out, 1 , 1 , result_128);

          // 计算下一批数据的hbm地址范围
          idx_next_128 += 128;
          valid_idx_128 = from_padded_idx_1024(idx_next_128, dim0_o, dim0_padded_o) - dst_storage_offset_rem;
          int8_128 idx_pad_value = soft_sdiv_1024(idx_next_128 + dim0_padded_o, dim0_padded_o) * dim0_o - 1;
          valid_idx_128 = v_s32_sel(v_s32_cmp(LS, valid_idx_128, idx_pad_value), idx_pad_value, valid_idx_128);
          offset_128_next = to_padded_idx_1024(idx_to_offset_1024(valid_idx_128, src_stride, src_shape, src_storage_offset), dim0_i, dim0_padded_i);
          temp_min = v_s32_min_row(offset_128_next);
          input_hbm_min_next = temp_min[0];
          temp_max = v_s32_max_row(offset_128_next);
          input_hbm_max_next = temp_max[0];
          // 如果下一批数据的hbm地址范围超出当前vmem的范围，则重新加载vmem
          if(ALIGN128(input_hbm_max_next) > input_hbm_end || input_hbm_min_next < input_hbm_start){
            input_hbm_start = input_hbm_min_next / 128 * 128;
            int vmem_in_size_handle_this = min(vmem_in_max, input_hbm_max_end - input_hbm_start);

            HBM2VMem(hbm_in + input_hbm_start / 64, vmem_in, vmem_in_size_handle_this);

            __f32ToBf16_256(vmem_in, vmem_in, vmem_in_size_handle_this / 2, false);

            input_hbm_end = input_hbm_start + vmem_in_size_handle_this;
          }
        }
        // __bf16ToF32_256(vmem_out, vmem_out, handle_size_vmem);
        __f32ToBf16_256(vmem_out, vmem_out, handle_size_vmem, false);

        Vmem2HBM(vmem_out, hbm_out + i / 64, handle_size_vmem / 2);
    }
}

// smem handle
inline void strided_get_long(SIM_X86::DLCTensor *input_, SIM_X86::DLCTensor *output_, SIM_X86::tensor vmem, int vmem_max, SIM_X86::tensor smem, int smem_idx_size){
     SIM_X86::tensor hbm_out = *(SIM_X86::tensor*)output_->address;
    SIM_X86::tensor hbm_in = *(SIM_X86::tensor*)input_->address;
    
    int dim0_i = input_->dim0;
    int dim0_padded_i = input_->dim0_padded;

    int dim1_i = input_->dim1;

    int dim0_o = output_->dim0;
    int dim0_padded_o = output_->dim0_padded;
    int dim1_o = output_->dim1;

    int dst_shape[5];
    int dst_stride[5];
    int src_shape[5];
    int src_stride[5];

    for(int i = 0; i < 5; i++) {
        dst_shape[i] = output_->shape[i];
        dst_stride[i] = output_->stride[i];
        src_shape[i] = input_->shape[i];
        src_stride[i] = input_->stride[i];
    }
    int dst_storage_offset = output_->storage_offset;
    int src_storage_offset = input_->storage_offset;
    // long type
    int in_hbm_length = dim0_padded_i * dim1_i;
    int out_hbm_length = dim0_padded_o * dim1_o;
    int numel = dst_shape[0] * dst_shape[1] * dst_shape[2] * dst_shape[3] * dst_shape[4]; // numel same as src
    int numel_aligned = ALIGN1024(numel);

    // 将smem分成六份 smem_in_data占两份， smem_in_offset占一份，smem_out_data占两份， smem_out_offset占一份，
    smem_idx_size = smem_idx_size / 1024 * 1024;

    SIM_X86::tensor smem_in_data  = *(SIM_X86::tensor*)(smem);
    SIM_X86::tensor smem_in_offset = (smem_in_data + smem_idx_size * 4);
    SIM_X86::tensor smem_out_data = (smem_in_offset + smem_idx_size * 4 * 2);
    SIM_X86::tensor smem_out_offset = (smem_out_data + smem_idx_size * 4);


    // vmem一批处理的size由smem决定，后续可能可以优化
    SIM_X86::tensor vmem_in_offset = *(SIM_X86::tensor*)(vmem);
    SIM_X86::tensor vmem_out_offset = (vmem_in_offset + vmem_max /32);





    for(int i = 0; i < numel_aligned; i += 1024){
        int8_128 idx_1024 = get_core_id() + v_u32_move_i(i);
        int8_128 offset_dst_1024 = to_padded_idx_1024(idx_1024 + dst_storage_offset, dim0_o, dim0_padded_o);
        int8_128 offset_src_1024 = to_padded_idx_1024(idx_to_offset_1024(idx_1024, src_stride, src_shape, src_storage_offset), dim0_i, dim0_padded_i);
        v_f32_st_tnsr_b(i / 32, vmem_out_offset,  $F(offset_dst_1024));
        v_f32_st_tnsr_b(i / 32, vmem_in_offset,  $F(offset_src_1024));
    }
    Vmem2SMem(vmem_out_offset, smem_out_offset, numel_aligned);
    Vmem2SMem(vmem_in_offset, smem_in_offset, numel_aligned);

    // HBM2SMem(hbm_out, smem_out_data, out_hbm_length);
    // HBM2SMem(hbm_in, smem_in_data, in_hbm_length);
    int handle = dlc_dma(hbm_out, D_HBM, (int*)((unsigned)(smem_out_data) / 128), D_SMEM, out_hbm_length, 256, 128, 128, 7);
    dlc_sync(handle);
    handle = dlc_dma(hbm_in, D_HBM, (int*)((unsigned)(smem_in_data) / 128), D_SMEM, in_hbm_length, 256, 128, 128, 7);
    dlc_sync(handle);
    // #pragma clang loop unroll_count(32)
    for(int i = 0; i < numel; i++){
      ((int*)smem_out_data)[((int*)smem_out_offset)[i]] = ((int*)smem_in_data)[((int*)(smem_in_offset))[i]];
    }
    handle = dlc_dma((int*)((unsigned)(smem_out_data) / 128), D_SMEM, hbm_out, D_HBM, out_hbm_length, 128, 256, 128, 7);
    dlc_sync(handle);

    handle = dlc_dma(hbm_out + 4, D_HBM, (int*)((unsigned)(smem_out_data) / 128), D_SMEM, out_hbm_length, 256, 128, 128, 7);
    dlc_sync(handle);
    handle = dlc_dma(hbm_in + 4, D_HBM, (int*)((unsigned)(smem_in_data) / 128), D_SMEM, in_hbm_length, 256, 128, 128, 7);
    dlc_sync(handle);
    // #pragma clang loop unroll_count(32)
    for(int i = 0; i < numel; i++){
      ((int*)smem_out_data)[((int*)smem_out_offset)[i]] = ((int*)smem_in_data)[((int*)(smem_in_offset))[i]];
    }
    handle = dlc_dma((int*)((unsigned)(smem_out_data) / 128), D_SMEM, hbm_out + 4, D_HBM, out_hbm_length, 128, 256, 128, 7);
    dlc_sync(handle);

    // SMem2HBM(smem_out_data, hbm_out, out_hbm_length);
}

inline void strided_unordered_set(SIM_X86::DLCTensor *input_, SIM_X86::DLCTensor *output_, SIM_X86::tensor vmem, SIM_X86::tensor smem, int smem_in_size){
  SIM_X86::tensor hbm_out = *(SIM_X86::tensor*)output_->address;
  SIM_X86::tensor hbm_in = *(SIM_X86::tensor*)input_->address;
  
  int dim0_i = input_->dim0;
  // int dim1_i = input_->dim1;

  int dim0_o = output_->dim0;
  int dim1_o = output_->dim1;

  int dim0_padded_o = output_->dim0_padded;
  int dim0_padded_i = input_->dim0_padded;
  int src_shape[5];
  int dst_shape[5];
  int dst_stride[5];

  for(int i = 0; i < 5; i++) {
    src_shape[i] = input_->shape[i];
    dst_stride[i] = output_->stride[i];
    dst_shape[i] = output_->shape[i];
  }
  int src_storage_offset = input_->storage_offset;
  int dst_storage_offset = output_->storage_offset;

  int in_hbm_length = src_shape[0] * src_shape[1] * src_shape[2] * src_shape[3] * src_shape[4];
  int smem_out_size = smem_in_size * 2;
  // input一批的size对齐到dim0_padded_i
  smem_in_size = soft_sdiv(smem_in_size, dim0_padded_i) * dim0_padded_i;
  smem_out_size = smem_out_size / 128 * 128;

  // 将smem分成四份 smem_out占两份 smem_in_data占一份 smem_in_index占一份
  float* smem_in_data  = (float*)smem;
  int* smem_in_index = (int*)((unsigned)smem_in_data + smem_in_size * 4);
  float* smem_out      = (float*)((unsigned)smem_in_index + smem_in_size * 4);

  // vmem一批处理的size由smem决定，后续可能可以优化
  int vmem_max = smem_in_size;
  SIM_X86::tensor vmem_in_index = vmem;

  int hbm_address_start = 0;
  int hbm_address_end = 0;
  int hbm_address_end_max = dim0_padded_o * dim1_o;
  int hbm_address_max_next = 0;
  int hbm_address_min_next = 0;
  int8_128 idx_1024 = 0;
  int8_128 valid_idx_1024 = 0;
  int8_128 offset_1024 = 0;

  int src_storage_offset_rem = 0;
  int src_storage_offset_res = soft_sdiv_remainder(src_storage_offset, dim0_i, &src_storage_offset_rem);
  int src_storage_offset_align = src_storage_offset_res * dim0_padded_i;
  int input_hbm_start = src_storage_offset_align;

  int input_hbm_end = soft_sdiv(src_storage_offset + in_hbm_length + dim0_i - 1, dim0_i) * dim0_padded_i;
  int src_storage_offset_last_rem = 0;
  soft_sdiv_remainder(src_storage_offset + in_hbm_length, dim0_i, &src_storage_offset_last_rem);
  if(src_storage_offset_last_rem == 0){
    src_storage_offset_last_rem = dim0_i;
  }
  for(int i = input_hbm_start; i < input_hbm_end; i += vmem_max) {
    int handle_size_vmem = min(vmem_max, input_hbm_end - i);
    for(int j = 0; j < handle_size_vmem; j += 1024) {
        int handle_size = min(1024, handle_size_vmem - j);
        int ldst_vmask = pre_exp2(handle_size / 128);
        idx_1024 = get_core_id() + v_u32_move_i(i + j - input_hbm_start);
        valid_idx_1024 = from_padded_idx_1024(idx_1024, dim0_i, dim0_padded_i) - src_storage_offset_rem;
        offset_1024 = to_padded_idx_1024(idx_to_offset_1024(valid_idx_1024, dst_stride, dst_shape, dst_storage_offset), dim0_o, dim0_padded_o);
        store8_128_stride_with_stmask_i(j / 32, 1, ldst_vmask, vmem_in_index, offset_1024);
    }
    //将计算得到的下标和相应的数据传入smem中
    Vmem2SMem(vmem_in_index, smem_in_index, handle_size_vmem);
    HBM2SMem(hbm_in + i / 32, *(SIM_X86::tensor*)smem_in_data, handle_size_vmem);
    hbm_address_min_next = smem_in_index[0];
    if(i == input_hbm_start){
      hbm_address_min_next = smem_in_index[src_storage_offset_rem];
    }
    int d = 0;
    if(i == input_hbm_start){
      d = src_storage_offset_rem;
    }
    for(; d < dim0_i; d++){
      hbm_address_min_next = min(hbm_address_min_next, smem_in_index[d]);     
    }

    hbm_address_start = hbm_address_min_next / 128 * 128;
    int smem_out_size_handle = min(smem_out_size, hbm_address_end_max - hbm_address_start);

    HBM2SMem(hbm_out + hbm_address_start / 32, smem_out, smem_out_size_handle);
    hbm_address_end = hbm_address_start + smem_out_size_handle;
    for(int c = 0; c < handle_size_vmem; c += dim0_padded_i){
      int dim0_i_start = 0;
      int d = dim0_i;

      // 整个input第一个dim0_i
      if(c == 0 && i == input_hbm_start){
        dim0_i_start = src_storage_offset_rem;
      }
      // 整个input最后一个dim0_i
      if(i + c + dim0_padded_i >= input_hbm_end){
        d = src_storage_offset_last_rem;
      }


      for(; dim0_i_start < d; dim0_i_start++){
        int real_idx = c + dim0_i_start;
        int idx_temp = smem_in_index[real_idx] - hbm_address_start;

        float temp = smem_in_data[real_idx];
        smem_out[idx_temp] = temp;
      }


      // 判断当前处理的dim0_padded_i是否为handle_size_vmem的最后一行
      if(c + dim0_padded_i < handle_size_vmem){
        // 判断下一批是否是input的最后一行
        if(i + c + 2 * dim0_padded_i >= input_hbm_end){
          d = src_storage_offset_last_rem;
        }
        // 找到下一批的最大最小值
        hbm_address_max_next = smem_in_index[c + dim0_padded_i];
        hbm_address_min_next = smem_in_index[c + dim0_padded_i];
        for(int a = 0; a < d; a++){
          int temp = smem_in_index[c + a + dim0_padded_i];
          hbm_address_max_next = max(hbm_address_max_next, temp);
          hbm_address_min_next = min(hbm_address_min_next, temp);
        }
        if(ALIGN128(hbm_address_max_next) > hbm_address_end || ALIGN128(hbm_address_min_next) < hbm_address_start){
          SMem2HBM(smem_out, hbm_out + hbm_address_start / 32, smem_out_size_handle);
          hbm_address_start = hbm_address_min_next / 128 * 128;
          int smem_out_size_this = min(smem_out_size, hbm_address_end_max - hbm_address_start);
          HBM2SMem(hbm_out + hbm_address_start / 32, smem_out, smem_out_size_this);
          hbm_address_end = hbm_address_start + smem_out_size_this;
        }
      }
    }

    smem_out_size_handle = min(smem_out_size, hbm_address_end_max - hbm_address_start);
    SMem2HBM(smem_out, hbm_out + hbm_address_start / 32, smem_out_size_handle);

  }
}

inline void strided_ordered_set(SIM_X86::DLCTensor *input_, SIM_X86::DLCTensor *output_, SIM_X86::tensor vmem, SIM_X86::tensor smem, int smem_in_size){
  SIM_X86::tensor hbm_out = *(SIM_X86::tensor*)output_->address;
  SIM_X86::tensor hbm_in = *(SIM_X86::tensor*)input_->address;
  
  int dim0_i = input_->dim0;
  // int dim1_i = input_->dim1;

  int dim0_o = output_->dim0;
  int dim1_o = output_->dim1;

  int dim0_padded_o = output_->dim0_padded;
  int dim0_padded_i = input_->dim0_padded;
  int src_shape[5];
  int dst_shape[5];
  int dst_stride[5];

  for(int i = 0; i < 5; i++) {
    src_shape[i] = input_->shape[i];
    dst_stride[i] = output_->stride[i];
    dst_shape[i] = output_->shape[i];
  }
  int src_storage_offset = input_->storage_offset;
  int dst_storage_offset = output_->storage_offset;

  int in_hbm_length = src_shape[0] * src_shape[1] * src_shape[2] * src_shape[3] * src_shape[4];
  int smem_out_size = smem_in_size * 2;
  // input一批的size对齐到dim0_padded_i
  smem_in_size = soft_sdiv(smem_in_size, dim0_padded_i) * dim0_padded_i;
  smem_out_size = smem_out_size / 128 * 128;

  // 将smem分成四份 smem_out占两份 smem_in_data占一份 smem_in_index占一份
  float* smem_in_data  = (float*)smem;
  int* smem_in_index = (int*)((unsigned)smem_in_data + smem_in_size * 4);
  float* smem_out      = (float*)((unsigned)smem_in_index + smem_in_size * 4);

  // vmem一批处理的size由smem决定，后续可能可以优化
  int vmem_max = smem_in_size;
  SIM_X86::tensor vmem_in_index = vmem;

  int hbm_address_start = 0;
  int hbm_address_end = 0;
  int hbm_address_end_max = dim0_padded_o * dim1_o;
  int hbm_address_max_next = 0;
  int hbm_address_min_next = 0;
  int8_128 idx_1024 = 0;
  int8_128 valid_idx_1024 = 0;
  int8_128 offset_1024 = 0;

  int src_storage_offset_rem = 0;
  int src_storage_offset_res = soft_sdiv_remainder(src_storage_offset, dim0_i, &src_storage_offset_rem);
  int src_storage_offset_align = src_storage_offset_res * dim0_padded_i;
  int input_hbm_start = src_storage_offset_align;

  int input_hbm_end = soft_sdiv(src_storage_offset + in_hbm_length + dim0_i - 1, dim0_i) * dim0_padded_i;
  int src_storage_offset_last_rem = 0;
  soft_sdiv_remainder(src_storage_offset + in_hbm_length, dim0_i, &src_storage_offset_last_rem);
  // src_storage_offset_last_rem += dim0_i * (src_storage_offset_last_rem == 0);
  if(src_storage_offset_last_rem == 0){
    src_storage_offset_last_rem = dim0_i;
  }
  for(int i = input_hbm_start; i < input_hbm_end; i += vmem_max) {
    int handle_size_vmem = min(vmem_max, input_hbm_end - i);
    for(int j = 0; j < handle_size_vmem; j += 1024) {
        int handle_size = min(1024, handle_size_vmem - j);
        int ldst_vmask = pre_exp2(handle_size / 128);
        idx_1024 = get_core_id() + v_u32_move_i(i + j - input_hbm_start);
        valid_idx_1024 = from_padded_idx_1024(idx_1024, dim0_i, dim0_padded_i) - src_storage_offset_rem;
        offset_1024 = to_padded_idx_1024(idx_to_offset_1024(valid_idx_1024, dst_stride, dst_shape, dst_storage_offset), dim0_o, dim0_padded_o);
        store8_128_stride_with_stmask_i(j / 32, 1, ldst_vmask, vmem_in_index, offset_1024);
    }
    //将计算得到的下标和相应的数据传入smem中

    Vmem2SMem(vmem_in_index, smem_in_index, handle_size_vmem);
    HBM2SMem(hbm_in + i / 32, *(SIM_X86::tensor*)smem_in_data, handle_size_vmem);


    hbm_address_min_next = smem_in_index[0];
    if(i == input_hbm_start){
      hbm_address_min_next = smem_in_index[src_storage_offset_rem];
    }
    hbm_address_start = hbm_address_min_next / 128 * 128;
    smem_out_size = min(smem_out_size, hbm_address_end_max - hbm_address_start);
    HBM2SMem(hbm_out + hbm_address_start / 32, smem_out, smem_out_size);
    hbm_address_end = hbm_address_start + smem_out_size;

    for(int c = 0; c < handle_size_vmem; c += dim0_padded_i){
      int dim0_i_start = 0;
      int dim0_i_end   = dim0_i;
      hbm_address_min_next = smem_in_index[c + dim0_padded_i];
      hbm_address_max_next = smem_in_index[c + dim0_padded_i + dim0_i - 1];
      if(c == 0 && i == input_hbm_start){
        dim0_i_start = src_storage_offset_rem;
      }
      if(c + dim0_padded_i >= handle_size_vmem){
        hbm_address_max_next = hbm_address_end;
        if(i + c + dim0_padded_i >= input_hbm_end){
          dim0_i_end = src_storage_offset_last_rem;
        }
      }
      for(; dim0_i_start < dim0_i_end; dim0_i_start++){
        int real_idx = c + dim0_i_start;
        int idx_temp = smem_in_index[real_idx] - hbm_address_start;
        float temp = smem_in_data[real_idx];
        smem_out[idx_temp] = temp;
      }

      if(ALIGN128(hbm_address_max_next) > hbm_address_end){
        SMem2HBM(smem_out, hbm_out + hbm_address_start / 32, smem_out_size);
        hbm_address_start = hbm_address_min_next / 128 * 128;
        smem_out_size = min(smem_out_size, hbm_address_end_max - hbm_address_start);
        HBM2SMem(hbm_out + hbm_address_start / 32, smem_out, smem_out_size);
        hbm_address_end = hbm_address_start + smem_out_size;
      }
    }
    SMem2HBM(smem_out, hbm_out + hbm_address_start / 32, smem_out_size);
  }
}



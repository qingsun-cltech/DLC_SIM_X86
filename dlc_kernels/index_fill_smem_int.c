
// #include "bf16.h"
// #include "ldst.h"
// #include "typehint.h"

// #include "kernel_arg_types.h"
// #include "align.h"

#include "../x86.h"

const int D_SMEM = 0;


// void main(DLCMem *mem_info, DLCTensor *input_hbm0_, DLCTensor *index_, DLCTensor *output_hbm_, int *_index_size  ,int *dim,
//            int *value) {
void main_x86(SIM_X86::DLCMem* mem_info, SIM_X86::DLCTensor* input_hbm0_, 
              SIM_X86::DLCTensor* index_, SIM_X86::DLCTensor* output_hbm_, 
              int* _index_size, 
              int* dim, int* value) {
/*##AUTO-GEN##*/TensorFixDims(input_hbm0_);TensorFixDims(index_);TensorFixDims(output_hbm_);/*##END-GEN##*/

  SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;
  SIM_X86::tensor input_hbm0 = *(SIM_X86::tensor*)input_hbm0_->address;
  SIM_X86::tensor index = *(SIM_X86::tensor*)index_->address;

  unsigned* InputSize = input_hbm0_->shape;

  unsigned dim0 = InputSize[0];
  unsigned dim1 = InputSize[1];
  unsigned dim2 = InputSize[2];
  unsigned dim3 = InputSize[3];
  unsigned dim4 = InputSize[4];
  unsigned dim0_pad = ALIGN128(dim0);

  int len = dim0_pad * dim1 * dim2 * dim3 * dim4;

  SIM_X86::tensor smem_input = *(SIM_X86::tensor*)mem_info->smem_addr;
  SIM_X86::tensor smem_index = smem_input + len / 32;
  // int* smem_input = (void*)mem_info->smem_addr;

  // int* smem_index = (void*)(mem_info->smem_addr + len * 4);

  if (dim[0] < 0) dim[0] += 5;  // range in torch is [-5, 4] for 5D tensor

  int step[5] = {1, dim0_pad, dim0_pad * dim1, dim0_pad * dim1 * dim2,
               dim0_pad * dim1 * dim2 * dim3};

  /* 从hbm读index的数据到smem */
  int handle_index = dlc_dma(index, HBM, smem_index, D_SMEM,
                             ALIGN128(_index_size[0]), 128, 128, 128, 7);
  dlc_sync(handle_index);
  // Print("smem_index = ", smem_index, 16, PrintType::INT);

  /* dma传输回所需要的值 */
  int handle1 = dlc_dma(input_hbm0, HBM, smem_input, D_SMEM, len, 128,
                        128, 128, 7);
  dlc_sync(handle1);
  // Print("smem_input = ", smem_input, 16, PrintType::FLOAT);


  // #pragma clang loop unroll_count(1)
  if (dim[0] == 4) dim4 = _index_size[0];
  if (dim[0] == 3) dim3 = _index_size[0];
  if (dim[0] == 2) dim2 = _index_size[0];
  if (dim[0] == 1) dim1 = _index_size[0];
  if (dim[0] == 0) dim0 = _index_size[0];

  /* 采用smem对单个位置进行fill */
  int coords[5] = {0};
  // #pragma clang loop unroll_count(1)
  for (coords[4] = 0; coords[4] < dim4; ++coords[4]) {
    // #pragma clang loop unroll_count(2)
    for (coords[3] = 0; coords[3] < dim3; ++coords[3]) {
      // #pragma clang loop unroll_count(1)
      for (coords[2] = 0; coords[2] < dim2; ++coords[2]) {
        // #pragma clang loop unroll_count(1)
        for (coords[1] = 0; coords[1] < dim1; ++coords[1]) {
          // #pragma clang loop unroll_count(8)
          for (coords[0] = 0; coords[0] < dim0; ++coords[0]) {
            // #pragma clang loop unroll_count(2)
            int origin = coords[dim[0]];
            coords[dim[0]] = *reinterpret_cast<int*>(&smem_index[coords[dim[0]]]);
            int coords_temp = s_u32_mul(coords[0], step[0]) + s_u32_mul(coords[1], step[1]) + s_u32_mul(coords[2], step[2]) +
                              s_u32_mul(coords[3], step[3]) + s_u32_mul(coords[4], step[4]);
            // smem_input[coords_temp] = value[0];
            smem_input.data_ptr[coords_temp] = *reinterpret_cast<float*>(&value[0]);
            // smem_input.data_ptr[coords_temp] = *value;
            coords[dim[0]] = origin;
          }
        }
      }
    }
  }

  /* dma传输回去需要的值 */
  int handle2 = dlc_dma(smem_input, D_SMEM, output_hbm, HBM, len,
                        128, 128, 128, 7);
  dlc_sync(handle2);
}
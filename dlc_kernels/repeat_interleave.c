#include "repeat_interleave.h"

/**
 * input0_hbm_: input tensor
 * input1_hbm_: input permute tensor, used for dim = 0
 * input2_hbm_: repeats input permute tensor, used for dim = 0
 * input3_hbm_: repeats tensor, 如果 repeats 是 int，input3_hbm_->shape[0] = 1, input3_hbm_ => repeats，即用 shape[1] 来存储 repeats
 * dim_: select dim
*/
void main_x86(SIM_X86::DLCMem *INFO,
              SIM_X86::DLCTensor *input0_hbm_,
              SIM_X86::DLCTensor *input1_hbm_,
              SIM_X86::DLCTensor *input2_hbm_,
              SIM_X86::DLCTensor *input3_hbm_,
              SIM_X86::DLCTensor *output_hbm_,
              int *repeats_, int *dim_) {
  /* get tensor shape */
  unsigned* input0_shape = input0_hbm_->shape;

  /* set input & output hbm */
  SIM_X86::tensor input0_hbm = *(SIM_X86::tensor*)input0_hbm_->address;
  SIM_X86::tensor input1_hbm = *(SIM_X86::tensor*)input1_hbm_->address;
  SIM_X86::tensor input2_hbm = *(SIM_X86::tensor*)input2_hbm_->address;
  SIM_X86::tensor repeats_hbm = *(SIM_X86::tensor*)input3_hbm_->address;
  SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;

  /* set vmem */
  const int VMEMSIZE = min((int)INFO->vmem_size / 4, 3072 * 1024);
  SIM_X86::tensor input0_vmem = *(SIM_X86::tensor*)(INFO->vmem_addr);

  /* set cmem */
  // const int CMEMSIZE = 1024 * 4096;
  // void *input0_cmem = (void *)(INFO->cmem_addr);

  /* get args */
  int dim = *dim_;
  int repeats = *repeats_;

  /**
   * TODO: special judge for small cases(vmem is big enough)
  */

  // printf("repeats = %d\n", repeats);
  // printf("dim = %d\n", dim);

  if (repeats >= 0) { // repeats = int
    if (dim == 0) {
      RepeatInterleaveDim0Int(input0_hbm, input1_hbm, input2_hbm, output_hbm,
                              input0_vmem, VMEMSIZE,
                              repeats, input0_hbm_->dim0, input0_hbm_->dim1);
    } else if (dim > 0) {
      RepeatInterleaveInt(input0_hbm, output_hbm,
                          input0_vmem, VMEMSIZE,
                          dim, repeats, (int *)input0_shape);
    } else {
      RepeatInterleaveArrayIntDefault(input0_hbm, input1_hbm, input2_hbm, output_hbm,
                                      input0_vmem, VMEMSIZE,
                                      repeats, input0_hbm_->dim0, input0_hbm_->dim1);
    }
  } else { // repeats = tensor
    if (dim == 0) {
      RepeatInterleaveDim0Tensor(input0_hbm, input1_hbm, input2_hbm, output_hbm,
                                 input0_vmem, VMEMSIZE,
                                 repeats_hbm, input0_hbm_->dim0, input0_hbm_->dim1);
    } else if (dim > 0) {
      RepeatInterleaveTensor(input0_hbm, output_hbm, 
                             input0_vmem, VMEMSIZE,
                             dim, repeats_hbm, (int *)input0_shape);
    } else {
      RepeatInterleaveArrayTensorDefault(input0_hbm, input1_hbm, input2_hbm, output_hbm,
                                          input0_vmem, VMEMSIZE,
                                          repeats_hbm, input0_hbm_->dim0, input0_hbm_->dim1);
    }
  }

  sync_device();
}
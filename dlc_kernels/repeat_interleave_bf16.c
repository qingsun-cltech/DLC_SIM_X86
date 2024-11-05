#include "repeat_interleave.h"

/**
 * input0_hbm_: input tensor
 * input1_hbm_: input permute tensor, used for dim = 0
 * input2_hbm_: repeats input permute tensor, used for dim = 0
 * input3_hbm_: repeats tensor
 * output_hbm_: output tensor
 * dim_:
 * repeats_:
*/
void main(DLCMem *INFO, DLCTensor *input0_hbm_, DLCTensor *input1_hbm_, DLCTensor *input2_hbm_,
          DLCTensor *input3_hbm_, DLCTensor *output_hbm_, int *repeats_, int *dim_) {
  /* get tensor shape */
  unsigned* input0_shape = input0_hbm_->shape;

  /* set input & output hbm */
  void *input0_hbm = input0_hbm_->address;
  void *input1_hbm = input1_hbm_->address;
  void *input2_hbm = input2_hbm_->address;
  void *repeats_hbm = input3_hbm_->address;
  void *output_hbm = output_hbm_->address;

  /* set vmem */
  const int VMEMSIZE = min((int)INFO->vmem_size / 4, 3072 * 1024);
  void *input0_vmem = (void *)(INFO->vmem_addr);

  /* set cmem */
  // const int CMEMSIZE = 2 * 1024 * 4096;
  // void *input0_cmem = (void *)(INFO->cmem_addr);

  /* get args */
  int dim = *dim_;
  int repeats = *repeats_;

  if (repeats >= 0) { // repeats = int
    if (dim == 0) {
      RepeatInterleaveDim0IntBf16(input0_hbm, input1_hbm, input2_hbm, output_hbm,
                                  input0_vmem, VMEMSIZE,
                                  repeats, input0_hbm_->dim0, input0_hbm_->dim1);
    } else if (dim > 0) {
      input0_shape[0] = ALIGN256(input0_shape[0]) / 2;
      RepeatInterleaveInt(input0_hbm, output_hbm,
                          input0_vmem, VMEMSIZE,
                          dim, repeats, (int *)input0_shape);
    } else {
      RepeatInterleaveArrayIntDefaultBf16(input0_hbm, input1_hbm, input2_hbm, output_hbm,
                                          input0_vmem, VMEMSIZE,
                                          repeats, input0_hbm_->dim0, input0_hbm_->dim1);
    }
  } else { // repeats = tensor
    if (dim == 0) {
      RepeatInterleaveDim0TensorBf16(input0_hbm, input1_hbm, input2_hbm, output_hbm,
                                      input0_vmem, VMEMSIZE,
                                      repeats_hbm, input0_hbm_->dim0, input0_hbm_->dim1);
    } else if (dim > 0) {
      input0_shape[0] = ALIGN256(input0_shape[0]) / 2;
      RepeatInterleaveTensor(input0_hbm, output_hbm, 
                             input0_vmem, VMEMSIZE,
                             dim, repeats_hbm, (int *)input0_shape);
    } else {
      RepeatInterleaveArrayTensorDefaultBf16(input0_hbm, input1_hbm, input2_hbm, output_hbm,
                                              input0_vmem, VMEMSIZE,
                                              repeats_hbm, input0_hbm_->dim0, input0_hbm_->dim1);
    }
  }

  sync_device();
}
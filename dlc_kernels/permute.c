// #include "kernel_arg_types.h"
// #include "permute.h"
// #include "typehint.h"

#include "../x86.h"

void main_x86(SIM_X86::DLCMem *mem,
              SIM_X86::DLCTensor *input0_hbm_, SIM_X86::DLCTensor *output_hbm_,
              SIM_X86::DLCScalar *new_dim) {
/*##AUTO-GEN##*/TensorFixDims(input0_hbm_);TensorFixDims(output_hbm_);/*##END-GEN##*/
  SIM_X86::tensor input0_hbm = *(SIM_X86::tensor*)input0_hbm_->address;
  SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;
  SIM_X86::tensor input0_vmem = *(SIM_X86::tensor*)mem->vmem_addr;

  int VMEMSize = min(1024 * 3072, mem->vmem_size / 4);
  int dim[5] = {input0_hbm_->dim0, input0_hbm_->shape[1], input0_hbm_->shape[2], input0_hbm_->shape[3], input0_hbm_->shape[4]};
  int perm[5] = {new_dim[0].value, new_dim[1].value, new_dim[2].value, new_dim[3].value, new_dim[4].value};

  if (input0_hbm_->dim0_padded * input0_hbm_->dim1 + output_hbm_->dim0_padded * output_hbm_->dim1 > VMEMSize) {
    /* new permute: support any shap with two xys, data from hbm*/
    _permute_hbm(input0_hbm, output_hbm, input0_vmem, VMEMSize, dim, perm);
  } else {
    /* old permute: shape can not exceed (1024 * 3072 / 2) */
    int sync = dlc_dma(input0_hbm, HBM, input0_vmem, VMEM, input0_hbm_->dim0_padded * input0_hbm_->dim1, 128, 128, 128, 7);
    dlc_sync(sync);

    SIM_X86::tensor output_vmem = *(SIM_X86::tensor*)mem->vmem_addr + (input0_hbm_->dim0_padded * input0_hbm_->dim1 / 32);
    _permute(input0_vmem, output_vmem, input0_hbm_->shape, input0_hbm_->dim0, perm[4], perm[3], perm[2], perm[1], perm[0]);

    int sync2 = dlc_dma(output_vmem, VMEM, output_hbm, HBM, output_hbm_->dim0_padded * output_hbm_->dim1, 128, 128, 128, 7);
    dlc_sync(sync2);
  }

  sync_device();
}
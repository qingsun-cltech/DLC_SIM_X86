#include "x86.h"


void main_x86(SIM_X86::DLCMem *INFO,
              SIM_X86::DLCTensor *input0_hbm_,
              SIM_X86::DLCTensor *output_hbm_) {
  SIM_X86::tensor input0_hbm = *(SIM_X86::tensor*)input0_hbm_->address;
  SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;

  SIM_X86::tensor smem = *(SIM_X86::tensor*)INFO->smem_addr;
  SIM_X86::tensor vmem = *(SIM_X86::tensor*)INFO->vmem_addr;
  SIM_X86::tensor cmem = *(SIM_X86::tensor*)INFO->cmem_addr;

  auto x = get_core_id();

  v_fxc_store();
}
// #include "convert_element_type.h"
// #include "typehint.h"

#include "../x86.h"

const int D_HBM = 1;
const int D_VMEM = 2;

inline int sz128(unsigned *siz) { return ALIGN128(siz[0]) * siz[1] * siz[2] * siz[3] * siz[4]; }

inline int sz256(unsigned *siz) { return ALIGN256(siz[0]) * siz[1] * siz[2] * siz[3] * siz[4]; }

inline void HBM2VMemWithSync(SIM_X86::tensor hbm_address, SIM_X86::tensor vmem_address, int length) {
    int handle = dlc_dma(hbm_address, D_HBM, vmem_address, D_VMEM, length, 128, 128, 128, 7);
    dlc_sync(handle);
}

inline void Vmem2HBMWithSync(SIM_X86::tensor vmem_address, SIM_X86::tensor hbm_address, int length) {
    int handle = dlc_dma(vmem_address, D_VMEM, hbm_address, D_HBM, length, 128, 128, 128, 7);
    dlc_sync(handle);
}

inline void vmem_cos(SIM_X86::tensor vmem_input, int VMEMsize) {
    int len = (VMEMsize >> 10) << 10;                // VMEMsize / 1024 * 1024
    int ldst_mask = pre_exp2((VMEMsize - len) >> 7); // (VMEMsize - len) / 128
    int vs;

    for (vs = 0; vs < len / 32; vs += 32) {
        float8_128 x = v_f32_ld_tnsr_b(vs, vmem_input);
        float8_128 res = __dlc_cosf(x);
        v_f32_st_tnsr_b(vs, vmem_input, res);
    }

    if (ldst_mask) {
        float8_128 x = v_f32_ld_tnsr_st_msk(vs, vmem_input, 1, ldst_mask);
        float8_128 res = __dlc_cosf(x);
        v_f32_st_tnsr_st_msk(vs, vmem_input, 1, ldst_mask, res);
    }
}

void main_x86(SIM_X86::DLCMem* INFO, SIM_X86::DLCTensor* input0_hbm_, SIM_X86::DLCTensor* output_hbm_) {
/*##AUTO-GEN##*/TensorFixDims(input0_hbm_);TensorFixDims(output_hbm_);/*##END-GEN##*/

    // set data length
    unsigned *Input0Size = input0_hbm_->shape;
    int HbmLength = sz128(Input0Size);

    // set input & output hbm
  SIM_X86::tensor input0_hbm = *(SIM_X86::tensor*)input0_hbm_->address;
  SIM_X86::tensor output_hbm = *(SIM_X86::tensor*)output_hbm_->address;

  // set vmem
  unsigned _VMEMsize = 3072 * 1024;
  SIM_X86::tensor input_vmem = *(SIM_X86::tensor*)(INFO->vmem_addr);

    // set device id
    int device_id = get_device_id();

    // cal offset & dma length
    // xys0 做前半段，xsy1做后半段，前半段的长度是数据长度一半的向下取整
    int offset;
    int DmaLength;
    if (device_id) {
        offset = (HbmLength >> 8) << 7; // HBMLength / 128 / 2 * 128 (向下取整对半)
        DmaLength = HbmLength - offset;
    } else {
        offset = 0;
        DmaLength = (HbmLength >> 8) << 7; // HBMLength / 128 / 2 * 128 (向下取整对半)
    }

    for (int l = 0; l < DmaLength; l += _VMEMsize) {
        int VMEMsize = min(DmaLength - l, _VMEMsize);

        HBM2VMemWithSync(input0_hbm + offset / 32 + l / 32, input_vmem, VMEMsize);

        vmem_cos(input_vmem, VMEMsize);

        Vmem2HBMWithSync(input_vmem, output_hbm + offset / 32 + l / 32, VMEMsize);
    }
    sync_device();
}
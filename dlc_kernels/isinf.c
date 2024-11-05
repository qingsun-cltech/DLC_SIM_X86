//引用头文件
// #include "ldst.h"
//以下两项为必要的define和include

// #include "kernel_arg_types.h"
// #include "libdevice.h"

#include "x86.h"

//核心计算函数
inline void vmem_isinf(SIM_X86::tensor in_out, int len) {

    for (int i = 0; i < len / 1024; i += 1) {
        float8_128 x = v_f32_ld_tnsr_st_msk(i * 32, in_out, 1, 255);
        int8_128 res_int = __dlc_isinff(x);
        bool8_128 isinf_bool = v_s32_cmp(EQ, res_int, 0);
        res_int = v_s32_sel(isinf_bool, 1, 0);
        v_st_generic(i * 32, in_out, 1, 255, res_int);
    }
    if (len % 1024 != 0) {
        int actual_len = len % 1024;
        int ldst_vmask = pre_exp2(actual_len / 128);
        float8_128 x = v_f32_ld_tnsr_st_msk((len - actual_len) / 32, in_out, 1, ldst_vmask);
        int8_128 res_int = __dlc_isinff(x);
        bool8_128 isinf_bool = v_s32_cmp(EQ, res_int, 0);
        res_int = v_s32_sel(isinf_bool, 1, 0);
        v_st_generic((len - actual_len) / 32, in_out, 1, ldst_vmask, res_int);
    }
}

//main函数
//mem_info: 包含vmem、smem、cmem的起始空间以及可用size         
//          vmem_size、cmem_size、smem_size的单位为: 1 byte
//          vmem_addr、cmem_addr的单位为： 128  bytes
//          smem_addr的单位为：            1  byte 
//SIM_X86::DLCTensor: 可以为多个Tensor(通过下标去取 如input_hbm_[i]) 
//每个tensor都固定为五维，shape参数包含了维度信息，shape[0]是最低维，且固定为128的倍数, 其余维size不变;
//若不为128的倍数，会填充到128的倍数,比如[4, 5, 1, 1, 1]会填充成[128, 5, 1, 1, 1]
//参数位置需为:(SIM_X86::DLCMem* mem_info, input的Tensor, output的Tensor)
void main_x86(SIM_X86::DLCMem *mem_info, SIM_X86::DLCTensor *hbm_input0_, SIM_X86::DLCTensor *hbm_output0_) {
/*##AUTO-GEN##*/TensorFixDims(hbm_input0_);TensorFixDims(hbm_output0_);/*##END-GEN##*/

    // HBM SIM_X86::tensor 得到输入和输出Tensor的地址坐标
    void *hbm_input0 = hbm_input0_->address;
    void *hbm_output0 = hbm_output0_->address;

    //得到vmem 可用size和地址坐标
    int VMEMsize_ = ((int)mem_info->vmem_size) / 4;
    int vmem_offset = min(VMEMsize_, 3072 * 1024);
    void *vmem_input0 = (void *)mem_info->vmem_addr;

    //得到输入数据的shape
    unsigned int *InputSize = hbm_input0_->shape;
    int without_dim0 = InputSize[1] * InputSize[2] * InputSize[3] * InputSize[4];
    int total_size = hbm_input0_->dim0_padded * without_dim0;
    int branch_size_count = 1;
    if (without_dim0 != 1) {
        branch_size_count = without_dim0 / 2;
    }

    int xys0_size = branch_size_count * hbm_input0_->dim0_padded;
    int size = xys0_size;

    //通过get_device_id()可以得到当前使用的xys的下标，一块芯片有两个xys，所以下标为0,1
    int device_id = get_device_id();
    if (device_id == 1) {
        size = total_size - xys0_size;
    }

    for (int i = 0; i < size; i += vmem_offset) {
        //1. 通过dma将数据从hbm搬到vmem  
        int len = min(vmem_offset, size - i);
        int flag1 = dlc_dma(tensor_slice(hbm_input0, (i + device_id * xys0_size) / 32), HBM, *(SIM_X86::tensor*)vmem_input0,
                            VMEM, len, 128, 128, 128, 7);
        dlc_sync(flag1);
        //2.计算
        vmem_isinf(*(SIM_X86::tensor*)vmem_input0, len);
        //3.通过dma将数据从vmem搬到hbm
        int flag2 = dlc_dma(*(SIM_X86::tensor*)vmem_input0, VMEM, tensor_slice(hbm_output0, (i + device_id * xys0_size) / 32),
                            HBM, len, 128, 128, 128, 7);
        dlc_sync(flag2);
    }

    //同步两个xys
    sync_device();
}

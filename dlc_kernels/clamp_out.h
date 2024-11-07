#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "bf16.h"
inline void Calculate_stride(unsigned* shape, unsigned* stride) {
    stride[0] = 1;
    stride[1] = stride[0] * shape[0];
    stride[2] = stride[1] * shape[1];
    stride[3] = stride[2] * shape[2];
    stride[4] = stride[3] * shape[3];
}

/**
 * 广播判断，使对应的 stride 为 0
 * 因为最低维度会对齐到128的倍数，所以需要传入对齐前的大小
 * @param shape 张量的大小
 * @param stride 步幅
 * @param dim1 张量最低维对齐前的大小
 * @exception none
 */
inline void broadcast_judge(unsigned* shape, unsigned* stride, unsigned dim1){
    if(dim1 == 1){
        stride[0] = 0;
    }
    for(int i = 1; i < 5; i++){
        if(shape[i] == 1){
            stride[i] = 0;
        }
    }
}

/**
 * 得到要拆分的维度 batch_dim 以及 拆分的大小 dim_split
 * @param shape 输出张量的大小
 * @param vmemsize vmem大小
 * @param dim_split 拆分的大小
 * @return batch_dim 拆分的维度, 0 ：拆分最低维
 * @exception none
 */
inline int Calculate_batchDim(unsigned* shape, unsigned vmemsize, unsigned* dim_split){
    int size = 1, use_vemsize = 128;
    for(int i=0; i < 5; i++){
        size = size * shape[i];
        if(size > vmemsize){
            *dim_split = soft_sdiv(vmemsize , use_vemsize);
            vmemsize = *dim_split * use_vemsize;
            if(i == 0){
                *dim_split = vmemsize;
            }
            return i;
        }
        use_vemsize = size;
    }

    *dim_split = shape[4];

    return 4;
}

/**
 * 计算张量的 MathSzie
 * shape 低 batch_dim 维大小 = MathSzie 低 batch_dim 维大小， MathSzie其余维度为 1
 * Mathsize 的拆分维度 的大小是不确定的，需要在dma过程中给Mathsize 的拆分维度赋值
 * @param batch_dim 一批中进行计算的维度大小
 * @param output_MathSzie 用于计算的 shape
 * @param shape 张量的形状
 * @return none
 * @exception none
 */
inline void Calculate_MathSize(unsigned batch_dim, unsigned* output_MathSzie, unsigned* shape){
    for(int i = 0; i < batch_dim; i++){
        output_MathSzie[i] = shape[i];
    }
}

/**
 * 计算张量的 DmaSize
 * 思路： shape 高 （5 - batch_dim） 维大小 = MathSzie 低 （5 - batch_dim） 维大小
 *         output_DmaSzie 的最低维度是要拆分的维度，所以需要通过对应的 shape 和 dim_split 计算其大小
 * @param batch_dim 要拆分的维度
 * @param output_DmaSzie 用于分批dma传输的张量形状
 * @param shape 张量的形状
 * @param dim_split 要拆分的维度的大小
 * @return none
 * @exception none
 */
inline void Calculate_DmaSize(unsigned batch_dim, unsigned* output_DmaSzie, unsigned* shape){
    for(int i = 0; i < 5 - batch_dim; i++){
        output_DmaSzie[i] = shape[i + batch_dim];
    }
}

/**
 * 得到张量最后一维的下标
 */
inline int get_dimInfo(unsigned* output_shape){
    for(int i=4; i >= 0; i--){
        if(output_shape[i] != 1){
            return i;
        }
    }
    return 0;
}

//*************************************************************************************************************************************/
//*************************************************************************************************************************************/
//*************************************************f32 verison************************************************************************/
//*************************************************************************************************************************************/
//*************************************************************************************************************************************/

inline void Min(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* input0_shape, unsigned* input1_shape, unsigned* output_shape,
                unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,
                int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, output_XysOff;
    /* 最后一维是最低维度，需要 AlIGN256 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN256(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN256(input1_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN256(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(output_shape[output_lastdim] == 128){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }
            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5], output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    float8_128 input_val, min_val, result;
    bool8_128 min_bool;

    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];
                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);
                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input1_HbmOff_Pre = input1_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 1024){
                                len = min(output_MathSzie[0] - dim1, 1024);
                                mask = pre_exp2(len/128);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];
                                                
                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, 1);
                                                    input_val = input_val[0];
                                                }else{
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, mask);
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    min_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, 1);
                                                    min_val = min_val[0];
                                                }else{
                                                    min_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, mask);
                                                }

                                                result = input_val;
                                                min_bool = v_f32_cmp(LS, result, min_val);
                                                result = v_f32_sel(min_bool, result, min_val);

                                                v_f32_st_tnsr_st_msk(output_VmemOff/32, output, 1, mask, result);
                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}



inline void Max(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* input0_shape, unsigned* input1_shape, unsigned* output_shape,
                unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,
                int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, output_XysOff;
    /* 最后一维是最低维度，需要 AlIGN256 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN256(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN256(input1_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN256(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(output_shape[output_lastdim] == 128){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }
            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5], output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    float8_128 input_val, max_val, result;
    bool8_128 max_bool;

    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];
                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);
                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input1_HbmOff_Pre = input1_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 1024){
                                len = min(output_MathSzie[0] - dim1, 1024);
                                mask = pre_exp2(len/128);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];
                                                
                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, 1);
                                                    input_val = input_val[0];
                                                }else{
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, mask);
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    max_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, 1);
                                                    max_val = max_val[0];
                                                }else{
                                                    max_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, mask);
                                                }

                                                result = input_val;
                                                max_bool = v_f32_cmp(GT, result, max_val);
                                                result = v_f32_sel(max_bool, result, max_val);

                                                v_f32_st_tnsr_st_msk(output_VmemOff/32, output, 1, mask, result);
                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}


inline void Min_Max(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm ,SIM_X86::tensor output_hbm,
                unsigned* input0_shape, unsigned* input1_shape, unsigned* input2_shape, unsigned* output_shape,
                unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,unsigned input2_dim1_unpad,
                int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 4;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor input2 = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 3 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, input2_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, input2_XysOff,output_XysOff;
    /* 最后一维是最低维度，需要 AlIGN256 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN256(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN256(input1_shape[output_lastdim]) / 2;
        input2_shape_lastdim = ALIGN256(input2_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN256(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            input2_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            input2_XysOff = 0;     
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(input2_dim1_unpad == 1){
                input2_XysOff = 0;
            }else{
                input2_shape[output_lastdim] = input2_shape[output_lastdim] - input2_shape_lastdim;
                input2_XysOff = input2_shape_lastdim;
            }


            if(output_shape[output_lastdim] == 128){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        input2_shape_lastdim = (input2_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, input2_block = 1,output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            input2_block *= input2_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            input2_shape[output_lastdim] = input2_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            input2_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }

            if(input2_shape[output_lastdim] == 1){
                input2_XysOff = 0;
            }else{
                input2_shape[output_lastdim] = input2_shape[output_lastdim] - input2_shape_lastdim;
                input2_XysOff = input2_shape_lastdim * input2_block;
            }

            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_input2[5] ,stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(input2_shape, stride_input2);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5],input2_DmaStride[5] ,output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    int block_input2 = stride_input2[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    broadcast_judge(input2_shape, stride_input2, input2_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        input2_DmaStride[i] = stride_input2[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, input2_HbmOff ,output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, input2_VmemOff,output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, input2_dmaLength,output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1 ,input2_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    float8_128 input_val, min_val, max_val,result;
    bool8_128 min_bool,max_bool;

    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];
                            input2_HbmOff = dma_dim1 * input2_DmaStride[0] + dma_dim2 * input2_DmaStride[1] + dma_dim3 * input2_DmaStride[2] + 
                                        dma_dim4 * input2_DmaStride[3] + dma_dim5 * input2_DmaStride[4];
                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);
                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input1_HbmOff_Pre = input1_HbmOff;
                            }
                            if(input2_HbmOff != input2_HbmOff_Pre){
                                int remain_size_input2 = min(remain_size, input2_shape[batch_dim]);
                                input2_dmaLength = remain_size_input2 * block_input2;
                                dma_flag = dlc_dma(tensor_slice(input2_hbm, (input2_HbmOff + input2_XysOff) / 32), HBM, input2, VMEM, input2_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input2_HbmOff_Pre = input2_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 1024){
                                len = min(output_MathSzie[0] - dim1, 1024);
                                mask = pre_exp2(len/128);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                input2_VmemOff = dim1 * stride_input2[0] + dim2 * stride_input2[1] + dim3 * stride_input2[2]
                                                                + dim4 * stride_input2[3] + dim5 * stride_input2[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];
                                                
                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, 1);
                                                    input_val = input_val[0];
                                                }else{
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, mask);
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    min_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, 1);
                                                    min_val = min_val[0];
                                                }else{
                                                    min_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, mask);
                                                }
                                                if(input2_dim1_unpad ==1){
                                                    max_val = v_f32_ld_tnsr_st_msk(input2_VmemOff/32, input2, 1, 1);
                                                    max_val = max_val[0];
                                                }else{
                                                    max_val = v_f32_ld_tnsr_st_msk(input2_VmemOff/32, input2, 1, mask);
                                                }

                                                result = input_val;
                                                min_bool = v_f32_cmp(LS, result, min_val);
                                                result = v_f32_sel(min_bool, result, min_val);

                                                max_bool = v_f32_cmp(GT, result, max_val);
                                                result = v_f32_sel(max_bool, result, max_val);

                                                v_f32_st_tnsr_st_msk(output_VmemOff/32, output, 1, mask, result);
                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}




inline void Min_no_broadcast(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 1024){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN256(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    float8_128 input_val, min_val, result;

    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 1024 * 1024;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;
            bool8_128 min_bool;
            for(; offset_vmem < process_length1024; offset_vmem += 1024){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, 255);
                min_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, 255);
                result = input_val;
                min_bool = v_f32_cmp(LS, result, min_val);
                result = v_f32_sel(min_bool, result, min_val);
                v_f32_st_tnsr_st_msk(offset_vmem / 32, output, 1, 255, result);
            }
            int mask = pre_exp2(process_length_remain/128);
            if(mask > 0){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, mask);
                min_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, mask);
                result = input_val;
                min_bool = v_f32_cmp(LS, result, min_val);
                result = v_f32_sel(min_bool, result, min_val);
                v_f32_st_tnsr_st_msk(offset_vmem / 32, output, 1, mask, result);
            }
            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}


inline void Max_no_broadcast(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 1024){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN256(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    float8_128 input_val, max_val, result;

    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 1024 * 1024;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;
            bool8_128 max_bool;
            for(; offset_vmem < process_length1024; offset_vmem += 1024){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, 255);
                max_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, 255);
                result = input_val;
                max_bool = v_f32_cmp(GT, result, max_val);
                result = v_f32_sel(max_bool, result, max_val);
                v_f32_st_tnsr_st_msk(offset_vmem / 32, output, 1, 255, result);
            }
            int mask = pre_exp2(process_length_remain/128);
            if(mask > 0){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, mask);
                max_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, mask);
                result = input_val;
                max_bool = v_f32_cmp(GT, result, max_val);
                result = v_f32_sel(max_bool, result, max_val);
                v_f32_st_tnsr_st_msk(offset_vmem / 32, output, 1, mask, result);
            }
            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}

inline void Min_Max_no_broadcast(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm,SIM_X86::tensor input2_hbm ,SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor input2 = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 1024){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN256(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    float8_128 input_val,min_val ,max_val, result;
    bool8_128 max_bool, min_bool;
    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input2_hbm, (xys_offset + offset) / 32), HBM, input2, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 1024 * 1024;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;

            for(; offset_vmem < process_length1024; offset_vmem += 1024){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, 255);
                min_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, 255);
                max_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input2, 1, 255);
                result = input_val;

                min_bool = v_f32_cmp(LS, result, min_val);
                result = v_f32_sel(min_bool, result, min_val);

                max_bool = v_f32_cmp(GT, result, max_val);
                result = v_f32_sel(max_bool, result, max_val);

                v_f32_st_tnsr_st_msk(offset_vmem / 32, input0, 1, 255, result);
            }
            int mask = pre_exp2(process_length_remain/128);
            if(mask > 0){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, mask);
                min_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, mask);
                max_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input2, 1, mask);

                result = input_val;

                min_bool = v_f32_cmp(LS, result, min_val);
                result = v_f32_sel(min_bool, result, min_val);

                max_bool = v_f32_cmp(GT, result, max_val);
                result = v_f32_sel(max_bool, result, max_val);

                v_f32_st_tnsr_st_msk(offset_vmem / 32, input0, 1, mask, result);
            }
            dma_flag = dlc_dma(input0, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}
//*************************************************************************************************************************************/
//*************************************************************************************************************************************/
//*************************************************bf16 verison************************************************************************/
//*************************************************************************************************************************************/
//*************************************************************************************************************************************/

inline void Min_bf16(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                    unsigned* input0_shape, unsigned* input1_shape, unsigned* output_shape,
                    unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,
                    int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, output_XysOff;
    /* 最后一维是最低维度，需要 AlIGN256 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN256(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN256(input1_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN256(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(output_shape[output_lastdim] == 128){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }
            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5], output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    float8_128 input_val, min_val;
    float8_128 input_val_low, min_val_low, result_low;
    float8_128 input_val_high, min_val_high, result_high;

    bool8_128 min_bool_low;
    bool8_128 min_bool_high;

    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];
                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);
                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input1_HbmOff_Pre = input1_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 1024){
                                len = min(output_MathSzie[0] - dim1, 1024);
                                mask = pre_exp2(len/128);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];
                                                
                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, 1);
                                                    input_val = input_val[0];
                                                    input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                                                    input_val_high = input_val_low;
                                                }else{
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, mask);
                                                    input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                                                    input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    min_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, 1);
                                                    min_val = min_val[0];
                                                    min_val_low = bfloat16_to_float(unpack_16b(__$S(min_val), 0));
                                                    min_val_high = min_val_low;
                                                }else{
                                                    min_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, mask);
                                                    min_val_high = bfloat16_to_float(unpack_16b(__$S(min_val), 1));
                                                    min_val_low = bfloat16_to_float(unpack_16b(__$S(min_val), 0));
                                                }

                                                result_low = input_val_low;
                                                result_high = input_val_high;

                                                min_bool_low = v_f32_cmp(LS, result_low, min_val_low);
                                                result_low = v_f32_sel(min_bool_low, result_low, min_val_low);

                                                min_bool_high = v_f32_cmp(LS, result_high, min_val_high);
                                                result_high = v_f32_sel(min_bool_high, result_high, min_val_high);

                                                v_f32_st_tnsr_st_msk(output_VmemOff/32, output, 1, mask, __$F(float_to_bfloat16(result_high, result_low)));
                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}



inline void Max_bf16(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                    unsigned* input0_shape, unsigned* input1_shape, unsigned* output_shape,
                    unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,
                    int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, output_XysOff;
    /* 最后一维是最低维度，需要 AlIGN256 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN256(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN256(input1_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN256(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(output_shape[output_lastdim] == 128){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }
            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5], output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    float8_128 input_val, max_val;
    float8_128 input_val_high, max_val_high, result_high;
    float8_128 input_val_low, max_val_low, result_low;

    bool8_128 max_bool_high;
    bool8_128 max_bool_low;


    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];
                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);
                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input1_HbmOff_Pre = input1_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 1024){
                                len = min(output_MathSzie[0] - dim1, 1024);
                                mask = pre_exp2(len/128);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];
                                                
                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, 1);
                                                    input_val = input_val[0];
                                                    input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                                                    input_val_high = input_val_low;
                                                }else{
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, mask);
                                                    input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                                                    input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    max_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, 1);
                                                    max_val = max_val[0];
                                                    max_val_low = bfloat16_to_float(unpack_16b(__$S(max_val), 0));
                                                    max_val_high = max_val_low;
                                                }else{
                                                    max_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, mask);
                                                    max_val_high = bfloat16_to_float(unpack_16b(__$S(max_val), 1));
                                                    max_val_low = bfloat16_to_float(unpack_16b(__$S(max_val), 0));
                                                }

                                                result_low = input_val_low;
                                                max_bool_low = v_f32_cmp(GT, result_low, max_val_low);
                                                result_low = v_f32_sel(max_bool_low, result_low, max_val_low);

                                                result_high = input_val_high;
                                                max_bool_high = v_f32_cmp(GT, result_high, max_val_high);
                                                result_high = v_f32_sel(max_bool_high, result_high, max_val_high);

                                                v_f32_st_tnsr_st_msk(output_VmemOff/32, output, 1, mask,  __$F(float_to_bfloat16(result_high, result_low)));
                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}


inline void Min_Max_bf16(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm ,SIM_X86::tensor output_hbm,
                unsigned* input0_shape, unsigned* input1_shape, unsigned* input2_shape, unsigned* output_shape,
                unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,unsigned input2_dim1_unpad,
                int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 4;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor input2 = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 3 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, input2_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, input2_XysOff,output_XysOff;
    /* 最后一维是最低维度，需要 AlIGN256 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN256(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN256(input1_shape[output_lastdim]) / 2;
        input2_shape_lastdim = ALIGN256(input2_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN256(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            input2_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            input2_XysOff = 0;     
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(input2_dim1_unpad == 1){
                input2_XysOff = 0;
            }else{
                input2_shape[output_lastdim] = input2_shape[output_lastdim] - input2_shape_lastdim;
                input2_XysOff = input2_shape_lastdim;
            }


            if(output_shape[output_lastdim] == 128){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        input2_shape_lastdim = (input2_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, input2_block = 1,output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            input2_block *= input2_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            input2_shape[output_lastdim] = input2_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            input2_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }

            if(input2_shape[output_lastdim] == 1){
                input2_XysOff = 0;
            }else{
                input2_shape[output_lastdim] = input2_shape[output_lastdim] - input2_shape_lastdim;
                input2_XysOff = input2_shape_lastdim * input2_block;
            }

            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_input2[5] ,stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(input2_shape, stride_input2);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5],input2_DmaStride[5] ,output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    int block_input2 = stride_input2[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    broadcast_judge(input2_shape, stride_input2, input2_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        input2_DmaStride[i] = stride_input2[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, input2_HbmOff ,output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, input2_VmemOff,output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, input2_dmaLength,output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1 ,input2_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    float8_128 input_val, min_val, max_val;
    float8_128 input_val_high, min_val_high, max_val_high,result_high;
    float8_128 input_val_low, min_val_low, max_val_low,result_low;

    bool8_128 min_bool_high,max_bool_high;
    bool8_128 min_bool_low,max_bool_low;


    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];
                            input2_HbmOff = dma_dim1 * input2_DmaStride[0] + dma_dim2 * input2_DmaStride[1] + dma_dim3 * input2_DmaStride[2] + 
                                        dma_dim4 * input2_DmaStride[3] + dma_dim5 * input2_DmaStride[4];
                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);
                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input1_HbmOff_Pre = input1_HbmOff;
                            }
                            if(input2_HbmOff != input2_HbmOff_Pre){
                                int remain_size_input2 = min(remain_size, input2_shape[batch_dim]);
                                input2_dmaLength = remain_size_input2 * block_input2;
                                dma_flag = dlc_dma(tensor_slice(input2_hbm, (input2_HbmOff + input2_XysOff) / 32), HBM, input2, VMEM, input2_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input2_HbmOff_Pre = input2_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 1024){
                                len = min(output_MathSzie[0] - dim1, 1024);
                                mask = pre_exp2(len/128);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                input2_VmemOff = dim1 * stride_input2[0] + dim2 * stride_input2[1] + dim3 * stride_input2[2]
                                                                + dim4 * stride_input2[3] + dim5 * stride_input2[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];
                                                
                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, 1);
                                                    input_val = input_val[0];
                                                    input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                                                    input_val_high = input_val_low;
                                                }else{
                                                    input_val = v_f32_ld_tnsr_st_msk(input0_VmemOff/32, input0, 1, mask);
                                                    input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                                                    input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    min_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, 1);
                                                    min_val = min_val[0];
                                                    min_val_low = bfloat16_to_float(unpack_16b(__$S(min_val), 0));
                                                    min_val_high = min_val_low;
                                                }else{
                                                    min_val = v_f32_ld_tnsr_st_msk(input1_VmemOff/32, input1, 1, mask);
                                                    min_val_high = bfloat16_to_float(unpack_16b(__$S(min_val), 1));
                                                    min_val_low = bfloat16_to_float(unpack_16b(__$S(min_val), 0));
                                                }
                                                if(input2_dim1_unpad ==1){
                                                    max_val = v_f32_ld_tnsr_st_msk(input2_VmemOff/32, input2, 1, 1);
                                                    max_val = max_val[0];
                                                    max_val_low = bfloat16_to_float(unpack_16b(__$S(max_val), 0));
                                                    max_val_high = max_val_low;
                                                }else{
                                                    max_val = v_f32_ld_tnsr_st_msk(input2_VmemOff/32, input2, 1, mask);
                                                    max_val_high = bfloat16_to_float(unpack_16b(__$S(max_val), 1));
                                                    max_val_low = bfloat16_to_float(unpack_16b(__$S(max_val), 0));
                                                }

                                                result_low = input_val_low;
                                                min_bool_low = v_f32_cmp(LS, result_low, min_val_low);
                                                result_low = v_f32_sel(min_bool_low, result_low, min_val_low);

                                                max_bool_low = v_f32_cmp(GT, result_low, max_val_low);
                                                result_low = v_f32_sel(max_bool_low, result_low, max_val_low);


                                                result_high = input_val_high;
                                                min_bool_high = v_f32_cmp(LS, result_high, min_val_high);
                                                result_high = v_f32_sel(min_bool_high, result_high, min_val_high);

                                                max_bool_high = v_f32_cmp(GT, result_high, max_val_high);
                                                result_high = v_f32_sel(max_bool_high, result_high, max_val_high);


                                                v_f32_st_tnsr_st_msk(output_VmemOff/32, output, 1, mask, __$F(float_to_bfloat16(result_high, result_low)));
                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}




inline void Min_no_broadcast_bf16(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 1024){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN256(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    float8_128 input_val, min_val;
    float8_128 input_val_high, min_val_high, result_high;
    float8_128 input_val_low, min_val_low, result_low;

    bool8_128 min_bool_high,min_bool_low;



    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 1024 * 1024;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;
            for(; offset_vmem < process_length1024; offset_vmem += 1024){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, 255);
                min_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, 255);
                input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                min_val_high = bfloat16_to_float(unpack_16b(__$S(min_val), 1));
                min_val_low = bfloat16_to_float(unpack_16b(__$S(min_val), 0));

                result_low = input_val_low;
                min_bool_low = v_f32_cmp(LS, result_low, min_val_low);
                result_low = v_f32_sel(min_bool_low, result_low, min_val_low);

                result_high = input_val_high;
                min_bool_high = v_f32_cmp(LS, result_high, min_val_high);
                result_high = v_f32_sel(min_bool_high, result_high, min_val_high);


                v_f32_st_tnsr_st_msk(offset_vmem / 32, output, 1, 255, __$F(float_to_bfloat16(result_high, result_low)));
            }
            int mask = pre_exp2(process_length_remain/128);
            if(mask > 0){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, mask);
                min_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, mask);
                input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                min_val_high = bfloat16_to_float(unpack_16b(__$S(min_val), 1));
                min_val_low = bfloat16_to_float(unpack_16b(__$S(min_val), 0));

                result_low = input_val_low;
                min_bool_low = v_f32_cmp(LS, result_low, min_val_low);
                result_low = v_f32_sel(min_bool_low, result_low, min_val_low);

                result_high = input_val_high;
                min_bool_high = v_f32_cmp(LS, result_high, min_val_high);
                result_high = v_f32_sel(min_bool_high, result_high, min_val_high);


                v_f32_st_tnsr_st_msk(offset_vmem / 32, output, 1, mask, __$F(float_to_bfloat16(result_high, result_low)));
            }
            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}


inline void Max_no_broadcast_bf16(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 1024){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN256(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    float8_128 input_val, max_val;
    float8_128 input_val_high, max_val_high, result_high;
    float8_128 input_val_low, max_val_low, result_low;

    bool8_128 max_bool_high,max_bool_low;

    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 1024 * 1024;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;
            for(; offset_vmem < process_length1024; offset_vmem += 1024){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, 255);
                max_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, 255);
                input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                max_val_high = bfloat16_to_float(unpack_16b(__$S(max_val), 1));
                max_val_low = bfloat16_to_float(unpack_16b(__$S(max_val), 0));
                
                result_low = input_val_low;
                max_bool_low = v_f32_cmp(GT, result_low, max_val_low);
                result_low = v_f32_sel(max_bool_low, result_low, max_val_low);

                result_high = input_val_high;
                max_bool_high = v_f32_cmp(GT, result_high, max_val_high);
                result_high = v_f32_sel(max_bool_high, result_high, max_val_high);

                v_f32_st_tnsr_st_msk(offset_vmem / 32, output, 1, 255, __$F(float_to_bfloat16(result_high, result_low)));
            }
            int mask = pre_exp2(process_length_remain/128);
            if(mask > 0){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, mask);
                max_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, mask);
                input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                max_val_high = bfloat16_to_float(unpack_16b(__$S(max_val), 1));
                max_val_low = bfloat16_to_float(unpack_16b(__$S(max_val), 0));
                
                result_low = input_val_low;
                max_bool_low = v_f32_cmp(GT, result_low, max_val_low);
                result_low = v_f32_sel(max_bool_low, result_low, max_val_low);

                result_high = input_val_high;
                max_bool_high = v_f32_cmp(GT, result_high, max_val_high);
                result_high = v_f32_sel(max_bool_high, result_high, max_val_high);

                v_f32_st_tnsr_st_msk(offset_vmem / 32, output, 1, mask, __$F(float_to_bfloat16(result_high, result_low)));
            }
            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}

inline void Min_Max_no_broadcast_bf16(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm,SIM_X86::tensor input2_hbm ,SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;

    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor input2 = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 1024){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN256(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    float8_128 input_val,min_val ,max_val;
    float8_128 input_val_high,min_val_high ,max_val_high, result_high;
    float8_128 input_val_low,min_val_low ,max_val_low, result_low;

    bool8_128 max_bool_high, min_bool_high;
    bool8_128 max_bool_low, min_bool_low;


    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input2_hbm, (xys_offset + offset) / 32), HBM, input2, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 1024 * 1024;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;

            for(; offset_vmem < process_length1024; offset_vmem += 1024){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, 255);
                min_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, 255);
                max_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input2, 1, 255);

                input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                min_val_high = bfloat16_to_float(unpack_16b(__$S(min_val), 1));
                min_val_low = bfloat16_to_float(unpack_16b(__$S(min_val), 0));
                max_val_high = bfloat16_to_float(unpack_16b(__$S(max_val), 1));
                max_val_low = bfloat16_to_float(unpack_16b(__$S(max_val), 0));

                result_low = input_val_low;
                min_bool_low = v_f32_cmp(LS, result_low, min_val_low);
                result_low = v_f32_sel(min_bool_low, result_low, min_val_low);
                max_bool_low = v_f32_cmp(GT, result_low, max_val_low);
                result_low = v_f32_sel(max_bool_low, result_low, max_val_low);

                result_high = input_val_high;
                min_bool_high = v_f32_cmp(LS, result_high, min_val_high);
                result_high = v_f32_sel(min_bool_high, result_high, min_val_high);
                max_bool_high = v_f32_cmp(GT, result_high, max_val_high);
                result_high = v_f32_sel(max_bool_high, result_high, max_val_high);

                v_f32_st_tnsr_st_msk(offset_vmem / 32, input0, 1, 255, __$F(float_to_bfloat16(result_high, result_low)));
            }
            int mask = pre_exp2(process_length_remain/128);
            if(mask > 0){
                input_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input0, 1, mask);
                min_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input1, 1, mask);
                max_val = v_f32_ld_tnsr_st_msk(offset_vmem / 32, input2, 1, mask);

                input_val_high = bfloat16_to_float(unpack_16b(__$S(input_val), 1));
                input_val_low = bfloat16_to_float(unpack_16b(__$S(input_val), 0));
                min_val_high = bfloat16_to_float(unpack_16b(__$S(min_val), 1));
                min_val_low = bfloat16_to_float(unpack_16b(__$S(min_val), 0));
                max_val_high = bfloat16_to_float(unpack_16b(__$S(max_val), 1));
                max_val_low = bfloat16_to_float(unpack_16b(__$S(max_val), 0));

                result_low = input_val_low;
                min_bool_low = v_f32_cmp(LS, result_low, min_val_low);
                result_low = v_f32_sel(min_bool_low, result_low, min_val_low);
                max_bool_low = v_f32_cmp(GT, result_low, max_val_low);
                result_low = v_f32_sel(max_bool_low, result_low, max_val_low);

                result_high = input_val_high;
                min_bool_high = v_f32_cmp(LS, result_high, min_val_high);
                result_high = v_f32_sel(min_bool_high, result_high, min_val_high);
                max_bool_high = v_f32_cmp(GT, result_high, max_val_high);
                result_high = v_f32_sel(max_bool_high, result_high, max_val_high);

                v_f32_st_tnsr_st_msk(offset_vmem / 32, input0, 1, mask, __$F(float_to_bfloat16(result_high, result_low)));
            }
            dma_flag = dlc_dma(input0, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}

//*************************************************************************************************************************************/
//*************************************************************************************************************************************/
//*************************************************long verison************************************************************************/
//*************************************************************************************************************************************/
//*************************************************************************************************************************************/


inline void Min_Long(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                    unsigned* input0_shape, unsigned* input1_shape, unsigned* output_shape,
                    unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,
                    int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;
    /* long */
    vmemSize = vmemSize / 256 * 256;
    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, output_XysOff;
    /* 最后一维是最低维度，需要 AlIGN512 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN512(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN512(input1_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN512(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(output_shape[output_lastdim] == 256){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }
            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5], output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    int8_128 input_val_high, min_val_high, result_high;
    int8_128 input_val_low, min_val_low, result_low;
    int8_128 temp_reg;
    bool8_128 min_bool_high, min_bool_temp;

    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];

                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);

                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);

                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);

                                input1_HbmOff_Pre = input1_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 2048){
                                len = min(output_MathSzie[0] - dim1, 2048);
                                mask = pre_exp2(len/256);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];

                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    temp_reg = load8_128_stride_with_ldmask_i(input0_VmemOff / 32, 1, 255, input0);
                                                    input_val_high = temp_reg[128];
                                                    input_val_low = temp_reg[0];
                                                }else{
                                                    input_val_high = load8_128_stride_with_ldmask_i((input0_VmemOff + 128) / 32, 2, mask, input0);
                                                    input_val_low = load8_128_stride_with_ldmask_i(input0_VmemOff / 32, 2, mask, input0);
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    temp_reg = load8_128_stride_with_ldmask_i(input1_VmemOff / 32, 1, 255, input1);
                                                    min_val_high = temp_reg[128];
                                                    min_val_low = temp_reg[0];
                                                }else{
                                                    min_val_high = load8_128_stride_with_ldmask_i((input1_VmemOff + 128) / 32, 2, mask, input1);
                                                    min_val_low = load8_128_stride_with_ldmask_i(input1_VmemOff / 32, 2, mask, input1);
                                                }
                                                result_high = input_val_high;
                                                result_low = input_val_low;

                                                min_bool_high = v_s32_cmp(LS, result_high, min_val_high);
                                                min_bool_temp = v_s32_cmp(EQ, result_high, min_val_high);
                                                result_high = v_s32_sel(min_bool_high, result_high, min_val_high);
                                                temp_reg = v_s32_sel(min_bool_temp,min_val_low,result_low);
                                                min_bool_temp = v_s32_cmp(LS,temp_reg,min_val_low);
                                                result_low = v_s32_sel(min_bool_high,result_low,min_val_low);
                                                result_low = v_s32_sel(min_bool_temp,result_low,min_val_low);

                                                store8_128_stride_stmk_i((output_VmemOff + 128)/32,2, output,result_high,mask);
                                                store8_128_stride_stmk_i(output_VmemOff/32,2, output,result_low,mask);

                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}



inline void Max_Long(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* input0_shape, unsigned* input1_shape, unsigned* output_shape,
                unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,
                int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;
    //long
    vmemSize = vmemSize / 256 * 256;
    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, output_XysOff;
    /* 最后一维是最低维度，需要 ALIGN512 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN512(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN512(input1_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN512(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(output_shape[output_lastdim] == 256){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }
            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5], output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    int8_128 input_val_high, max_val_high, result_high;
    int8_128 input_val_low, max_val_low, result_low;
    int8_128 temp_reg;
    bool8_128 max_bool_high, max_bool_temp;

    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];
                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);
                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input1_HbmOff_Pre = input1_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 2048){
                                len = min(output_MathSzie[0] - dim1, 2048);
                                mask = pre_exp2(len/256);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];
                                                
                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    temp_reg = load8_128_stride_with_ldmask_i(input0_VmemOff / 32, 1, 255, input0);
                                                    input_val_high = temp_reg[128];
                                                    input_val_low = temp_reg[0];
                                                }else{
                                                    input_val_high = load8_128_stride_with_ldmask_i((input0_VmemOff + 128) / 32, 2, mask, input0);
                                                    input_val_low = load8_128_stride_with_ldmask_i(input0_VmemOff / 32, 2, mask, input0);
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    temp_reg = load8_128_stride_with_ldmask_i(input1_VmemOff / 32, 1, 255, input1);
                                                    max_val_high = temp_reg[128];
                                                    max_val_low = temp_reg[0];
                                                }else{
                                                    max_val_high = load8_128_stride_with_ldmask_i((input1_VmemOff + 128) / 32, 2, mask, input1);
                                                    max_val_low = load8_128_stride_with_ldmask_i(input1_VmemOff / 32, 2, mask, input1);
                                                }

                                                result_high = input_val_high;
                                                result_low = input_val_low;


                                                max_bool_high = v_s32_cmp(GT, result_high, max_val_high);
                                                max_bool_temp = v_s32_cmp(EQ, result_high, max_val_high);
                                                result_high = v_s32_sel(max_bool_high, result_high, max_val_high);
                                                temp_reg = v_s32_sel(max_bool_temp,max_val_low,result_low);
                                                max_bool_temp = v_s32_cmp(GT,temp_reg,max_val_low);
                                                result_low = v_s32_sel(max_bool_temp,result_low,max_val_low);
                                                result_low = v_s32_sel(max_bool_temp,result_low,max_val_low);

                                                store8_128_stride_stmk_i((output_VmemOff + 128)/32,2, output,result_high,mask);
                                                store8_128_stride_stmk_i(output_VmemOff/32,2, output,result_low,mask);

                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}


inline void Min_Max_Long(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm ,SIM_X86::tensor output_hbm,
                unsigned* input0_shape, unsigned* input1_shape, unsigned* input2_shape, unsigned* output_shape,
                unsigned input0_dim1_unpad, unsigned input1_dim1_unpad,unsigned input2_dim1_unpad,
                int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 4;
    unsigned vmemSize = available_vmemSize / vmem_num;
    //long
    vmemSize = vmemSize / 256 * 256;
    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor input2 = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 3 * vmemSize / 32);

    // 双 xys
    int use_xys;
    int output_lastdim = get_dimInfo(output_shape);
    int input0_shape_lastdim, input1_shape_lastdim, input2_shape_lastdim, output_shape_lastdim;
    int input0_XysOff, input1_XysOff, input2_XysOff,output_XysOff;
    /* 最后一维是最低维度，需要 ALIGN512 再对半分，xys1 可能不使用 */
    if(output_lastdim == 0){
        input0_shape_lastdim = ALIGN512(input0_shape[output_lastdim]) / 2;
        input1_shape_lastdim = ALIGN512(input1_shape[output_lastdim]) / 2;
        input2_shape_lastdim = ALIGN512(input2_shape[output_lastdim]) / 2;
        output_shape_lastdim = ALIGN512(output_shape[output_lastdim]) / 2;
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            input2_shape[output_lastdim] = input1_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            input2_XysOff = 0;     
            output_XysOff = 0;
            use_xys = 1;
        }else{
            /* 根据最后一维是否需要广播，最后一维是否需要切分
               切分：input_XysOff 偏移，output shape 的最后一维对半分，两个xys各自处理自己的 */
            if(input0_dim1_unpad == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim;
            }

            if(input1_dim1_unpad == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim;
            }

            if(input2_dim1_unpad == 1){
                input2_XysOff = 0;
            }else{
                input2_shape[output_lastdim] = input2_shape[output_lastdim] - input2_shape_lastdim;
                input2_XysOff = input2_shape_lastdim;
            }


            if(output_shape[output_lastdim] == 256){
                use_xys = 0;
            }else{
                use_xys = 1;
                output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
                output_XysOff = output_shape_lastdim;
            }
        }
    }else{
        /* 最后一维不是最低维的情况，一定会使用双xys */
        input0_shape_lastdim = (input0_shape[output_lastdim] + 1) / 2;
        input1_shape_lastdim = (input1_shape[output_lastdim] + 1) / 2;
        input2_shape_lastdim = (input2_shape[output_lastdim] + 1) / 2;
        output_shape_lastdim = (output_shape[output_lastdim] + 1) / 2;
        int input0_block = 1, input1_block = 1, input2_block = 1,output_block = 1;
        for(int i = 0; i < output_lastdim; i++){
            input0_block *= input0_shape[i];
            input1_block *= input1_shape[i];
            input2_block *= input2_shape[i];
            output_block *= output_shape[i];
        }
        if(device_id == 0){
            input0_shape[output_lastdim] = input0_shape_lastdim;
            input1_shape[output_lastdim] = input1_shape_lastdim;
            input2_shape[output_lastdim] = input2_shape_lastdim;
            output_shape[output_lastdim] = output_shape_lastdim;
            input0_XysOff = 0;
            input1_XysOff = 0;
            input2_XysOff = 0;
            output_XysOff = 0;
            use_xys = 1;
        }else{
            if(input0_shape[output_lastdim] == 1){
                input0_XysOff = 0;
            }else{
                input0_shape[output_lastdim] = input0_shape[output_lastdim] - input0_shape_lastdim;
                input0_XysOff = input0_shape_lastdim * input0_block;
            }

            if(input1_shape[output_lastdim] == 1){
                input1_XysOff = 0;
            }else{
                input1_shape[output_lastdim] = input1_shape[output_lastdim] - input1_shape_lastdim;
                input1_XysOff = input1_shape_lastdim * input1_block;
            }

            if(input2_shape[output_lastdim] == 1){
                input2_XysOff = 0;
            }else{
                input2_shape[output_lastdim] = input2_shape[output_lastdim] - input2_shape_lastdim;
                input2_XysOff = input2_shape_lastdim * input2_block;
            }

            use_xys = 1;
            output_shape[output_lastdim] = output_shape[output_lastdim] - output_shape_lastdim;
            output_XysOff = output_shape_lastdim * output_block;
        }
    }
    
    // stride
    unsigned stride_input0[5], stride_input1[5], stride_input2[5] ,stride_output[5];
    Calculate_stride(input0_shape, stride_input0);
    Calculate_stride(input1_shape, stride_input1);
    Calculate_stride(input2_shape, stride_input2);
    Calculate_stride(output_shape, stride_output);
    // batch
    unsigned dim_split, batch_dim;
    unsigned output_MathSzie[5] = {1, 1, 1, 1, 1};
    unsigned output_DmaSzie[5] = {1, 1, 1, 1, 1};
    unsigned input0_DmaStride[5], input1_DmaStride[5],input2_DmaStride[5] ,output_DmaStride[5];
    batch_dim = Calculate_batchDim(output_shape, vmemSize, &dim_split);
    Calculate_MathSize(batch_dim, output_MathSzie, output_shape);
    Calculate_DmaSize(batch_dim, output_DmaSzie, output_shape);
    // broadcast
    /* block_input0 是分批维度前数据的大小*/
    int block_input0 = stride_input0[batch_dim];
    int block_input1 = stride_input1[batch_dim];
    int block_input2 = stride_input2[batch_dim];
    /* shape 为 1的 stride 置0
        需要先计算stride，再调用 broadcast_judge */
    broadcast_judge(input0_shape, stride_input0, input0_dim1_unpad);
    broadcast_judge(input1_shape, stride_input1, input1_dim1_unpad);
    broadcast_judge(input2_shape, stride_input2, input2_dim1_unpad);
    for(int i = 0; i < 5 - batch_dim; i++){
        input0_DmaStride[i] = stride_input0[i + batch_dim];
        input1_DmaStride[i] = stride_input1[i + batch_dim];
        input2_DmaStride[i] = stride_input2[i + batch_dim];
        output_DmaStride[i] = stride_output[i + batch_dim];
    }
    // calculate
    unsigned input0_HbmOff, input1_HbmOff, input2_HbmOff ,output_HbmOff;
    unsigned input0_VmemOff, input1_VmemOff, input2_VmemOff,output_VmemOff;
    unsigned input0_dmaLength, input1_dmaLength, input2_dmaLength,output_dmaLength;
    int input0_HbmOff_Pre = -1, input1_HbmOff_Pre = -1 ,input2_HbmOff_Pre = -1;
    int dma_flag, len, mask;
    
    int8_128 input_val_high, min_val_high ,max_val_high, result_high;
    int8_128 input_val_low, min_val_low ,max_val_low, result_low;
    int8_128 temp_reg;
    bool8_128 min_bool_high, min_bool_temp;
    bool8_128 max_bool_high, max_bool_temp;

    if(use_xys){
        // the outer five layers loop to do DMA batch
        for(int dma_dim5 = 0; dma_dim5 < output_DmaSzie[4]; dma_dim5++){
            for(int dma_dim4 = 0; dma_dim4 < output_DmaSzie[3]; dma_dim4++){
                for(int dma_dim3 = 0; dma_dim3 < output_DmaSzie[2]; dma_dim3++){
                    for(int dma_dim2 = 0; dma_dim2 < output_DmaSzie[1]; dma_dim2++){
                        for(int dma_dim1 = 0; dma_dim1 < output_DmaSzie[0]; dma_dim1 += dim_split){
                            /* offset */
                            input0_HbmOff = dma_dim1 * input0_DmaStride[0] + dma_dim2 * input0_DmaStride[1] + dma_dim3 * input0_DmaStride[2] + 
                                        dma_dim4 * input0_DmaStride[3] + dma_dim5 * input0_DmaStride[4];
                            input1_HbmOff = dma_dim1 * input1_DmaStride[0] + dma_dim2 * input1_DmaStride[1] + dma_dim3 * input1_DmaStride[2] + 
                                        dma_dim4 * input1_DmaStride[3] + dma_dim5 * input1_DmaStride[4];
                            input2_HbmOff = dma_dim1 * input2_DmaStride[0] + dma_dim2 * input2_DmaStride[1] + dma_dim3 * input2_DmaStride[2] + 
                                        dma_dim4 * input2_DmaStride[3] + dma_dim5 * input2_DmaStride[4];
                            int remain_size = min(dim_split, output_DmaSzie[0] - dma_dim1);
                            /* 如果要拆分的维度需要广播，可以只做1次dma */
                            if(input0_HbmOff != input0_HbmOff_Pre){
                                /* 如果要拆分维度为1（需要广播），拆分的大小应该是 1，而不是拆分的大小 remain_size */
                                int remain_size_input0 = min(remain_size, input0_shape[batch_dim]);
                                input0_dmaLength = remain_size_input0 * block_input0;
                                dma_flag = dlc_dma(tensor_slice(input0_hbm, (input0_HbmOff + input0_XysOff) / 32), HBM, input0, VMEM, input0_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input0_HbmOff_Pre = input0_HbmOff;
                            }
                            if(input1_HbmOff != input1_HbmOff_Pre){
                                int remain_size_input1 = min(remain_size, input1_shape[batch_dim]);
                                input1_dmaLength = remain_size_input1 * block_input1;
                                dma_flag = dlc_dma(tensor_slice(input1_hbm, (input1_HbmOff + input1_XysOff) / 32), HBM, input1, VMEM, input1_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input1_HbmOff_Pre = input1_HbmOff;
                            }
                            if(input2_HbmOff != input2_HbmOff_Pre){
                                int remain_size_input2 = min(remain_size, input2_shape[batch_dim]);
                                input2_dmaLength = remain_size_input2 * block_input2;
                                dma_flag = dlc_dma(tensor_slice(input2_hbm, (input2_HbmOff + input2_XysOff) / 32), HBM, input2, VMEM, input2_dmaLength, 128, 128, 128, 7);
                                dlc_sync(dma_flag);
                                input2_HbmOff_Pre = input2_HbmOff;
                            }

                            // 在计算过程中，给对应的 output_MathSzie 要拆分的维度的大小计算并赋值
                            output_MathSzie[batch_dim] = remain_size;

                            for (int dim1 = 0; dim1 < output_MathSzie[0]; dim1 += 2048){
                                len = min(output_MathSzie[0] - dim1, 2048);
                                mask = pre_exp2(len/256);
                                for (int dim5 = 0; dim5 < output_MathSzie[4]; dim5++) {
                                    for (int dim4 = 0; dim4 < output_MathSzie[3]; dim4++) {
                                        for (int dim3 = 0; dim3 < output_MathSzie[2]; dim3++) {
                                            for (int dim2 = 0; dim2 < output_MathSzie[1]; dim2++) {
                                                
                                                input0_VmemOff = dim1 * stride_input0[0] + dim2 * stride_input0[1] + dim3 * stride_input0[2]
                                                                + dim4 * stride_input0[3] + dim5 * stride_input0[4];
                                                input1_VmemOff = dim1 * stride_input1[0] + dim2 * stride_input1[1] + dim3 * stride_input1[2]
                                                                + dim4 * stride_input1[3] + dim5 * stride_input1[4];
                                                input2_VmemOff = dim1 * stride_input2[0] + dim2 * stride_input2[1] + dim3 * stride_input2[2]
                                                                + dim4 * stride_input2[3] + dim5 * stride_input2[4];
                                                output_VmemOff = dim1 * stride_output[0] + dim2 * stride_output[1] + dim3 * stride_output[2]
                                                                + dim4 * stride_output[3] + dim5 * stride_output[4];
                                                
                                                // 如果第一维为1，先将数据放到vector中，然后填满整个vector，
                                                if(input0_dim1_unpad == 1){
                                                    temp_reg = load8_128_stride_with_ldmask_i(input0_VmemOff / 32, 1, 255, input0);
                                                    input_val_high = temp_reg[128];
                                                    input_val_low = temp_reg[0];
                                                }else{
                                                    input_val_high = load8_128_stride_with_ldmask_i((input0_VmemOff + 128) / 32, 2, mask, input0);
                                                    input_val_low = load8_128_stride_with_ldmask_i(input0_VmemOff / 32, 2, mask, input0);
                                                }
                                                if(input1_dim1_unpad ==1){
                                                    temp_reg = load8_128_stride_with_ldmask_i(input1_VmemOff / 32, 1, 255, input1);
                                                    min_val_high = temp_reg[128];
                                                    min_val_low = temp_reg[0];
                                                }else{
                                                    min_val_high = load8_128_stride_with_ldmask_i((input1_VmemOff + 128) / 32, 2, mask, input1);
                                                    min_val_low = load8_128_stride_with_ldmask_i(input1_VmemOff / 32, 2, mask, input1);
                                                }
                                                if(input2_dim1_unpad ==1){
                                                    temp_reg = load8_128_stride_with_ldmask_i(input2_VmemOff / 32, 1, 255, input2);
                                                    max_val_high = temp_reg[128];
                                                    max_val_low = temp_reg[0];
                                                }else{
                                                    max_val_high = load8_128_stride_with_ldmask_i((input2_VmemOff + 128) / 32, 2, mask, input2);
                                                    max_val_low = load8_128_stride_with_ldmask_i(input2_VmemOff / 32, 2, mask, input2);
                                                }

                                                result_high = input_val_high;
                                                result_low = input_val_low;

                                                min_bool_high = v_s32_cmp(LS, result_high, min_val_high);
                                                min_bool_temp = v_s32_cmp(EQ, result_high, min_val_high);
                                                result_high = v_s32_sel(min_bool_high, result_high, min_val_high);
                                                temp_reg = v_s32_sel(min_bool_temp,min_val_low,result_low);
                                                min_bool_temp = v_s32_cmp(LS,temp_reg,min_val_low);
                                                result_low = v_s32_sel(min_bool_high,result_low,min_val_low);
                                                result_low = v_s32_sel(min_bool_temp,result_low,min_val_low);


                                                max_bool_high = v_s32_cmp(GT, result_high, max_val_high);
                                                max_bool_temp = v_s32_cmp(EQ, result_high, max_val_high);
                                                result_high = v_s32_sel(max_bool_high, result_high, max_val_high);
                                                temp_reg = v_s32_sel(max_bool_temp,max_val_low,result_low);
                                                max_bool_temp = v_s32_cmp(GT,temp_reg,max_val_low);
                                                result_low = v_s32_sel(max_bool_temp,result_low,max_val_low);
                                                result_low = v_s32_sel(max_bool_temp,result_low,max_val_low);

                                                store8_128_stride_stmk_i((output_VmemOff + 128)/32,2, output,result_high,mask);
                                                store8_128_stride_stmk_i(output_VmemOff/32,2, output,result_low,mask);
                                            }
                                        }
                                    }
                                }
                            }
                            // __attribute__((unused)) volatile float wait = vstore_wait(v_f32_ld_tnsr_st_msk(0, output, 1, 1));
                            output_HbmOff = dma_dim1 * output_DmaStride[0] + dma_dim2 * output_DmaStride[1] + dma_dim3 * output_DmaStride[2] + 
                                        dma_dim4 * output_DmaStride[3] + dma_dim5 * output_DmaStride[4];
                            output_dmaLength = remain_size * output_DmaStride[0];
                            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (output_HbmOff + output_XysOff) / 32), HBM, output_dmaLength, 128, 128, 128, 7);
                            dlc_sync(dma_flag);
                        }
                    }
                }
            }
        }
    }
}




inline void Min_no_broadcast_Long(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;
    /* long */
    vmemSize = vmemSize / 256 * 256;
    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 2048){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN512(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    int8_128 input_val_high, min_val_high, result_high;
    int8_128 input_val_low, min_val_low, result_low;
    bool8_128 min_bool_high,min_bool_temp;
    int8_128 temp_reg;

    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 2048 * 2048;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;
            for(; offset_vmem < process_length1024; offset_vmem += 2048){

                input_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,255, input0);
                min_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,255, input1);

                input_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32,2,255, input0);
                min_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32, 2,255, input1);

                result_high = input_val_high;
                result_low = input_val_low;

                min_bool_high = v_s32_cmp(LS, result_high, min_val_high);
                min_bool_temp = v_s32_cmp(EQ, result_high, min_val_high);
                result_high = v_s32_sel(min_bool_high, result_high, min_val_high);
                temp_reg = v_s32_sel(min_bool_temp,min_val_low,result_low);
                min_bool_temp = v_s32_cmp(LS,temp_reg,min_val_low);
                result_low = v_s32_sel(min_bool_high,result_low,min_val_low);
                result_low = v_s32_sel(min_bool_temp,result_low,min_val_low);

                store8_128_stride_stmk_i((offset_vmem + 128) / 32,2, output,result_high,255);
                store8_128_stride_stmk_i(offset_vmem / 32,2 ,output,result_low,255);

            }
            int mask = pre_exp2(process_length_remain/256);
            if(mask > 0){
                input_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,mask, input0);
                min_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,mask, input1);

                input_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32,2,mask, input0);
                min_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32, 2,mask, input1);

                result_high = input_val_high;
                result_low = input_val_low;

                min_bool_high = v_s32_cmp(LS, result_high, min_val_high);
                min_bool_temp = v_s32_cmp(EQ, result_high, min_val_high);
                result_high = v_s32_sel(min_bool_high, result_high, min_val_high);
                temp_reg = v_s32_sel(min_bool_temp,min_val_low,result_low);
                min_bool_temp = v_s32_cmp(LS,temp_reg,min_val_low);
                result_low = v_s32_sel(min_bool_high,result_low,min_val_low);
                result_low = v_s32_sel(min_bool_temp,result_low,min_val_low);

                store8_128_stride_stmk_i((offset_vmem + 128) / 32,2, output,result_high,mask);
                store8_128_stride_stmk_i(offset_vmem / 32,2 ,output,result_low,mask);
            }
            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}


inline void Max_no_broadcast_Long(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;
    /* long */
    vmemSize = vmemSize / 256 * 256;
    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor output = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 2048){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN512(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    int8_128 input_val_high, max_val_high, result_high;
    int8_128 input_val_low, max_val_low, result_low;
    bool8_128 max_bool_high,max_bool_temp;
    int8_128 temp_reg;
    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 2048 * 2048;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;
            for(; offset_vmem < process_length1024; offset_vmem += 2048){
                input_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,255, input0);
                max_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,255, input1);

                input_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32,2,255, input0);
                max_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32, 2,255, input1);

                result_high = input_val_high;
                result_low = input_val_low;

                max_bool_high = v_s32_cmp(GT, result_high, max_val_high);
                max_bool_temp = v_s32_cmp(EQ, result_high, max_val_high);
                result_high = v_s32_sel(max_bool_high, result_high, max_val_high);
                temp_reg = v_s32_sel(max_bool_temp,max_val_low,result_low);
                max_bool_temp = v_s32_cmp(GT,temp_reg,max_val_low);
                result_low = v_s32_sel(max_bool_high,result_low,max_val_low);
                result_low = v_s32_sel(max_bool_temp,result_low,max_val_low);

                store8_128_stride_stmk_i((offset_vmem + 128) / 32,2, output,result_high,255);
                store8_128_stride_stmk_i(offset_vmem / 32,2 ,output,result_low,255);
            }
            int mask = pre_exp2(process_length_remain/256);
            if(mask > 0){
                input_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,mask, input0);
                max_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,mask, input1);

                input_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32,2,mask, input0);
                max_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32, 2,mask, input1);

                result_high = input_val_high;
                result_low = input_val_low;

                max_bool_high = v_s32_cmp(GT, result_high, max_val_high);
                max_bool_temp = v_s32_cmp(EQ, result_high, max_val_high);
                result_high = v_s32_sel(max_bool_high, result_high, max_val_high);
                temp_reg = v_s32_sel(max_bool_temp,max_val_low,result_low);
                max_bool_temp = v_s32_cmp(GT,temp_reg,max_val_low);
                result_low = v_s32_sel(max_bool_high,result_low,max_val_low);
                result_low = v_s32_sel(max_bool_temp,result_low,max_val_low);

                store8_128_stride_stmk_i((offset_vmem + 128) / 32,2, output,result_high,mask);
                store8_128_stride_stmk_i(offset_vmem / 32,2 ,output,result_low,mask);
            }
            dma_flag = dlc_dma(output, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}

inline void Min_Max_no_broadcast_Long(SIM_X86::DLCMem* mem_info, SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm,SIM_X86::tensor input2_hbm ,SIM_X86::tensor output_hbm,
                unsigned* output_shape, int device_id){

    // vmem
    unsigned available_vmemSize = 3072 * 1024;
    unsigned vmem_num = 3;
    unsigned vmemSize = available_vmemSize / vmem_num;
    /* long */
    vmemSize = vmemSize / 256 * 256;
    SIM_X86::tensor input0 = (SIM_X86::tensor)mem_info->vmem_addr;
    SIM_X86::tensor input1 = (SIM_X86::tensor)(mem_info->vmem_addr + vmemSize / 32);
    SIM_X86::tensor input2 = (SIM_X86::tensor)(mem_info->vmem_addr + 2 * vmemSize / 32);
    
    int dma_flag;
    int length = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4];

    /* two xys */    
    int xys_length, xys_offset, use_xys;
    if(length < 2048){
        if(device_id == 0){
            xys_length = length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            use_xys = 0;
        }
    }else{
        int xys0_length = ALIGN512(length) / 2;
        if(device_id == 0){
            xys_length = xys0_length;
            xys_offset = 0;
            use_xys = 1;
        }else{
            xys_length = length - xys0_length;
            xys_offset = xys0_length;
            use_xys = 1;
        }
    }
    
    int8_128 input_val_high, min_val_high,max_val_high, result_high;
    int8_128 input_val_low, min_val_low,max_val_low, result_low;
    bool8_128 max_bool_high, min_bool_high, min_bool_temp,max_bool_temp;
    int8_128 temp_reg;


    if(use_xys){
        for(int offset = 0; offset <xys_length; offset += vmemSize){
            int process_length = min(vmemSize, xys_length - offset);
            dma_flag = dlc_dma(tensor_slice(input0_hbm, (xys_offset + offset) / 32), HBM, input0, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input1_hbm, (xys_offset + offset) / 32), HBM, input1, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            dma_flag = dlc_dma(tensor_slice(input2_hbm, (xys_offset + offset) / 32), HBM, input2, VMEM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
            int process_length1024 = process_length / 2048 * 2048;
            int process_length_remain = process_length - process_length1024;
            int offset_vmem = 0;

            for(; offset_vmem < process_length1024; offset_vmem += 2048){

                input_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32,2,255, input0);
                min_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,255, input1);
                max_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,255, input2);

                input_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32,2,255, input0);
                min_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32, 2,255, input1);
                max_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32, 2,255, input2);


                result_high = input_val_high;
                result_low = input_val_low;

                min_bool_high = v_s32_cmp(LS, result_high, min_val_high);
                min_bool_temp = v_s32_cmp(EQ, result_high, min_val_high);
                result_high = v_s32_sel(min_bool_high, result_high, min_val_high);
                temp_reg = v_s32_sel(min_bool_temp,min_val_low,result_low);
                min_bool_temp = v_s32_cmp(LS,temp_reg,min_val_low);
                result_low = v_s32_sel(min_bool_high,result_low,min_val_low);
                result_low = v_s32_sel(min_bool_temp,result_low,min_val_low);


                max_bool_high = v_s32_cmp(GT, result_high, max_val_high);
                max_bool_temp = v_s32_cmp(EQ, result_high, max_val_high);
                result_high = v_s32_sel(max_bool_high, result_high, max_val_high);
                temp_reg = v_s32_sel(max_bool_temp,max_val_low,result_low);
                max_bool_temp = v_s32_cmp(GT,temp_reg,max_val_low);
                result_low = v_s32_sel(max_bool_high,result_low,max_val_low);
                result_low = v_s32_sel(max_bool_temp,result_low,max_val_low);


                store8_128_stride_stmk_i((offset_vmem + 128) / 32,2, input0,result_high,255);
                store8_128_stride_stmk_i(offset_vmem / 32,2 ,input0,result_low,255);
            }
            int mask = pre_exp2(process_length_remain/256);
            if(mask > 0){

                input_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32,2,mask, input0);
                min_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,mask, input1);
                max_val_high = load8_128_stride_with_ldmask_i((offset_vmem + 128) / 32, 2,mask, input2);

                input_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32,2,mask, input0);
                min_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32, 2,mask, input1);
                max_val_low = load8_128_stride_with_ldmask_i(offset_vmem / 32, 2,mask, input2);


                result_high = input_val_high;
                result_low = input_val_low;

                min_bool_high = v_s32_cmp(LS, result_high, min_val_high);
                min_bool_temp = v_s32_cmp(EQ, result_high, min_val_high);
                result_high = v_s32_sel(min_bool_high, result_high, min_val_high);
                temp_reg = v_s32_sel(min_bool_temp,min_val_low,result_low);
                min_bool_temp = v_s32_cmp(LS,temp_reg,min_val_low);
                result_low = v_s32_sel(min_bool_high,result_low,min_val_low);
                result_low = v_s32_sel(min_bool_temp,result_low,min_val_low);


                max_bool_high = v_s32_cmp(GT, result_high, max_val_high);
                max_bool_temp = v_s32_cmp(EQ, result_high, max_val_high);
                result_high = v_s32_sel(max_bool_high, result_high, max_val_high);
                temp_reg = v_s32_sel(max_bool_temp,max_val_low,result_low);
                max_bool_temp = v_s32_cmp(GT,temp_reg,max_val_low);
                result_low = v_s32_sel(max_bool_high,result_low,max_val_low);
                result_low = v_s32_sel(max_bool_temp,result_low,max_val_low);


                store8_128_stride_stmk_i((offset_vmem + 128) / 32,2, input0,result_high,mask);
                store8_128_stride_stmk_i(offset_vmem / 32,2 ,input0,result_low,mask);
            }
            dma_flag = dlc_dma(input0, VMEM, tensor_slice(output_hbm, (xys_offset + offset) / 32), HBM, process_length, 128, 128, 128, 7);
            dlc_sync(dma_flag);
        }
    }
}



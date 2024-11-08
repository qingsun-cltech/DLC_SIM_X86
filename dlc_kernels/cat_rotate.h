#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "ldst.h"
#include "permute.h"
// #include "typehint.h"

#include "align.h"
// #include "kernel_arg_types.h"
#include "math.h"

const int D_CMEM = 3;


// 进行24次rotate_single
inline void dim0InnerLoop_rotate_push(int bool_cat_value, int ldst_stride, int ldst_stride_count,
                                      DLCTensorLite *cat_vmem) {

    int length = ldst_stride_count * 128 / 32;
    int ldst_offset = ldst_stride * 1024 * 2 / 32;
    int length2 = length + ldst_offset / 2;
    // int bool_cat_value = last_cat_stride % 128;

    float8_128 cat2_value00 = load8_128_stride_with_ldmask(length , ldst_stride, 255, cat_vmem[1].address);                                                             
    float8_128 cat2_value10 = load8_128_stride_with_ldmask(length2, ldst_stride, 255, cat_vmem[1].address);
    float8_128 cat2_value01 = load8_128_stride_with_ldmask((length + 1 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    float8_128 cat2_value11 = load8_128_stride_with_ldmask((length2 + 1 * ldst_offset),ldst_stride, 255, cat_vmem[1].address);
    float8_128 cat2_value02 = load8_128_stride_with_ldmask((length + 2 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    float8_128 cat2_value12 = load8_128_stride_with_ldmask((length2 + 2 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    float8_128 cat2_value03 = load8_128_stride_with_ldmask((length + 3 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    //第一部分push        
    float8_128 cat2_value13 = load8_128_stride_with_ldmask((length2 + 3 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value00, bool_cat_value, 0);

    cat2_value00 = load8_128_stride_with_ldmask((length + 4 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value10, bool_cat_value, 1);


    cat2_value10 = load8_128_stride_with_ldmask((length2 + 4 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value01, bool_cat_value, 0);


    cat2_value01 = load8_128_stride_with_ldmask((length + 5 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value11, bool_cat_value, 1);


    cat2_value11 = load8_128_stride_with_ldmask((length2 + 5 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value02, bool_cat_value, 0);


    cat2_value02 = load8_128_stride_with_ldmask((length + 6 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value12, bool_cat_value, 1);


    cat2_value12 = load8_128_stride_with_ldmask((length2 + 6 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value03, bool_cat_value, 0);


    cat2_value03 = load8_128_stride_with_ldmask((length + 7 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value13, bool_cat_value, 1);

    //第二部分
    cat2_value13 = load8_128_stride_with_ldmask((length2 + 7 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value00, bool_cat_value, 0);


    cat2_value00 = load8_128_stride_with_ldmask((length + 8 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value10, bool_cat_value, 1);


    cat2_value10 = load8_128_stride_with_ldmask((length2 + 8 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value01, bool_cat_value, 0);


    cat2_value01 = load8_128_stride_with_ldmask((length + 9 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value11, bool_cat_value, 1);


    cat2_value11 = load8_128_stride_with_ldmask((length2 + 9 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value02, bool_cat_value, 0);


    cat2_value02 = load8_128_stride_with_ldmask((length + 10 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value12, bool_cat_value, 1);


    cat2_value12 = load8_128_stride_with_ldmask((length2 + 10 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value03, bool_cat_value, 0);


    cat2_value03 = load8_128_stride_with_ldmask((length + 11 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);
    m_rotate_single(cat2_value13, bool_cat_value, 1);

    //第三部分
    cat2_value13 = load8_128_stride_with_ldmask((length2 + 11 * ldst_offset), ldst_stride, 255, cat_vmem[1].address);

    m_rotate_single(cat2_value00, bool_cat_value, 0);

    m_rotate_single(cat2_value10, bool_cat_value, 1);

    m_rotate_single(cat2_value01, bool_cat_value, 0);

    m_rotate_single(cat2_value11, bool_cat_value, 1);

    m_rotate_single(cat2_value02, bool_cat_value, 0);

    m_rotate_single(cat2_value12, bool_cat_value, 1);

    m_rotate_single(cat2_value03, bool_cat_value, 0);

    m_rotate_single(cat2_value13, bool_cat_value, 1);

}

// 进行24次pop的同时进行24次m_rotate_single
inline void dim0InnerLoop_rotate_push_pop(int rotate_count, int bool_cat_value, int ldst_stride,
                                          int ldst_stride_count, int cat_ldst_stride, int ldst_out_stride,
                                          int cat_count, DLCTensorLite *cat_vmem, DLCTensor *cat_hbm,
                                          DLCTensor *output_hbm) {

    // 计算对应的stride偏移
    int length = ldst_stride_count * 128 + rotate_count * 24576 * ldst_stride;
    int cat1_stride = (cat_ldst_stride - 1) * 128 + ldst_stride_count * 128 +
                      ((rotate_count - 1) * 24576 * ldst_out_stride);

    float8_128 cat2_special_value0;
    float8_128 cat2_special_value1;

    float8_128 cat_true_value;
    int8_128 core_id = get_core_id();
    core_id = v_u32_and(core_id, 127);
    // int bool_cat_value = last_cat_stride % 128;
    int ldst_offset_cat1 = 1024 * ldst_out_stride * 2;
    int ldst_offset_cat2 = 1024 * ldst_stride * 2;

    // 进行cat操作，补齐128中缺失的部分
    bool8_128 cat_special_bool = v_s32_cmp(LS, core_id, bool_cat_value);
    int bool_value = cat_hbm[cat_count].shape[0] - ldst_stride_count * 128 - (128 - bool_cat_value);

    // 当bool_value > 0 时rotate完剩余的值也需要store回去
    if (bool_value > 0) {

        float8_128 cat1_value00 = load8_128_stride_with_ldmask((cat1_stride ) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value00 = load8_128_stride_with_ldmask((length) / 32, ldst_stride, 255, cat_vmem[1].address);

        float8_128 cat1_value10 = load8_128_stride_with_ldmask((cat1_stride  + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value10 = load8_128_stride_with_ldmask((length + 1024 * ldst_stride) / 32, ldst_stride, 255, cat_vmem[1].address);

        float8_128 cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 1 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value01 = load8_128_stride_with_ldmask((length + 1 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);

        float8_128 cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value11 = load8_128_stride_with_ldmask((length + 1 * ldst_offset_cat2 + 1024 * ldst_stride) / 32, ldst_stride, 255, cat_vmem[1].address);


        float8_128 cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 2 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value02 = load8_128_stride_with_ldmask((length + 2 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);


        float8_128 cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value12 = load8_128_stride_with_ldmask((length + 2 * ldst_offset_cat2 + 1024 * ldst_stride) / 32, ldst_stride, 255, cat_vmem[1].address);


        float8_128 cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 3 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value03 = load8_128_stride_with_ldmask((length + 3 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);

        //第二部分pop + math + store
        float8_128 cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value13 = load8_128_stride_with_ldmask((length + 3 * ldst_offset_cat2 + 1024 * ldst_stride) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value0 = m_pop_trf(0);

        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value00, bool_cat_value, 0);



        cat1_value00 = load8_128_stride_with_ldmask((cat1_stride + 4 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value00 = load8_128_stride_with_ldmask((length + 4 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value10, bool_cat_value, 1);


        cat1_value10 = load8_128_stride_with_ldmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value10 = load8_128_stride_with_ldmask((length + 4 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value01, bool_cat_value, 0);


        cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 5 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value01 = load8_128_stride_with_ldmask((length + 5 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value11, bool_cat_value, 1);



        cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value11 = load8_128_stride_with_ldmask((length + 5 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value02, bool_cat_value, 0);



        cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 6 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value02 = load8_128_stride_with_ldmask((length + 6 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value12, bool_cat_value, 1);



        cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value12 = load8_128_stride_with_ldmask((length + 6 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value03, bool_cat_value, 0);



        cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 7 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value03 = load8_128_stride_with_ldmask((length + 7 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value13, bool_cat_value, 1);

        //第三部分


        cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value13 = load8_128_stride_with_ldmask((length + 7 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value00, bool_cat_value, 0);


        cat1_value00 = load8_128_stride_with_ldmask((cat1_stride + 8 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value00 = load8_128_stride_with_ldmask((length + 8 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value10, bool_cat_value, 1);



        cat1_value10 = load8_128_stride_with_ldmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value10 = load8_128_stride_with_ldmask((length + 8 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value01, bool_cat_value, 0);


        cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 9 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value01 = load8_128_stride_with_ldmask((length + 9 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value11, bool_cat_value, 1);


        cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value11 = load8_128_stride_with_ldmask((length + 9 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value02, bool_cat_value, 0);


        cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 10 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value02 = load8_128_stride_with_ldmask((length + 10 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value12, bool_cat_value, 1);

        

        cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value12 = load8_128_stride_with_ldmask((length + 10 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value03, bool_cat_value, 0);
        

        cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 11 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value03 = load8_128_stride_with_ldmask((length + 11 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value13, bool_cat_value, 1);

        //最后一部分
        cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value13 = load8_128_stride_with_ldmask((length + 11 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value00, bool_cat_value, 0);


        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value10, bool_cat_value, 1);


        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value01, bool_cat_value, 0);


        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value11, bool_cat_value, 1);

        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value02, bool_cat_value, 0);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value12, bool_cat_value, 1);


        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        m_rotate_single(cat2_value03, bool_cat_value, 0);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
        m_rotate_single(cat2_value13, bool_cat_value, 1);


    } else {

        float8_128 cat1_value00 = load8_128_stride_with_ldmask((cat1_stride ) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value00 = load8_128_stride_with_ldmask((length) / 32, ldst_stride, 255, cat_vmem[1].address);

        float8_128 cat1_value10 = load8_128_stride_with_ldmask((cat1_stride  + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value10 = load8_128_stride_with_ldmask((length + 1024 * ldst_stride) / 32, ldst_stride, 255, cat_vmem[1].address);

        float8_128 cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 1 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value01 = load8_128_stride_with_ldmask((length + 1 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);

        float8_128 cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value11 = load8_128_stride_with_ldmask((length + 1 * ldst_offset_cat2 + 1024 * ldst_stride) / 32, ldst_stride, 255, cat_vmem[1].address);


        float8_128 cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 2 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value02 = load8_128_stride_with_ldmask((length + 2 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);


        float8_128 cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value12 = load8_128_stride_with_ldmask((length + 2 * ldst_offset_cat2 + 1024 * ldst_stride) / 32, ldst_stride, 255, cat_vmem[1].address);


        float8_128 cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 3 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value03 = load8_128_stride_with_ldmask((length + 3 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);

        //第二部分pop + math + store
        float8_128 cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_value13 = load8_128_stride_with_ldmask((length + 3 * ldst_offset_cat2 + 1024 * ldst_stride) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value0 = m_pop_trf(0);

        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value00, bool_cat_value, 0);



        cat1_value00 = load8_128_stride_with_ldmask((cat1_stride + 4 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value00 = load8_128_stride_with_ldmask((length + 4 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value10, bool_cat_value, 1);


        cat1_value10 = load8_128_stride_with_ldmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value10 = load8_128_stride_with_ldmask((length + 4 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value01, bool_cat_value, 0);


        cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 5 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value01 = load8_128_stride_with_ldmask((length + 5 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value11, bool_cat_value, 1);



        cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value11 = load8_128_stride_with_ldmask((length + 5 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);

        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value02, bool_cat_value, 0);



        cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 6 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value02 = load8_128_stride_with_ldmask((length + 6 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value12, bool_cat_value, 1);



        cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value12 = load8_128_stride_with_ldmask((length + 6 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value03, bool_cat_value, 0);



        cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 7 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value03 = load8_128_stride_with_ldmask((length + 7 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value13, bool_cat_value, 1);

        //第三部分


        cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value13 = load8_128_stride_with_ldmask((length + 7 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value00, bool_cat_value, 0);


        cat1_value00 = load8_128_stride_with_ldmask((cat1_stride + 8 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value00 = load8_128_stride_with_ldmask((length + 8 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value10, bool_cat_value, 1);



        cat1_value10 = load8_128_stride_with_ldmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value10 = load8_128_stride_with_ldmask((length + 8 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value01, bool_cat_value, 0);


        cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 9 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value01 = load8_128_stride_with_ldmask((length + 9 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value11, bool_cat_value, 1);


        cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value11 = load8_128_stride_with_ldmask((length + 9 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value02, bool_cat_value, 0);


        cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 10 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value02 = load8_128_stride_with_ldmask((length + 10 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value12, bool_cat_value, 1);

        

        cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value12 = load8_128_stride_with_ldmask((length + 10 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value03, bool_cat_value, 0);
        

        cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 11 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value03 = load8_128_stride_with_ldmask((length + 11 * ldst_offset_cat2) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value13, bool_cat_value, 1);

        //最后一部分
        cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_value13 = load8_128_stride_with_ldmask((length + 11 * ldst_offset_cat2 + ldst_stride * 1024) / 32, ldst_stride, 255, cat_vmem[1].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value00, bool_cat_value, 0);


        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value10, bool_cat_value, 1);


        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value01, bool_cat_value, 0);


        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value11, bool_cat_value, 1);

        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value02, bool_cat_value, 0);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value12, bool_cat_value, 1);


        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value03, bool_cat_value, 0);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        m_rotate_single(cat2_value13, bool_cat_value, 1);


    }
}

// 进行最后的24次pop计算
inline void dim0InnerLoop_rotate_pop(int rotate_count, int bool_cat_value, int ldst_stride,
                                     int ldst_stride_count, int cat_ldst_stride, int ldst_out_stride,
                                     int cat_count, DLCTensorLite *cat_vmem, DLCTensor *cat_hbm,
                                     DLCTensor *output_hbm) {

    // 计算对应的stride
    int cat1_stride = (cat_ldst_stride - 1) * 128 + ldst_stride_count * 128 +
                      ((rotate_count - 1) * 24576 * ldst_out_stride);

    int8_128 core_id = get_core_id();
    core_id = v_u32_and(core_id, 127);

    // int bool_cat_value = last_cat_stride % 128;

    float8_128 cat2_special_value0;
    float8_128 cat2_special_value1;

    float8_128 cat_true_value;

    // 进行cat操作，补齐128中缺失的部分
    bool8_128 cat_special_bool = v_s32_cmp(LS, core_id, bool_cat_value);
    int bool_value = cat_hbm[cat_count].shape[0] - ldst_stride_count * 128 - (128 - bool_cat_value);

    int ldst_offset_cat1 = 1024 * ldst_out_stride * 2;

    // 当bool_value > 0 时rotate完剩余的值也需要store回去

    if (bool_value > 0) {

        float8_128 cat1_value00 = load8_128_stride_with_ldmask((cat1_stride ) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value10 = load8_128_stride_with_ldmask((cat1_stride  + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 1 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 2 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 3 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);

        //第二部分pop + math + store
        float8_128 cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);


        cat1_value00 = load8_128_stride_with_ldmask((cat1_stride + 4 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);


        cat1_value10 = load8_128_stride_with_ldmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);


        cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 5 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);



        cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
                                        



        cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 6 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);



        cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);



        cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 7 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);

        //第三部分


        cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);


        cat1_value00 = load8_128_stride_with_ldmask((cat1_stride + 8 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);



        cat1_value10 = load8_128_stride_with_ldmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);


        cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 9 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);


        cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);


        cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 10 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);

        

        cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);
        

        cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 11 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);

        //最后一部分
        cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);


        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);


        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);


        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);
                                        

        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);


        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value0);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024 + 128) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat2_special_value1);


    } else {

        float8_128 cat1_value00 = load8_128_stride_with_ldmask((cat1_stride ) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value10 = load8_128_stride_with_ldmask((cat1_stride  + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 1 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 2 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 3 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);

        //第二部分pop + math + store
        float8_128 cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);



        cat1_value00 = load8_128_stride_with_ldmask((cat1_stride + 4 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        float8_128 cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat1_value10 = load8_128_stride_with_ldmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 5 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 1 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);



        cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);



        cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 6 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 2 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);



        cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);



        cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 7 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 3 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);

        //第三部分


        cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat1_value00 = load8_128_stride_with_ldmask((cat1_stride + 8 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + 4 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);



        cat1_value10 = load8_128_stride_with_ldmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat1_value01 = load8_128_stride_with_ldmask((cat1_stride + 9 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 5 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat1_value11 = load8_128_stride_with_ldmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat1_value02 = load8_128_stride_with_ldmask((cat1_stride + 10 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 6 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);

        

        cat1_value12 = load8_128_stride_with_ldmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);
        

        cat1_value03 = load8_128_stride_with_ldmask((cat1_stride + 11 * ldst_offset_cat1) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 7 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);

        //最后一部分
        cat1_value13 = load8_128_stride_with_ldmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride, 255, cat_vmem[0].address);
        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value00);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value10);
        store8_128_stride_with_stmask((cat1_stride + 8 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value01);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value11);
        store8_128_stride_with_stmask((cat1_stride + 9 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);

        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value02);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value12);
        store8_128_stride_with_stmask((cat1_stride + 10 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);


        cat2_special_value0 = m_pop_trf(0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value0, cat1_value03);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);

        cat2_special_value1 = m_pop_trf(1);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value1, cat1_value13);
        store8_128_stride_with_stmask((cat1_stride + 11 * ldst_offset_cat1 + ldst_out_stride * 1024) / 32, ldst_out_stride,
                                        255, cat_vmem[0].address, cat_true_value);

    }
}

inline void dim0InnerLoop_1024(int vs, int last_cat_stride, int ldst_stride, int ldst_stride_count,
                               int cat_ldst_stride, int ldst_out_stride, int cat_count,
                               DLCTensorLite *cat_vmem, DLCTensor *cat_hbm, DLCTensor *output_hbm) {

    int length1 = vs * ldst_stride;
    int length3 = ldst_stride_count * 128;

    int cat1_stride = vs * ldst_out_stride + (cat_ldst_stride - 1) * 128 + ldst_stride_count * 128;

    float8_128 cat2_value =
        load8_128_stride_with_ldmask((length1 + length3) / 32, ldst_stride, 255, cat_vmem[1].address);
    float8_128 cat1_value;
    int8_128 core_id = get_core_id();
    core_id = v_u32_and(core_id, 127);
    int bool_cat_value = last_cat_stride % 128;

    cat1_value = load8_128_stride_with_ldmask(cat1_stride / 32, ldst_out_stride, 255, cat_vmem[0].address);
    float8_128 cat2_special_value = m_rotate(cat2_value, bool_cat_value, 0);

    // 进行cat操作，补齐128中缺失的部分
    bool8_128 cat_special_bool = v_s32_cmp(LS, core_id, bool_cat_value);
    int bool_value = cat_hbm[cat_count].shape[0] - ldst_stride_count * 128 - (128 - bool_cat_value);
    float8_128 cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value, cat1_value);

    if (bool_value > 0) {
        store8_128_stride_with_stmask(cat1_stride / 32, ldst_out_stride, 255, cat_vmem[0].address,
                                      cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 128) / 32, ldst_out_stride, 255, cat_vmem[0].address,
                                      cat2_special_value);
    } else {
        store8_128_stride_with_stmask(cat1_stride / 32, ldst_out_stride, 255, cat_vmem[0].address,
                                      cat_true_value);
        return;
    }
}

inline void dim0InnerLoop(int len, int vs, int last_cat_stride, int ldst_stride, int ldst_stride_count,
                          int cat_ldst_stride, int ldst_out_stride, int cat_count, DLCTensorLite *cat_vmem,
                          DLCTensor *cat_hbm, DLCTensor *output_hbm) {
    int ldst_vmask = pre_exp2(len / 128);
    // 第一次cat的计算
    int length1 = vs * ldst_stride;
    int length3 = ldst_stride_count * 128;

    int cat1_stride = vs * ldst_out_stride + (cat_ldst_stride - 1) * 128 + ldst_stride_count * 128;

    float8_128 cat2_value =
        load8_128_stride_with_ldmask((length1 + length3) / 32, ldst_stride, ldst_vmask, cat_vmem[1].address);
    float8_128 cat1_value;
    int8_128 core_id = get_core_id();
    core_id = v_u32_and(core_id, 127);
    int bool_cat_value = last_cat_stride % 128;

    cat1_value =
        load8_128_stride_with_ldmask(cat1_stride / 32, ldst_out_stride, ldst_vmask, cat_vmem[0].address);
    float8_128 cat2_special_value = m_rotate(cat2_value, bool_cat_value, 0);

    // 进行cat操作，补齐128中缺失的部分
    bool8_128 cat_special_bool = v_s32_cmp(LS, core_id, bool_cat_value);
    int bool_value = cat_hbm[cat_count].shape[0] - ldst_stride_count * 128 - (128 - bool_cat_value);
    float8_128 cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value, cat1_value);

    if (bool_value > 0) {
        store8_128_stride_with_stmask(cat1_stride / 32, ldst_out_stride, ldst_vmask, cat_vmem[0].address,
                                      cat_true_value);
        store8_128_stride_with_stmask((cat1_stride + 128) / 32, ldst_out_stride, ldst_vmask,
                                      cat_vmem[0].address, cat2_special_value);
    } else {
        // 进入此处表示没有下一批需要cat，直接return跳过不运行下面的判断
        store8_128_stride_with_stmask(cat1_stride / 32, ldst_out_stride, ldst_vmask, cat_vmem[0].address,
                                      cat_true_value);

        return;
    }
}

inline void dim0InnerLoop_128_1024(int vs, int last_cat_stride, int ldst_stride, int ldst_stride_count,
                                   int cat_ldst_stride, int ldst_out_stride, int cat_count,
                                   DLCTensorLite *cat_vmem, DLCTensor *cat_hbm, DLCTensor *output_hbm) {

    int length1 = vs * ldst_stride;
    int length3 = ldst_stride_count * 128;

    float8_128 cat2_value =
        load8_128_stride_with_ldmask((length1 + length3) / 32, ldst_stride, 255, cat_vmem[1].address);

    // 128的倍数不需要进行value1和value2的cat操作
    store8_128_stride_with_stmask((vs * ldst_out_stride + cat_ldst_stride * 128 + ldst_stride_count * 128) /
                                      32,
                                  ldst_out_stride, 255, cat_vmem[0].address, cat2_value);
}

inline void dim0InnerLoop_128(int len, int vs, int last_cat_stride, int ldst_stride, int ldst_stride_count,
                              int cat_ldst_stride, int ldst_out_stride, int cat_count,
                              DLCTensorLite *cat_vmem, DLCTensor *cat_hbm, DLCTensor *output_hbm) {
    int ldst_vmask = pre_exp2(len / 128);
    // 第一次cat的计算
    int length1 = vs * ldst_stride;
    int length3 = ldst_stride_count * 128;

    float8_128 cat2_value =
        load8_128_stride_with_ldmask((length1 + length3) / 32, ldst_stride, ldst_vmask, cat_vmem[1].address);

    store8_128_stride_with_stmask((vs * ldst_out_stride + cat_ldst_stride * 128 + ldst_stride_count * 128) / 32, ldst_out_stride,
                                  ldst_vmask, cat_vmem[0].address, cat2_value);
}

inline void dim0InnerLoop_max(int *cat1_stride,int last_cat_stride,int ldst_stride_count,int cat_count ,DLCTensorLite *cat_vmem, DLCTensor *cat_hbm) {


    float8_128 cat2_value = load8_128_stride_with_ldmask(ldst_stride_count * 4, 1, 1, cat_vmem[1].address);
    float8_128 cat1_value;
    int8_128 core_id = get_core_id();
    core_id = v_u32_and(core_id, 127);
    int bool_cat_value = last_cat_stride % 128;

    cat1_value = load8_128_stride_with_ldmask(*cat1_stride, 1, 1, cat_vmem[0].address);
    float8_128 cat2_special_value = m_rotate(cat2_value, bool_cat_value, 0);

    // 进行cat操作，补齐128中缺失的部分
    bool8_128 cat_special_bool = v_s32_cmp(LS, core_id, bool_cat_value);
    int bool_value = cat_hbm[cat_count].shape[0] - ldst_stride_count * 128 - (128 - bool_cat_value);
    float8_128 cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value, cat1_value);

    if (bool_value > 0) {
        store8_128_stride_with_stmask(*cat1_stride, 1, 1, cat_vmem[0].address,
                                      cat_true_value);
        store8_128_stride_with_stmask(*cat1_stride + 4, 1, 1,
                                      cat_vmem[0].address, cat2_special_value);
        *cat1_stride = *cat1_stride + 4;
    } else {
        // 进入此处表示没有下一批需要cat，直接return跳过不运行下面的判断
        store8_128_stride_with_stmask(*cat1_stride, 1, 1, cat_vmem[0].address,
                                      cat_true_value);
    }

}
inline void dim0InnerLoop_bf16_max(int *cat1_stride,int last_cat_stride,int ldst_stride_count,int cat_count ,DLCTensorLite *cat_vmem, DLCTensor *cat_hbm) {


    float8_128 cat2_value = load8_128_stride_with_ldmask(ldst_stride_count * 4, 1, 1, cat_vmem[1].address);
    
    float8_128 cat1_value,cat1_low_value,cat1_high_value;
    float8_128 cat2_high_value,cat2_low_value;
    int8_128 core_id = get_core_id();
    core_id = v_u32_and(core_id, 127);
    if(last_cat_stride > 128){

        int bool_cat_low_value = 128;
        int bool_cat_high_value = last_cat_stride - bool_cat_low_value;
        //取出cat1的高低
        cat1_value = load8_128_stride_with_ldmask(*cat1_stride, 1, 1, cat_vmem[0].address);
        cat1_high_value = bfloat16_to_float(unpack_16b(__$S(cat1_value), 1));
        cat1_low_value = bfloat16_to_float(unpack_16b(__$S(cat1_value), 0));
        //取出cat2的高低
        cat2_high_value = bfloat16_to_float(unpack_16b(__$S(cat2_value), 1));
        cat2_low_value = bfloat16_to_float(unpack_16b(__$S(cat2_value), 0));
        //将cat2的高进行rotate
        float8_128 cat2_special_value = m_rotate(cat2_low_value, bool_cat_high_value, 0);

        // 进行cat操作，补齐128中缺失的部分
        bool8_128 cat_special_bool = v_s32_cmp(LS, core_id, bool_cat_high_value);
        float8_128 cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value, cat1_high_value);
        //将cat好的cat1进行封装回bf16形式
        float8_128 cat1_res = __$F(float_to_bfloat16(cat_true_value, cat1_low_value));


        //对cat2剩下部分进行cat操作
        float8_128 cat2_high_special_value = m_rotate(cat2_high_value, bool_cat_high_value, 0);
        cat_true_value = v_f32_sel(cat_special_bool, cat2_high_special_value, cat2_special_value);

        //将cat好的cat2进行封装回bf16形式
        float8_128 cat2_res = __$F(float_to_bfloat16(cat2_high_special_value, cat_true_value));


        int bool_value = cat_hbm[cat_count].shape[0] - ldst_stride_count * 256 - (256 - last_cat_stride);

        if (bool_value > 0) {
            store8_128_stride_with_stmask(*cat1_stride, 1, 1, cat_vmem[0].address,
                                        cat1_res);
            store8_128_stride_with_stmask(*cat1_stride + 4, 1, 1,
                                        cat_vmem[0].address, cat2_res);
            *cat1_stride = *cat1_stride + 4;
        } else {
            // 进入此处表示没有下一批需要cat，直接return跳过不运行下面的判断
            store8_128_stride_with_stmask(*cat1_stride, 1, 1, cat_vmem[0].address,
                                        cat1_res);
        }


    } else {


        int bool_cat_low_value = last_cat_stride;
        //取出cat1的高低
        cat1_value = load8_128_stride_with_ldmask(*cat1_stride, 1, 1, cat_vmem[0].address);
        cat1_high_value = bfloat16_to_float(unpack_16b(__$S(cat1_value), 1));
        cat1_low_value = bfloat16_to_float(unpack_16b(__$S(cat1_value), 0));
        //取出cat2的高低
        cat2_high_value = bfloat16_to_float(unpack_16b(__$S(cat2_value), 1));
        cat2_low_value = bfloat16_to_float(unpack_16b(__$S(cat2_value), 0));
        //将cat2的高进行rotate
        float8_128 cat2_special_value = m_rotate(cat2_low_value, bool_cat_low_value, 0);

        // 进行cat操作，补齐128中缺失的部分
        bool8_128 cat_special_bool = v_s32_cmp(LS, core_id, bool_cat_low_value);
        float8_128 cat_true_value = v_f32_sel(cat_special_bool, cat2_special_value, cat1_low_value);
        //对cat2剩下部分进行cat操作
        float8_128 cat2_high_special_value = m_rotate(cat2_high_value, bool_cat_low_value, 0);
        float8_128 cat_true_value_high = v_f32_sel(cat_special_bool, cat2_high_special_value, cat2_special_value);

        //将cat好的cat1进行封装回bf16形式
        float8_128 cat1_res = __$F(float_to_bfloat16(cat_true_value_high, cat_true_value));


        //将cat好的cat2进行封装回bf16形式
        float8_128 cat2_res = __$F(float_to_bfloat16(0, cat2_high_special_value));


        int bool_value = cat_hbm[cat_count].shape[0] - ldst_stride_count * 256 - (256 - last_cat_stride);

        if (bool_value > 0) {
            store8_128_stride_with_stmask(*cat1_stride, 1, 1, cat_vmem[0].address,
                                        cat1_res);
            store8_128_stride_with_stmask(*cat1_stride + 4, 1, 1,
                                        cat_vmem[0].address, cat2_res);
            *cat1_stride = *cat1_stride + 4;
        } else {
            // 进入此处表示没有下一批需要cat，直接return跳过不运行下面的判断
            store8_128_stride_with_stmask(*cat1_stride, 1, 1, cat_vmem[0].address,
                                        cat1_res);
        }

    }

}
inline void dim0InnerLoop_bf16(int *cat1_stride,int last_cat_stride,int ldst_stride_count,int cat_count ,DLCTensorLite *cat_vmem, DLCTensor *cat_hbm) {


    float8_128 cat2_value = load8_128_stride_with_ldmask(ldst_stride_count * 4, 1, 1, cat_vmem[1].address);
    float8_128 cat2_value_high = bfloat16_to_float(unpack_16b(__$S(cat2_value), 1));
    float8_128 cat2_value_low = bfloat16_to_float(unpack_16b(__$S(cat2_value), 0));

    float8_128 cat1_value,cat1_value_low,cat_true_value;

    cat1_value = load8_128_stride_with_ldmask(*cat1_stride, 1, 1, cat_vmem[0].address);

    if(ldst_stride_count == 0){
        cat1_value_low = bfloat16_to_float(unpack_16b(__$S(cat1_value), 0));
    } else {
        cat1_value_low = bfloat16_to_float(unpack_16b(__$S(cat1_value), 1));
    }

    cat_true_value = __$F(float_to_bfloat16(cat2_value_low, cat1_value_low));

    store8_128_stride_with_stmask(*cat1_stride, 1, 1, cat_vmem[0].address,
                                    cat_true_value);

    *cat1_stride = *cat1_stride + 4;

    int bool_value = cat_hbm[cat_count].shape[0] % 256;

    if ((bool_value == 0) && ((ldst_stride_count + 1) * 256 == cat_hbm[cat_count].shape[0])) {

        store8_128_stride_with_stmask(*cat1_stride, 1, 1, cat_vmem[0].address,
                                        __$F(float_to_bfloat16(0, cat2_value_high)));
    }

}
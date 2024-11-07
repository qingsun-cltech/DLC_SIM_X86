#include "x86.h"

#include "dlc_kernels/ldst.h"

#define __DLC_ARCH__ 1

#include "dlc_kernels/chunk.h"
#include "dlc_kernels/mm_t.h"
#include "dlc_kernels/dma.h"
#include "dlc_kernels/pingpong.h"
inline void tile_trans_transfer_bf16_h(SIM_X86::tensor src, SIM_X86::tensor dst, int srcaddr, int src_h, int src_w, int dstaddr, int stride_src, int stride_dst) {
    int i = 0;
    for (; i + 256 <= src_h; i += 256) {
        for (int j = 0; j < src_w; j += 256) {
            int addr0 = src_h * j / 2 + i * 128;
            int addr1 = src_h * j / 2 + (i + 128) * 128;
            int st = 1;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            data1[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_packed_start(data1[0], 128, 1);
            for (int index = 1; index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 32), 255);
                m_transpose_packed_mid(data1[index], 1);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 32), 255);
            m_transpose_packed_end(data0[7], 0);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 7 * 32), 255);
            m_transpose_packed_end(data1[7], 1);


            float8_128 __attribute__((address_space(2))) res01[16];
            for (int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }

            int addr2 = src_h * j / 2 + (i + 64) * 128;
            int addr3 = src_h * j / 2 + (i + 192) * 128;
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);
            SIM_X86::tensor t3 = tensor_slice(src, srcaddr / 32 + addr3 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            float8_128 __attribute__((address_space(2))) data3[8];
            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);
            data3[0] = load8_128_stride_ldmk(0, st, t3, 255);
            m_transpose_packed_start(data3[0], 128, 1);
            for (int index = 1; index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 32), 255);
                m_transpose_packed_mid(data2[index], 0);
                data3[index] = load8_128_stride_ldmk(0, st, tensor_slice(t3, index * 32), 255);
                m_transpose_packed_mid(data3[index], 1);
            }

            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 32), 255);
            m_transpose_packed_end(data2[7], 0);
            data3[7] = load8_128_stride_ldmk(0, st, tensor_slice(t3, 7 * 32), 255);
            m_transpose_packed_end(data3[7], 1);

            float8_128 __attribute__((address_space(2))) res23[16];
            for (int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);

                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res23[index] = __$F(float_to_bfloat16(x1, x0));
            }

            int store_addr = src_w * i / 2 + j * 128;
            int store_st = 1;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for (int index = 0; index < store_num; index++) {

                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(index * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((index + 16) * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
    int rest_h = src_h - i;
    if (rest_h == 0) return;
    if (rest_h <= 64) {
        for (int j = 0; j < src_w; j += 256) {
            int addr0 = src_h * j / 2 + i * 128;
            int st = 1;
            int push_num = (rest_h + 7) / 8;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            int cur_sth = min(rest_h, 8);
            data0[0] = load8_128_stride_ldmk(0, st, t0, (1 << cur_sth) - 1);

            m_transpose_packed_start(data0[0], 128, 0);

            for (int index = 1; index < push_num - 1; index++) {
                cur_sth = min(rest_h - index * 8, 8);
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data0[index], 0);
            }

            cur_sth = min(rest_h - (push_num - 1) * 8, 8);
            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, (push_num - 1) * 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data0[7], 0);

            float8_128 __attribute__((address_space(2))) res01[16];
            for (int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                x0 = __$F(__$S(x0) << 16);
                float8_128 zero = v_u32_move_b(0);
                res01[index] = __$F(float_to_bfloat16(zero, x0));
            }

            int store_addr = src_w * i / 2 + j * 128;
            int store_st = 1;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);

            for (int index = 0; index < store_num; index++) {
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(index * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((index + 16) * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), up, (1 << cur_sth2) - 1);
            }
        }
    }
    else if (rest_h <= 128) {
        for (int j = 0; j < src_w; j += 256) {
            int addr0 = src_h * j / 2 + i * 128;
            int addr2 = src_h * j / 2 + (i + 64) * 128;
            int st = 1;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data2[8];
            int push_num = (rest_h - 64 + 7) / 8;
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            int cur_sth = min(rest_h - 64, 8);
            data2[0] = load8_128_stride_ldmk(0, st, t2, (1 << cur_sth) - 1);
            m_transpose_packed_start(data2[0], 128, 1);

            for (int index = 1; index < push_num - 1; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                cur_sth = min(rest_h - 64 - index * 8, 8);
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data2[index], 1);
            }

            for (int index = max(push_num - 1, 1); index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 32), 255);
                m_transpose_packed_mid(data0[index], 0);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 32), 255);
            m_transpose_packed_end(data0[7], 0);
            cur_sth = min(rest_h - 64 - (push_num - 1) * 8, 8);
            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, (push_num - 1) * 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data2[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            float8_128 __attribute__((address_space(2))) res23[16];
            for (int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                float8_128 zero = v_u32_move_b(0);
                res01[index] = __$F(float_to_bfloat16(zero, x0));
                res23[index] = __$F(float_to_bfloat16(zero, x1));
            }

            int store_addr = src_w * i / 2 + j * 128;
            int store_st = 1;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for (int index = 0; index < store_num; index++) {

                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(index * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((index + 16) * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
    else if (rest_h <= 192) {
        for (int j = 0; j < src_w; j += 256) {
            int addr0 = src_h * j / 2 + i * 128;
            int addr1 = src_h * j / 2 + (i + 128) * 128;
            int st = 1;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            int cur_sth = min(rest_h - 128, 8);
            data1[0] = load8_128_stride_ldmk(0, st, t1, (1 << cur_sth) - 1);
            int push_num = (rest_h - 128 + 7) / 8;

            m_transpose_packed_start(data1[0], 128, 1);

            for (int index = 1; index < push_num - 1; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                cur_sth = min(rest_h - 128 - index * 8, 8);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data1[index], 1);
            }

            for (int index = max(push_num - 1, 1); index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 32), 255);
                m_transpose_packed_mid(data0[index], 0);
            }
            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 32), 255);
            m_transpose_packed_end(data0[7], 0);
            cur_sth = min(rest_h - 128 - (push_num - 1) * 8, 8);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, (push_num - 1) * 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data1[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            for (int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }

            int addr2 = src_h * j / 2 + (i + 64) * 128;
            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);

            for (int index = 1; index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 32), 255);
                m_transpose_packed_mid(data2[index], 0);
            }

            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 32), 255);
            m_transpose_packed_end(data2[7], 0);

            float8_128 __attribute__((address_space(2))) res23[16];
            for (int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                x0 = __$F(__$S(x0) << 16);
                float8_128 zero = v_u32_move_b(0);
                res23[index] = __$F(float_to_bfloat16(zero, x0));
            }

            int store_addr = src_w * i / 2 + j * 128;
            int store_st = 1;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for (int index = 0; index < store_num; index++) {
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(index * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((index + 16) * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
    else if (rest_h < 256) {
        for (int j = 0; j < src_w; j += 256) {
            int addr0 = src_h * j / 2 + i * 128;
            int addr1 = src_h * j / 2 + (i + 128) * 128;
            int st = 1;
            SIM_X86::tensor t0 = tensor_slice(src, srcaddr / 32 + addr0 / 32);
            SIM_X86::tensor t1 = tensor_slice(src, srcaddr / 32 + addr1 / 32);

            float8_128 __attribute__((address_space(2))) data0[8];
            float8_128 __attribute__((address_space(2))) data1[8];
            data0[0] = load8_128_stride_ldmk(0, st, t0, 255);
            m_transpose_packed_start(data0[0], 128, 0);
            data1[0] = load8_128_stride_ldmk(0, st, t1, 255);
            m_transpose_packed_start(data1[0], 128, 1);

            for (int index = 1; index < 7; index++) {
                data0[index] = load8_128_stride_ldmk(0, st, tensor_slice(t0, index * 32), 255);
                m_transpose_packed_mid(data0[index], 0);
                data1[index] = load8_128_stride_ldmk(0, st, tensor_slice(t1, index * 32), 255);
                m_transpose_packed_mid(data1[index], 1);
            }

            data0[7] = load8_128_stride_ldmk(0, st, tensor_slice(t0, 7 * 32), 255);
            m_transpose_packed_end(data0[7], 0);
            data1[7] = load8_128_stride_ldmk(0, st, tensor_slice(t1, 7 * 32), 255);
            m_transpose_packed_end(data1[7], 1);

            float8_128 __attribute__((address_space(2))) res01[16];
            for (int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);
                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res01[index] = __$F(float_to_bfloat16(x1, x0));
            }
            int addr2 = src_h * j / 2 + (i + 64) * 128;
            int addr3 = src_h * j / 2 + (i + 192) * 128;

            SIM_X86::tensor t2 = tensor_slice(src, srcaddr / 32 + addr2 / 32);
            SIM_X86::tensor t3 = tensor_slice(src, srcaddr / 32 + addr3 / 32);

            float8_128 __attribute__((address_space(2))) data2[8];
            float8_128 __attribute__((address_space(2))) data3[8];
            int push_num = (rest_h - 192 + 7) / 8;

            data2[0] = load8_128_stride_ldmk(0, st, t2, 255);
            m_transpose_packed_start(data2[0], 128, 0);
            int cur_sth = min(rest_h - 192, 8);
            data3[0] = load8_128_stride_ldmk(0, st, t3, (1 << cur_sth) - 1);
            m_transpose_packed_start(data3[0], 128, 1);

            for (int index = 1; index < push_num - 1; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 32), 255);
                m_transpose_packed_mid(data2[index], 0);
                cur_sth = min(rest_h - 192 - index * 8, 8);
                data3[index] = load8_128_stride_ldmk(0, st, tensor_slice(t3, index * 32), (1 << cur_sth) - 1);
                m_transpose_packed_mid(data3[index], 1);
            }

            for (int index = max(push_num - 1, 1); index < 7; index++) {
                data2[index] = load8_128_stride_ldmk(0, st, tensor_slice(t2, index * 32), 255);
                m_transpose_packed_mid(data2[index], 0);
            }
            data2[7] = load8_128_stride_ldmk(0, st, tensor_slice(t2, 7 * 32), 255);
            m_transpose_packed_end(data2[7], 0);
            cur_sth = min(rest_h - 192 - (push_num - 1) * 8, 8);
            data3[7] = load8_128_stride_ldmk(0, st, tensor_slice(t3, (push_num - 1) * 32), (1 << cur_sth) - 1);
            m_transpose_packed_end(data3[7], 1);

            float8_128 __attribute__((address_space(2))) res23[16];
            for (int index = 0; index < 16; index++) {
                float8_128 x0 = m_pop_trf(0);
                float8_128 x1 = m_pop_trf(1);

                x0 = __$F(__$S(x0) << 16);
                x1 = __$F(__$S(x1) << 16);
                res23[index] = __$F(float_to_bfloat16(x1, x0));
            }

            int store_addr = src_w * i / 2 + j * 128;
            int store_st = 1;
            int cur_h = min(src_w - j, 256);
            int padding_cur_h = (cur_h + 7) & 0xfffffff8;
            int store_num = min(padding_cur_h / 8, 16);
            for (int index = 0; index < store_num; index++) {
                m_permute(res23[index], 0);
                float8_128 res23_odd = m_pop_trf(0);
                m_permute(res23[index], 1);
                float8_128 res23_even = m_pop_trf(1);
                m_permute(res01[index], 0);
                float8_128 res01_odd = m_pop_trf(0);
                m_permute(res01[index], 1);
                float8_128 res01_even = m_pop_trf(1);

                bool8_128 cmp = v_set_vmask(64);
                float8_128 zero = v_u32_move_b(0);
                float8_128 up = v_f32_sel(cmp, zero, res01_odd);
                float8_128 down = v_f32_sel(cmp, res23_odd, zero);
                int8_128 res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth = min(cur_h - index * 8, 8);
                store8_128_stride_stmk(index * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth) - 1);

                up = v_f32_sel(cmp, zero, res01_even);
                down = v_f32_sel(cmp, res23_even, zero);
                res_bf16 = v_s32_add(__$S(up), __$S(down));
                int cur_sth2 = min((cur_h - 128 - index * 8) < 0 ? 0 : (cur_h - 128 - index * 8), 8);
                store8_128_stride_stmk((index + 16) * 32, store_st, tensor_slice(dst, (dstaddr + store_addr) / 32), __$F(res_bf16), (1 << cur_sth2) - 1);
            }
        }
    }
}

//input、weight、out:hbm
inline void matmul_bf16_lhsT_2xys(SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor vmem,
                     SIM_X86::tensor cmem, int AH, int AW, int BW, int ah, int aw, int bw)
{
    SIM_X86::tensor input0 = vmem;
    SIM_X86::tensor input0_tmp = input0 + aw * ALIGN256(ah) / 32;
    SIM_X86::tensor input1 = input0_tmp + ah * ALIGN256(aw) / 32;
    SIM_X86::tensor output = input1 + aw * ALIGN256(bw) / 32;
    SIM_X86::tensor output_tmp = output + ah * ALIGN256(bw) / 32;
    

    int process_ah, process_aw, process_bw;

    int bf_BW, bf_AH;
    bf_AH = bf16len(ALIGN128(AH), ALIGN128(AH));
    bf_BW = bf16len(ALIGN128(BW), ALIGN128(BW));
    
    int device_id = get_device_id();
    //双xys拆分AH
    int AH_XYS = ALIGN256(AH / 2);
    if(device_id == 1){
        AH_XYS = AH - AH_XYS; 
    }

    set_permute_for_tsps_bf16();
    //dma可以优化
    for (int i = 0; i < AH; i += ah) {
        process_ah = min(AH - i, ah);
        for (int j = 0; j < BW; j += bw) {
            process_bw = min(BW - j, bw);
            int bf_j = bf16len(ALIGN128(j), ALIGN128(j));
            int bf_bw = bf16len(ALIGN128(process_bw), ALIGN128(process_bw));
            for (int k = 0; k < AW; k += aw) {
                process_aw = min(AW - k, aw);
                int bf_ah = bf16len(ALIGN128(process_ah), ALIGN128(process_ah));
                int bf_i = bf16len(ALIGN128(i), ALIGN128(i));
                load_mat_0123_h(input0_hbm, input0_tmp, 1, 1, AW, bf_AH, 0, 0, k, bf_i,
                    process_aw, bf_ah);
                int input_handle0 = dlc_dma(input0_hbm, HBM, input0, VMEM, 0, 128, 128, 128, 7);
                dlc_sync(input_handle0);
                tile_trans_transfer_bf16_h(input0_tmp, input0, 0, process_aw, process_ah, 0, ALIGN256(process_ah) / 2, ALIGN256(process_aw) / 2);
                load_mat_0123_h(input1_hbm, input1, 1, 1, AW, bf_BW, 0, 0, k, bf_j,
                    process_aw, bf_bw);
                matmul_all_bf16(input0, input1, output, output_tmp, process_ah, process_aw, process_bw, k, 1.0, 1.0);
            }
            store_mat_0123_h(output_tmp, output_hbm, 1, 1, AH, bf_BW, 0, 0, i, bf_j, process_ah, bf_bw);
        }
    }
    sync_device();
}

inline void matmul_bf16_rhsT_2xys(SIM_X86::tensor input_hbm, SIM_X86::tensor mat2_hbm, SIM_X86::tensor out_hbm, SIM_X86::tensor vmem,
    SIM_X86::tensor cmem, int AH, int AW, int BW, int ah, int aw, int bw)
{
    // //通过get_device_id()可以得到当前使用的xys的下标，一块芯片有两个xys，所以下标为0,1
    int device_id = get_device_id();

    SIM_X86::tensor input = vmem;
    SIM_X86::tensor mat2 = input + ah * ALIGN256(aw) / 32;
    SIM_X86::tensor out = mat2 + bw * ALIGN256(aw) / 32;
    SIM_X86::tensor out_tmp = out + ah * ALIGN256(bw) / 32;
    //参数
    int AW_bf16_128 = ALIGN256(AW) / 2, BW_bf16_128 = ALIGN256(BW) / 2;

    int process_ah, process_aw, process_bw;

    //双xys拆分AH
    int AH_XYS = AH / 2;
    int AH_offset = 0;
    if (device_id == 1) {
        AH_offset = AH_XYS;
        AH_XYS = AH - AH_XYS;
    }
    input_hbm = input_hbm + AH_offset * AW_bf16_128 / 32;
    out_hbm = out_hbm + AH_offset * BW_bf16_128 / 32;
    for (int i = 0; i < AH_XYS; i += ah) {
        for (int j = 0; j < BW; j += bw) {
            process_bw = min(BW - j, bw);
            int bf_bw = ALIGN256(process_bw) / 2;
            int bf_j = ALIGN256(j) / 2;
            for (int k = 0; k < AW; k += aw) {
                process_ah = min(AH_XYS - i, ah);
                process_aw = min(AW - k, aw);
                int bf_aw = ALIGN256(process_aw) / 2;
                int bf_k = ALIGN256(k) / 2;
                load_mat_0123_h(input_hbm, input, 1, 1, AH_XYS, AW_bf16_128, 0, 0, i, bf_k, process_ah, bf_aw);
                load_mat_0123_h(mat2_hbm, mat2, 1, 1, BW, AW_bf16_128, 0, 0, j, bf_k, process_bw, bf_aw);
                matmul_all_bf16_RHST(input, mat2, out, out_tmp, process_ah, process_aw, process_bw, k, 1.0, 1.0);
            }

            store_mat_0123_h(out_tmp, out_hbm, 1, 1, AH_XYS, BW_bf16_128, 0, 0, i, bf_j, process_ah, bf_bw);
        }
    }
    sync_device();
}

//input、weight、out:hbm
inline void matmul_bf16_resT_2xys(SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor vmem,
                     SIM_X86::tensor cmem, int AH, int AW, int BW, int ah, int aw, int bw)
{
    // vmemsize
    SIM_X86::tensor input0 = vmem;
    SIM_X86::tensor input1 = input0 + ah * ALIGN256(aw) / 32;
    SIM_X86::tensor output = input1 + aw * ALIGN256(bw) / 32;
    SIM_X86::tensor output_tmp = output + bw * ALIGN256(ah) / 32;
    
    int process_ah, process_aw, process_bw;

    int bf_AW, bf_BW, bf_AH;
    bf_AH = bf16len(ALIGN128(AH), ALIGN128(AH));
    bf_AW = bf16len(ALIGN128(AW), ALIGN128(AW));
    bf_BW = bf16len(ALIGN128(BW), ALIGN128(BW));
    
    int device_id = get_device_id();
    //双xys拆分AH
    int AH_XYS = ALIGN256(AH / 2);
    if(device_id == 1){
        AH_XYS = AH - AH_XYS; 
    }
    for (int i = 0; i < AH; i += ah) {
        process_ah = min(AH - i, ah);
        for (int j = 0; j < BW; j += bw) {
            process_bw = min(BW - j, bw);
            int bf_j = bf16len(ALIGN128(j), ALIGN128(j));
            int bf_i = bf16len(ALIGN128(i), ALIGN128(i));
            int bf_bw = bf16len(ALIGN128(process_bw), ALIGN128(process_bw));
            int bf_ah = bf16len(ALIGN128(process_ah), ALIGN128(process_ah));
            for (int k = 0; k < AW; k += aw) {
                process_aw = min(AW - k, aw);
                int bf_aw = bf16len(ALIGN128(process_aw), ALIGN128(process_aw));
                int bf_k = bf16len(ALIGN128(k), ALIGN128(k));
                load_mat_0123_h(input0_hbm, input0, 1, 1, AH, bf_AW, 0, 0, i, bf_k,
                    process_ah, bf_aw);
                load_mat_0123_h(input1_hbm, input1, 1, 1, AW, bf_BW, 0, 0, k, bf_j,
                    process_aw, bf_bw);
                matmul_all_bf16(input0, input1, output, output_tmp, process_ah, process_aw, process_bw, k, 1.0, 1.0);
            }
            set_permute_for_tsps_bf16();
            tile_trans_transfer_bf16_h(output_tmp, output, 0, process_ah, process_bw, 0, ALIGN256(process_bw) / 2, ALIGN256(process_ah) / 2);
            store_mat_0123_h(output, output_hbm, 1, 1, BW, bf_AH, 0, 0, j, bf_i, process_bw, bf_ah);
        }
    }
    sync_device();
}

inline void matmul_bf16_2xys(SIM_X86::tensor input0_hbm, SIM_X86::tensor input1_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor vmem,
                     int AH, int AW, int BW, int ah, int aw, int bw)
{
    SIM_X86::tensor input_cur = vmem;
    SIM_X86::tensor input_next = input_cur + ah * ALIGN256(aw) / 64;
    SIM_X86::tensor mat2_cur = input_next + ah * ALIGN256(aw) / 64;
    SIM_X86::tensor mat2_next = mat2_cur + aw * ALIGN256(bw) / 64;
    SIM_X86::tensor out = mat2_next + aw * ALIGN256(bw) / 64;
    SIM_X86::tensor output_tmp = out + ah * ALIGN256(bw) / 32;
    //元素个数
    int process_ah, process_aw, process_bw;

    int bf_AW, bf_BW;
    bf_AW = bf16len(ALIGN128(AW), ALIGN128(AW));
    bf_BW = bf16len(ALIGN128(BW), ALIGN128(BW));
    
    int device_id = get_device_id();
    //双xys拆分AH
    int AH_XYS = AH / 2;
    int AH_offset = 0;
    if(device_id == 1){
        AH_offset = AH_XYS;
        AH_XYS = AH - AH_XYS; 
    }
    input0_hbm = input0_hbm + AH_offset * bf_AW / 32;
    output_hbm = output_hbm + AH_offset * bf_BW / 32;


    int num_blocks_AH = soft_sdiv(AH_XYS + ah -1, ah);
    int num_blocks_BW = soft_sdiv(BW + bw - 1, bw);
    int num_blocks_AW = soft_sdiv(AW + aw - 1, aw);
    int prea = -1, preb = -1;
    int sync0 = DONE, sync1 = DONE, sync2 = DONE;
    int next_process_ah = 0, next_process_aw = 0, next_process_bw = 0; 


    for(int i = 0; i < num_blocks_AH; i ++){
        process_ah = min(AH_XYS - i * ah, ah);
        for(int j = 0; j < num_blocks_BW; j ++){
            process_bw = min(BW - j * bw, bw);
            int bf_j = ALIGN256(j * bw) / 2;
            int bf_bw = ALIGN256(process_bw) / 2;
            int Coffset = i * ah * bf_BW + bf_j;
            for(int k = 0; k < num_blocks_AW; k ++){
                process_aw = min(AW - k * aw, aw);
                int bf_aw = ALIGN256(process_aw) / 2;

                //计算当前块下标
                int idxa = i * num_blocks_AW + k;
                int idxb = k * num_blocks_BW + j;
                
                //计算下一次的地址
                int Aoffset = next_posa_bf16(AH_XYS, AW, i * ah, k * aw, j, num_blocks_BW, ah, aw, &next_process_ah, &next_process_aw);
                int Boffset = next_posb_bf16(AW, BW, k * aw, j * bw, aw, bw, &next_process_bw);  
                if(idxa == 0 && idxb == 0){
                    HBMtoVMEM2(input0_hbm, input_cur, process_ah, bf_aw, bf_AW);
                    HBMtoVMEM2(input1_hbm, mat2_cur, process_aw, bf_bw, bf_BW);
                    sync0 = HBMtoVMEM2_nosync(input0_hbm + Aoffset, input_next, next_process_ah, ALIGN256(next_process_aw)/2, bf_AW);
                    sync1 = HBMtoVMEM2_nosync(input1_hbm + Boffset, mat2_next, next_process_aw, ALIGN256(next_process_bw)/2, bf_BW);
                    prea = idxa, preb = idxb;
                }else{
                    HBM2VMEM_pingpong_bf16(input0_hbm, &input_cur, &input_next,
                                    input1_hbm, &mat2_cur, &mat2_next,
                                    idxa, idxb, &prea, &preb, 
                                    bf_AW, bf_BW, 
                                    Aoffset, Boffset,
                                    next_process_ah, next_process_aw, next_process_bw,
                                    &sync0, &sync1);   
                }
                matmul_all_bf16(input_cur, mat2_cur, out, output_tmp, process_ah, process_aw, process_bw, k, 1.0, 1.0);            
            }
            VMEMtoHBM2(output_tmp, output_hbm + Coffset / 32, process_ah, bf_bw, bf_BW);
        }
    }
    dlc_sync(sync0);
    dlc_sync(sync1);
    sync_device();
}

void main_x86( SIM_X86::DLCMem* mem, SIM_X86::DLCTensor* input_hbm_, SIM_X86::DLCTensor* mat2_hbm_, SIM_X86::DLCTensor* out_hbm_, SIM_X86::DLCScalar* ChunkBlock,
    SIM_X86::DLCScalar* Is_Trans){
    bool input_T = Is_Trans[0].value;
    bool mat2_T = Is_Trans[1].value;
    bool out_T = Is_Trans[2].value;
    int flag = 0;
    if(input_T) flag |= 4;
    if(mat2_T) flag |= 2;
    if(out_T) flag |= 1;
    SIM_X86::tensor lhs = *(SIM_X86::tensor*)input_hbm_->address;
    SIM_X86::tensor rhs = *(SIM_X86::tensor*)mat2_hbm_->address;
    SIM_X86::tensor out = *(SIM_X86::tensor*)out_hbm_->address;
    int AH = input_hbm_->shape[1], AW = input_hbm_->shape[0], BW = mat2_hbm_->shape[0];
    int ah = ChunkBlock[0].value, aw = ChunkBlock[1].value, bw = ChunkBlock[2].value; 

    if(flag == 0 || flag == 7){
        if(flag == 7){
            lhs = *(SIM_X86::tensor*)mat2_hbm_->address;
            rhs = *(SIM_X86::tensor*)input_hbm_->address;
            AH = mat2_hbm_->shape[1];
            AW = mat2_hbm_->shape[0];
            BW = input_hbm_->shape[0];
        }
        matmul_bf16_2xys(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, AH, AW, BW, ah, aw, bw);
    }else if(flag == 1 || flag == 6){
        if(flag == 6){
            lhs = *(SIM_X86::tensor*)mat2_hbm_->address;
            rhs = *(SIM_X86::tensor*)input_hbm_->address;
            AH = mat2_hbm_->shape[1];
            AW = mat2_hbm_->shape[0];
            BW = input_hbm_->shape[0];
        }
        matmul_bf16_resT_2xys(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, *(SIM_X86::tensor*)mem->cmem_addr, AH, AW, BW, ah, aw, bw);
    }else if(flag == 2 || flag == 3){
        if(flag == 3){
            lhs = *(SIM_X86::tensor*)mat2_hbm_->address;
            rhs = *(SIM_X86::tensor*)input_hbm_->address;
            AH = mat2_hbm_->shape[1];
            AW = mat2_hbm_->shape[0];
            BW = input_hbm_->shape[1];
        }else{
            BW = mat2_hbm_->shape[1];
        }
        matmul_bf16_rhsT_2xys(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, *(SIM_X86::tensor*)mem->cmem_addr, AH, AW, BW, ah, aw, bw);
    }else{
        if(flag == 5){
            lhs = *(SIM_X86::tensor*)mat2_hbm_->address;
            rhs = *(SIM_X86::tensor*)input_hbm_->address;
            AH = mat2_hbm_->shape[0];
            AW = mat2_hbm_->shape[1];
            BW = input_hbm_->shape[0];
        }else{
            AH = input_hbm_->shape[0];
            AW = input_hbm_->shape[1];
        }
        matmul_bf16_lhsT_2xys(lhs, rhs, out, *(SIM_X86::tensor*)mem->vmem_addr, *(SIM_X86::tensor*)mem->cmem_addr, AH, AW, BW, ah, aw, bw);
    }
    sync_device();
}

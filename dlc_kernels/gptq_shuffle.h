#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// #include "typehint.h"
#include "libdevice.h"

// #include "kernel_arg_types.h"
#include "align.h"

#define MAX_Q_GEMM_ROWS 50
#define MAX_Q_GEMM_ROWS_8BIT 24
#define MAX_ALT_GEMM_ROWS 8


inline void rearrangeDataOrder(/*__attribute__((address_space(2)))*/ int8_128 *a,
                               /*__attribute__((address_space(2)))*/ int8_128 *b,
                               /*__attribute__((address_space(2)))*/ int8_128 *c,
                               /*__attribute__((address_space(2)))*/ int8_128 *d,
                               /*__attribute__((address_space(2)))*/ int8_128 *e,
                               /*__attribute__((address_space(2)))*/ int8_128 *f,
                               /*__attribute__((address_space(2)))*/ int8_128 *g,
                               /*__attribute__((address_space(2)))*/ int8_128 *h, int length) {
    float8_128 f_v[8];
    float8_128 h_temp[8];
    float8_128 res[8];
    float8_128 temp[8];
    temp[0] = _$F(*a);
    temp[1] = _$F(*b);
    temp[2] = _$F(*c);
    temp[3] = _$F(*d);
    temp[4] = _$F(*e);
    temp[5] = _$F(*f);
    temp[6] = _$F(*g);
    temp[7] = _$F(*h);

    int8_128 core_id = get_core_id();
    int8_128 rearrange_id = core_id / 8;

    bool8_128 mask0 = v_s32_cmp(EQ, core_id % 8, 0);
    bool8_128 mask1 = v_s32_cmp(EQ, core_id % 8, 1);
    bool8_128 mask2 = v_s32_cmp(EQ, core_id % 8, 2);
    bool8_128 mask3 = v_s32_cmp(EQ, core_id % 8, 3);
    bool8_128 mask4 = v_s32_cmp(EQ, core_id % 8, 4);
    bool8_128 mask5 = v_s32_cmp(EQ, core_id % 8, 5);
    bool8_128 mask6 = v_s32_cmp(EQ, core_id % 8, 6);
    bool8_128 mask7 = v_s32_cmp(EQ, core_id % 8, 7);

    int cycle_num = ALIGN128(length) / 128;

    for (int i = 0; i < cycle_num; i++) {
        /*这里出来的是将每个寄存器的前128分成16去拆，然后合起来，128扩展到1024,
        这里之后可以优化，少做几次，比如最后一个维度小于16，只用做一次*/
        for (int j = 0; j < 8; j++) {
            float8_128 use_temp = v_f32_sel(v_s32_cmp(LS, core_id, 128), 0, temp[j]);
            // 这里如果输入的最后一维的size比较小，那么不需要把8个都做完, 后续补充判断来做完
            h_temp[0] = m_f32_perm(use_temp, rearrange_id, 0, 0);

            h_temp[1] = m_f32_perm(use_temp, core_id + 16, 0, 0);
            h_temp[1] = m_f32_perm(h_temp[1], rearrange_id, 1, 0);

            h_temp[2] = m_f32_perm(use_temp, core_id + 32, 0, 0);
            h_temp[2] = m_f32_perm(h_temp[2], rearrange_id, 1, 0);

            h_temp[3] = m_f32_perm(use_temp, core_id + 48, 0, 0);
            h_temp[3] = m_f32_perm(h_temp[3], rearrange_id, 1, 0);

            h_temp[4] = m_f32_perm(use_temp, core_id + 64, 0, 0);
            h_temp[4] = m_f32_perm(h_temp[4], rearrange_id, 1, 0);

            h_temp[5] = m_f32_perm(use_temp, core_id + 80, 0, 0);
            h_temp[5] = m_f32_perm(h_temp[5], rearrange_id, 1, 0);

            h_temp[6] = m_f32_perm(use_temp, core_id + 96, 0, 0);
            h_temp[6] = m_f32_perm(h_temp[6], rearrange_id, 1, 0);

            h_temp[7] = m_f32_perm(use_temp, core_id + 112, 0, 0);
            h_temp[7] = m_f32_perm(h_temp[7], rearrange_id, 1, 0);

            h_temp[1] = v_row_rotate(h_temp[1], 1);
            h_temp[2] = v_row_rotate(h_temp[2], 1);
            h_temp[2] = v_row_rotate(h_temp[2], 1);
            h_temp[3] = v_row_rotate(h_temp[3], 1);
            h_temp[3] = v_row_rotate(h_temp[3], 1);
            h_temp[3] = v_row_rotate(h_temp[3], 1);

            h_temp[4] = v_row_rotate(h_temp[4], 1);
            h_temp[4] = v_row_rotate(h_temp[4], 1);
            h_temp[4] = v_row_rotate(h_temp[4], 1);
            h_temp[4] = v_row_rotate(h_temp[4], 1);

            h_temp[5] = v_row_rotate(h_temp[5], 0);
            h_temp[5] = v_row_rotate(h_temp[5], 0);
            h_temp[5] = v_row_rotate(h_temp[5], 0);

            h_temp[6] = v_row_rotate(h_temp[6], 0);
            h_temp[6] = v_row_rotate(h_temp[6], 0);

            h_temp[7] = v_row_rotate(h_temp[7], 0);

            f_v[j] = _$F(((_$S(h_temp[0]) | _$S(h_temp[1])) | _$S(h_temp[2])) | _$S(h_temp[3]) |
                         _$S(h_temp[4]) | _$S(h_temp[5]) | _$S(h_temp[6]) | _$S(h_temp[7]));
        }
        if (i + 1 < cycle_num) {
            temp[0] = v_row_rotate(temp[0], 0);
            temp[1] = v_row_rotate(temp[1], 0);
            temp[2] = v_row_rotate(temp[2], 0);
            temp[3] = v_row_rotate(temp[3], 0);
            temp[4] = v_row_rotate(temp[4], 0);
            temp[5] = v_row_rotate(temp[5], 0);
            temp[6] = v_row_rotate(temp[6], 0);
            temp[7] = v_row_rotate(temp[7], 0);
        }

        f_v[0] = v_f32_sel(mask0, 0, f_v[0]);
        f_v[1] = v_f32_sel(mask1, 0, f_v[1]);
        f_v[2] = v_f32_sel(mask2, 0, f_v[2]);
        f_v[3] = v_f32_sel(mask3, 0, f_v[3]);
        f_v[4] = v_f32_sel(mask4, 0, f_v[4]);
        f_v[5] = v_f32_sel(mask5, 0, f_v[5]);
        f_v[6] = v_f32_sel(mask6, 0, f_v[6]);
        f_v[7] = v_f32_sel(mask7, 0, f_v[7]);

        res[i] = _$F(_$S(f_v[0]) | _$S(f_v[1]) | _$S(f_v[2]) | _$S(f_v[3]) | _$S(f_v[4]) | _$S(f_v[5]) |
                     _$S(f_v[6]) | _$S(f_v[7]));

    }
    *a = _$S(res[0]);
    *b = _$S(res[1]);
    *c = _$S(res[2]);
    *d = _$S(res[3]);
    *e = _$S(res[4]);
    *f = _$S(res[5]);
    *g = _$S(res[6]);
    *h = _$S(res[7]);
}


inline float8_128 dlc_half2float(int8_128 packed_half) {
  int8_128 vr10 = packed_half & 0xFFFF;
  int8_128 vr0 = vr10 & 1023;
  int8_128 vr1 = vr10 & 31744;
  bool8_128 mask0 = v_s32_cmp(EQ, vr0, v_u32_move_i(0));
  bool8_128 mask1 = v_s32_cmp(EQ, vr1, v_u32_move_i(0));
  bool8_128 mask2 = v_s32_cmp(EQ, vr1, v_u32_move_i(31744));
  int8_128 vr2 = vr10 & 32768;
  int8_128 vr3 = v_u32_move_i(0);
  int8_128 vr4 = (32704 << 16);
  int8_128 vr5 = (32640 << 16);
  int8_128 vr6 = (vr10 & 32768) << 16;
  int8_128 vr7 = (vr10 & 32767) << 13;
  vr7 += (14336 << 16);
  vr6 = vr6 | vr7;
  vr7 = v_u32_clz(vr0) + 1;
  int8_128 vr8 = vr7 - 22;
  vr7 = v_u32_shl(vr0, vr7);

  vr8 = 113 - vr8;
  bool8_128 mask7 = v_s32_cmp(EQ, vr10, v_u32_move_i(32768));
  vr7 = v_u32_shr(vr7, 9);
  vr8 = vr8 << 23;

  vr0 = vr7 | vr8;
  vr7 = v_s32_sel(mask0, vr4, vr3);
  vr8 = v_s32_sel(mask2, vr3, vr4);
  vr7 = vr7 & vr8;
  vr8 = v_s32_sel(mask0, vr3, vr5);
  int8_128 vr9 = v_s32_sel(mask2, vr3, vr5);
  vr8 = vr8 & vr9;
  bool8_128 mask5 = v_s32_cmp(EQ, vr8, vr5);
  vr1 = v_s32_sel(mask5, vr3, vr2);
  vr1 = vr1 << 16;
  vr8 = vr8 | vr1;
  vr9 = v_s32_sel(mask1, vr6, vr3);
  vr10 = v_s32_sel(mask2, vr6, vr3);

  vr9 = vr9 & vr10;
  bool8_128 mask6 = v_s32_cmp(EQ, vr9, vr6);
  vr10 = v_s32_sel(mask0, vr0, vr3);

  int8_128 vr11 = v_s32_sel(mask1, vr3, vr0);
  vr10 = vr10 & vr11;
  mask6 = v_s32_cmp(EQ, vr10, vr0);
  vr1 = v_s32_sel(mask6, vr3, vr2);
  vr1 = vr1 << 16;
  vr10 = vr10 | vr1;
  vr11 = vr7 | vr8 | vr9;
  vr10 = vr11 | vr10;
  vr10 = v_s32_sel(mask7, vr10, v_u32_move_i(32768 << 16));
  return *(float8_128*)&vr10;
}



inline int8_128 shuffle_4bit_8(int8_128 q) {
  int8_128 qa = q;
  int8_128 qb = 0;

  for (int i = 0; i < 4; i++) {
    int8_128 qa0 = qa & 0x0f;
    int8_128 qa1 = v_u32_shr((qa & 0xf0),v_u32_move_i(4));
    qa >>= 8;
    qb |= (v_u32_shl(qa1,v_u32_move_i(i * 4 + 16)));
    qb |= (v_u32_shl(qa0,v_u32_move_i(i * 4)));
  }
  return qb;
}

inline int8_128 shuffle_2bit_16(int8_128 q) {
  
  int8_128 qa = q;
  int8_128 qb = 0;

  for (int i = 0; i < 8; i++) {
    int8_128 qa0 = qa & 0x03;
    int8_128 qa1 = v_u32_shr((qa & 0x0c),v_u32_move_i(2));
    qa >>= 4;
    qb |= (v_u32_shl(qa1 , v_u32_move_i(i * 2 + 16)));
    qb |= (v_u32_shl(qa0 , v_u32_move_i(i * 2)));
  }
  return qb;
}


inline void shuffle_3bit_32(int8_128* qa,int8_128* qb, int8_128* qc) {

  // qa: aa999888 77766655  54443332 22111000
  // qb: lkkkjjji iihhhggg  fffeeedd dcccbbba
  // qc: vvvuuutt tsssrrrq  qqpppooo nnnmmmll

//   int8_128 qd = *qc >> 26;
//   *qc <<= 4;
//   *qc |= *qb >> 28;
//   *qb <<= 2;
//   *qb |= *qa >> 30;

  int8_128 qd = v_u32_shr(*qc,v_u32_move_i(26));
  *qc = v_u32_shl(*qc,v_u32_move_i(4));
  *qc |= v_u32_shr(*qb,v_u32_move_i(28));
  *qb = v_u32_shl(*qb,v_u32_move_i(2)); 
  *qb |= v_u32_shr(*qa,v_u32_move_i(30));

  // qa: ..999888 77766655  54443332 22111000
  // qb: ..jjjiii hhhgggff  feeedddc ccbbbaaa
  // qc: ..tttsss rrrqqqpp  pooonnnm mmlllkkk
  // qd:                               vvvuuu

  int8_128 za = 0;
  int8_128 zb = 0;
  int8_128 zc = 0;

  for (int i = 0; i < 5; i++) {
    int8_128 t0 = *qa & 0x07;
    int8_128 t1 = v_u32_shr((*qa & 0x38),v_u32_move_i(3));

    *qa = v_u32_shr(*qa,v_u32_move_i(6));

    za |= v_u32_shl(t0,v_u32_move_i((i * 3)));
    za |= v_u32_shl(t1,v_u32_move_i((i * 3 + 16)));
  }
  for (int i = 0; i < 5; i++) {
    int8_128 t0 = *qb & 0x07;
    int8_128 t1 = v_u32_shr((*qb & 0x38),v_u32_move_i(3));

    *qb = v_u32_shr(*qb,v_u32_move_i(6));
    zb |= v_u32_shl(t0,v_u32_move_i((i * 3)));
    zb |= v_u32_shl(t1,v_u32_move_i((i * 3 + 16)));
  }
  for (int i = 0; i < 5; i++) {
    int8_128 t0 = *qc & 0x07;
    int8_128 t1 = v_u32_shr((*qc & 0x38),v_u32_move_i(3));

    *qc = v_u32_shr(*qc,v_u32_move_i(6));
    zc |= v_u32_shl(t0,v_u32_move_i((i * 3)));
    zc |= v_u32_shl(t1,v_u32_move_i((i * 3 + 16)));
  }


  // za:  9997775 55333111   8886664 44222000
  // zb:  jjjhhhf ffdddbbb   iiiggge eecccaaa
  // zc:  tttrrrp ppnnnlll   sssqqqo oommmkkk
  // qd:                               vvvuuu

//   za |= ((qd & 0x01) >> 0) << 15;
//   zb |= ((qd & 0x02) >> 1) << 15;
//   zc |= ((qd & 0x04) >> 2) << 15;
//   za |= ((qd & 0x08) >> 3) << 31;
//   zb |= ((qd & 0x10) >> 4) << 31;
//   zc |= ((qd & 0x20) >> 5) << 31;

  za |= v_u32_shl((v_u32_shr((qd & 0x01),v_u32_move_i(0))),v_u32_move_i(15));
  zb |= v_u32_shl((v_u32_shr((qd & 0x02),v_u32_move_i(1))),v_u32_move_i(15));
  zc |= v_u32_shl((v_u32_shr((qd & 0x04),v_u32_move_i(2))),v_u32_move_i(15));
  za |= v_u32_shl((v_u32_shr((qd & 0x08),v_u32_move_i(3))),v_u32_move_i(31));
  zb |= v_u32_shl((v_u32_shr((qd & 0x10),v_u32_move_i(4))),v_u32_move_i(31));
  zc |= v_u32_shl((v_u32_shr((qd & 0x20),v_u32_move_i(5))),v_u32_move_i(31));


  // za: v9997775 55333111  u8886664 44222000  (u, v lsb)
  // zb: vjjjhhhf ffdddbbb  uiiiggge eecccaaa
  // zc: vtttrrrp ppnnnlll  usssqqqo oommmkkk

  *qa = za;
  *qb = zb;
  *qc = zc;
}

inline int8_128 shuffle_8bit_4(int8_128 q) {
      return q;
}


inline void shuffle_4bit_kernel(SIM_X86::tensor b_q_weight,int len) {

    int vs = 0;
    for(;vs + 1024 <= len; vs += 1024){
        int8_128 weight = v_i32_ld_tnsr(vs / 32, b_q_weight, 1, 255);
        int8_128 weight_res = shuffle_4bit_8(weight);
        v_st_generic(vs / 32,b_q_weight,1,255,weight_res);
    }
    if(vs < len){
        int ldst_len = len - vs;
        int ldst_vmask = pre_exp2(ldst_len / 128);
        int8_128 weight = v_i32_ld_tnsr(vs / 32, b_q_weight, 1, ldst_vmask);
        int8_128 weight_res = shuffle_4bit_8(weight);
        v_st_generic(vs / 32,b_q_weight,1,ldst_vmask,weight_res);
    }
}

inline void shuffle_2bit_kernel(SIM_X86::tensor b_q_weight,int len) {

    int vs = 0;
    for(;vs + 1024 <= len; vs += 1024){
        int8_128 weight = v_i32_ld_tnsr(vs / 32, b_q_weight, 1, 255);
        int8_128 weight_res = shuffle_2bit_16(weight);
        v_st_generic(vs / 32,b_q_weight,1,255,weight_res);
    }
    if(vs < len){
        int ldst_len = len - vs;
        int ldst_vmask = pre_exp2(ldst_len / 128);
        int8_128 weight = v_i32_ld_tnsr(vs / 32, b_q_weight, 1, ldst_vmask);
        int8_128 weight_res = shuffle_2bit_16(weight);
        v_st_generic(vs / 32,b_q_weight,1,ldst_vmask,weight_res);
    }
}



inline void shuffle_3bit_kernel(SIM_X86::tensor b_q_weight,int h, int w) {

    for(int math_h = 0 ; math_h < h; math_h += 3){
        int vs = 0;
        for(;vs + 1024 <= w; vs += 1024){
            int8_128 weight_a = v_i32_ld_tnsr((vs + math_h * w)/ 32, b_q_weight, 1, 255);
            int8_128 weight_b = v_i32_ld_tnsr((vs + (math_h + 1) * w)/ 32, b_q_weight, 1, 255);
            int8_128 weight_c = v_i32_ld_tnsr((vs + (math_h + 2) * w)/ 32, b_q_weight, 1, 255);

            shuffle_3bit_32(&weight_a,&weight_b,&weight_c);

            v_st_generic((vs + math_h * w)/ 32,b_q_weight,1,255,weight_a);
            v_st_generic((vs + (math_h + 1) * w)/ 32,b_q_weight,1,255,weight_b);
            v_st_generic((vs + (math_h + 2) * w)/ 32,b_q_weight,1,255,weight_c);

        }
        if(vs < w){
            int ldst_len = w - vs;
            int ldst_vmask = pre_exp2(ldst_len / 128);
            int8_128 weight_a = v_i32_ld_tnsr((vs + math_h * w)/ 32, b_q_weight, 1, ldst_vmask);
            int8_128 weight_b = v_i32_ld_tnsr((vs + (math_h + 1) * w)/ 32, b_q_weight, 1, ldst_vmask);
            int8_128 weight_c = v_i32_ld_tnsr((vs + (math_h + 2) * w)/ 32, b_q_weight, 1, ldst_vmask);

            shuffle_3bit_32(&weight_a,&weight_b,&weight_c);

            v_st_generic((vs + math_h * w)/ 32,b_q_weight,1,ldst_vmask,weight_a);
            v_st_generic((vs + (math_h + 1) * w)/ 32,b_q_weight,1,ldst_vmask,weight_b);
            v_st_generic((vs + (math_h + 2) * w)/ 32,b_q_weight,1,ldst_vmask,weight_c);
        }  
    }
}


inline void shuffle_8bit_kernel(SIM_X86::tensor b_q_weight,int len) {

    int vs = 0;
    for(;vs + 1024 <= len; vs += 1024){
        int8_128 weight = v_i32_ld_tnsr(vs / 32, b_q_weight, 1, 255);
        int8_128 weight_res = shuffle_8bit_4(weight);
        v_st_generic(vs / 32,b_q_weight,1,255,weight_res);
    }
    if(vs < len){
        int ldst_len = len - vs;
        int ldst_vmask = pre_exp2(ldst_len / 128);
        int8_128 weight = v_i32_ld_tnsr(vs / 32, b_q_weight, 1, ldst_vmask);
        int8_128 weight_res = shuffle_8bit_4(weight);
        v_st_generic(vs / 32,b_q_weight,1,ldst_vmask,weight_res);
    }
}


inline void make_sequential_4bit_kernel(SIM_X86::tensor q_weight,SIM_X86::tensor q_weight_temp,int* q_perm, int dma_offset,int h, int w) {

    int w_new2_row = 0;
    for(;w_new2_row < h ; w_new2_row++){

        int vs = 0;
        for(;vs + 1024 <= w; vs += 1024){

            int q_perm_idx = (w_new2_row + dma_offset)<< 3; 
            int8_128 dst = 0;
            for (int i = 0; i < 8; i++) {

                int source_row = q_perm[q_perm_idx++];
                int w2_row = source_row >> 3; // 0
                int w2_subrow = source_row & 0x07; // 0
                int w2_row_shift = w2_subrow << 2; // 0
                int wnew2_row_shift = i << 2; // 0 - 32

                int8_128 src = v_i32_ld_tnsr((vs + w2_row * w)/ 32, q_weight, 1, 255);
                
                // src >>= w2_row_shift;
                src = v_u32_shr(src,v_u32_move_i(w2_row_shift));
                src &= 0x0000000f;
                // src <<= wnew2_row_shift;
                src = v_u32_shl(src,v_u32_move_i(wnew2_row_shift));
                dst |= src;
            }
            v_st_generic((vs + w_new2_row * w)/ 32,q_weight_temp,1,255,dst);

        }
        if(vs < w){
            int q_perm_idx = (w_new2_row + dma_offset)<< 3; 
            int8_128 dst = 0;

            int ldst_len = w - vs;
            int ldst_vmask = pre_exp2(ldst_len / 128);


            for (int i = 0; i < 8; i++) {

                int source_row = q_perm[q_perm_idx++];
                int w2_row = source_row >> 3; 
                int w2_subrow = source_row & 0x07; 
                int w2_row_shift = w2_subrow << 2; 
                int wnew2_row_shift = i << 2; 

                int8_128 src = v_i32_ld_tnsr((vs + w2_row * w)/ 32, q_weight, 1, ldst_vmask);
                src = v_u32_shr(src,v_u32_move_i(w2_row_shift));
                src &= 0x0000000f;
                src = v_u32_shl(src,v_u32_move_i(wnew2_row_shift));

                dst |= src;
            }

            v_st_generic((vs + w_new2_row * w)/ 32,q_weight_temp,1,ldst_vmask,dst);

        }
    }

}

inline void make_sequential_2bit_kernel(SIM_X86::tensor q_weight,SIM_X86::tensor q_weight_temp ,int* q_perm, int dma_offset,int h, int w) {
   
  int w_new2_row = 0;
  for(; w_new2_row < h; w_new2_row++)
  {
    int vs = 0;
    for(;vs + 1024 <= w; vs += 1024){
        int q_perm_idx = (w_new2_row + dma_offset)<< 4; 
        int8_128 dst = 0;

        for (int i = 0; i < 16; i++) {
            int source_row = q_perm[q_perm_idx++];
            int w2_row = source_row >> 4;
            int w2_subrow = source_row & 0x0f;
            int w2_row_shift = w2_subrow << 1;
            int wnew2_row_shift = i << 1;

            int8_128 src = v_i32_ld_tnsr((vs + w2_row * w)/ 32, q_weight, 1, 255);
            src = v_u32_shr(src,v_u32_move_i(w2_row_shift));

            src &= 0x00000003;
            src = v_u32_shl(src,v_u32_move_i(wnew2_row_shift));

            dst |= src;
        }
    v_st_generic((vs + w_new2_row * w)/ 32,q_weight_temp,1,255,dst);
    }
    if(vs < w){
        int q_perm_idx = (w_new2_row + dma_offset)<< 4; 
        int8_128 dst = 0;

        int ldst_len = w - vs;
        int ldst_vmask = pre_exp2(ldst_len / 128);

        for (int i = 0; i < 16; i++) {
            int source_row = q_perm[q_perm_idx++];
            int w2_row = source_row >> 4;
            int w2_subrow = source_row & 0x0f;
            int w2_row_shift = w2_subrow << 1;
            int wnew2_row_shift = i << 1;

            int8_128 src = v_i32_ld_tnsr((vs + w2_row * w)/ 32, q_weight, 1, ldst_vmask);
            src = v_u32_shr(src,v_u32_move_i(w2_row_shift));

            src &= 0x00000003;
            src = v_u32_shl(src,v_u32_move_i(wnew2_row_shift));

            dst |= src;
        }
    v_st_generic((vs + w_new2_row * w)/ 32,q_weight_temp,1,ldst_vmask,dst);
    }

  }
}



inline void make_sequential_3bit_kernel(SIM_X86::tensor q_weight,SIM_X86::tensor q_weight_temp,int* q_perm, int dma_offset,int h, int w) {

  for(int math_h = 0; math_h < (h / 32); math_h += 1){
    int vs = 0;

    for(;vs + 1024 <= w; vs += 1024){
        int w_new_row = math_h * 3;
        int q_perm_idx = (math_h + dma_offset) << 5; // * 32( 0 ~ h)
        int8_128 dst[3] = {0,0,0};

        for (int i = 0; i < 32; i++) {
            int source_row = q_perm[q_perm_idx++];
            int z_w = (source_row / 32) * 3;
            int z_mod = source_row % 32;
            int z_bit;

            if (z_mod != 10) {
            if (z_mod != 21) {
                z_bit = z_mod;
                if (z_bit > 21) {
                z_bit *= 3;
                z_bit -= 64;
                z_w += 2;
                } else if (z_bit > 10) {
                z_bit *= 3;
                z_bit -= 32;
                z_w += 1;
                } else {
                z_bit *= 3;
                }
            } else {
                z_w += 1;
            }
            }

            int8_128 src;
            if (z_mod == 10) {
                int8_128 res1 = v_i32_ld_tnsr((vs + z_w * w)/ 32, q_weight, 1, 255);
                int8_128 res2 = v_i32_ld_tnsr((vs + (z_w + 1)* w)/ 32, q_weight, 1, 255);
                src = ( v_u32_shr(res1,v_u32_move_i(30))) | ((v_u32_shl(res2,v_u32_move_i(2))) & 0x4);

            } else if (z_mod == 21) {

                int8_128 res1 = v_i32_ld_tnsr((vs + z_w * w)/ 32, q_weight, 1, 255);
                int8_128 res2 = v_i32_ld_tnsr((vs + (z_w + 1)* w)/ 32, q_weight, 1, 255);
                src = ( v_u32_shr(res1,v_u32_move_i(31))) | ((v_u32_shl(res2,v_u32_move_i(1))) & 0x6);

            } else {
                src = v_i32_ld_tnsr((vs + z_w * w)/ 32, q_weight, 1, 255);
                src = v_u32_shr(src,v_u32_move_i(z_bit));
                src &= 0x07;
            }

            z_w = 0;
            if (i != 10) {
            if (i != 21) {
                z_bit = i;
                if (z_bit > 21) {
                z_bit *= 3;
                z_bit -= 64;
                z_w += 2;
                } else if (z_bit > 10) {
                z_bit *= 3;
                z_bit -= 32;
                z_w += 1;
                } else {
                z_bit *= 3;
                }
            } else {
                z_w += 1;
            }
            }
            if (i == 10) {
                dst[z_w] |= v_u32_shl((src & 0x03),v_u32_move_i(30));
                dst[z_w + 1] |= (v_u32_shr((src & 0x4),v_u32_move_i(2)));
            } else if (i == 21) {
                dst[z_w] |= v_u32_shl((src & 0x01) , v_u32_move_i(31));
                dst[z_w + 1] |= (v_u32_shr((src & 0x6) , v_u32_move_i(1)));
            } else {
                dst[z_w] |= v_u32_shl(src ,v_u32_move_i(z_bit));
            }
        }
        v_st_generic((vs + w_new_row * w)/ 32,q_weight_temp,1,255,dst[0]);
        v_st_generic((vs + (w_new_row + 1)* w)/ 32,q_weight_temp,1,255,dst[1]);
        v_st_generic((vs + (w_new_row + 1)* w)/ 32,q_weight_temp,1,255,dst[2]);
    }
    if(vs < w){
        int w_new_row = math_h * 3;
        int q_perm_idx = (math_h + dma_offset) << 5; // * 32( 0 ~ h)
        int8_128 dst[3] = {0,0,0};
        int ldst_len = w - vs;
        int ldst_vmask = pre_exp2(ldst_len / 128);

        for (int i = 0; i < 32; i++) {
            int source_row = q_perm[q_perm_idx++];
            int z_w = (source_row / 32) * 3;
            int z_mod = source_row % 32;
            int z_bit;

            if (z_mod != 10) {
            if (z_mod != 21) {
                z_bit = z_mod;
                if (z_bit > 21) {
                z_bit *= 3;
                z_bit -= 64;
                z_w += 2;
                } else if (z_bit > 10) {
                z_bit *= 3;
                z_bit -= 32;
                z_w += 1;
                } else {
                z_bit *= 3;
                }
            } else {
                z_w += 1;
            }
            }

            int8_128 src;
            if (z_mod == 10) {
                int8_128 res1 = v_i32_ld_tnsr((vs + z_w * w)/ 32, q_weight, 1, ldst_vmask);
                int8_128 res2 = v_i32_ld_tnsr((vs + (z_w + 1)* w)/ 32, q_weight, 1, ldst_vmask);
                src = ( v_u32_shr(res1,v_u32_move_i(30))) | ((v_u32_shl(res2,v_u32_move_i(2))) & 0x4);

            } else if (z_mod == 21) {

                int8_128 res1 = v_i32_ld_tnsr((vs + z_w * w)/ 32, q_weight, 1, ldst_vmask);
                int8_128 res2 = v_i32_ld_tnsr((vs + (z_w + 1)* w)/ 32, q_weight, 1, ldst_vmask);
                src = ( v_u32_shr(res1,v_u32_move_i(31))) | ((v_u32_shl(res2,v_u32_move_i(1))) & 0x6);

            } else {
                src = v_i32_ld_tnsr((vs + z_w * w)/ 32, q_weight, 1, ldst_vmask);
                src = v_u32_shr(src,v_u32_move_i(z_bit));
                src &= 0x07;
            }

            z_w = 0;
            if (i != 10) {
            if (i != 21) {
                z_bit = i;
                if (z_bit > 21) {
                z_bit *= 3;
                z_bit -= 64;
                z_w += 2;
                } else if (z_bit > 10) {
                z_bit *= 3;
                z_bit -= 32;
                z_w += 1;
                } else {
                z_bit *= 3;
                }
            } else {
                z_w += 1;
            }
            }
            if (i == 10) {
                dst[z_w] |= v_u32_shl((src & 0x03),v_u32_move_i(30));
                dst[z_w + 1] |= (v_u32_shr((src & 0x4),v_u32_move_i(2)));
            } else if (i == 21) {
                dst[z_w] |= v_u32_shl((src & 0x01) , v_u32_move_i(31));
                dst[z_w + 1] |= (v_u32_shr((src & 0x6) , v_u32_move_i(1)));
            } else {
                dst[z_w] |= v_u32_shl(src ,v_u32_move_i(z_bit));
            }
        }
        v_st_generic((vs + w_new_row * w)/ 32,q_weight_temp,1,ldst_vmask,dst[0]);
        v_st_generic((vs + (w_new_row + 1)* w)/ 32,q_weight_temp,1,ldst_vmask,dst[1]);
        v_st_generic((vs + (w_new_row + 1)* w)/ 32,q_weight_temp,1,ldst_vmask,dst[2]);
    }
  }
}


inline void make_sequential_8bit_kernel(SIM_X86::tensor q_weight,SIM_X86::tensor q_weight_temp ,int* q_perm, int dma_offset,int h, int w) {

    int w_new2_row = 0;
    for(;w_new2_row < h ; w_new2_row++){

        int vs = 0;
        for(;vs + 1024 <= w; vs += 1024){

            int q_perm_idx = (w_new2_row + dma_offset)<< 2; 
            int8_128 dst = 0;
            for (int i = 0; i < 4; i++) {

                int source_row = q_perm[q_perm_idx++];
                int w2_row = source_row >> 2;
                int w2_subrow = source_row & 0x03;
                int w2_row_shift = w2_subrow << 3;
                int wnew2_row_shift = i << 3;

                int8_128 src = v_i32_ld_tnsr((vs + w2_row * w)/ 32, q_weight, 1, 255);
                
                // src >>= w2_row_shift;
                src = v_u32_shr(src,v_u32_move_i(w2_row_shift));
                src &= 0x000000ff;
                // src <<= wnew2_row_shift;
                src = v_u32_shl(src,v_u32_move_i(wnew2_row_shift));
                dst |= src;
            }
            v_st_generic((vs + w_new2_row * w)/ 32,q_weight_temp,1,255,dst);

        }
        if(vs < w){
            int q_perm_idx = (w_new2_row + dma_offset)<< 2; 
            int8_128 dst = 0;

            int ldst_len = w - vs;
            int ldst_vmask = pre_exp2(ldst_len / 128);

            for (int i = 0; i < 4; i++) {

                int source_row = q_perm[q_perm_idx++];
                int w2_row = source_row >> 2;
                int w2_subrow = source_row & 0x03;
                int w2_row_shift = w2_subrow << 3;
                int wnew2_row_shift = i << 3;

                int8_128 src = v_i32_ld_tnsr((vs + w2_row * w)/ 32, q_weight, 1, ldst_vmask);
                src = v_u32_shr(src,v_u32_move_i(w2_row_shift));
                src &= 0x000000ff;
                src = v_u32_shl(src,v_u32_move_i(wnew2_row_shift));

                dst |= src;
            }

            v_st_generic((vs + w_new2_row * w)/ 32,q_weight_temp,1,ldst_vmask,dst);

        }
    }
}

inline void dequant_4bit_8_prep_zero(/*__attribute__((address_space(2)))*/ int8_128* zero, 
                                    /*__attribute__((address_space(2)))*/ float8_128* z1z16_0,
                                    /*__attribute__((address_space(2)))*/ float8_128* z1z16_1,
                                    /*__attribute__((address_space(2)))*/ float8_128* y1y16_0, 
                                    /*__attribute__((address_space(2)))*/ float8_128* y1y16_1) {
    float8_128 z1 = -1024.0;
    float8_128 z16_1 = -64.0f;
    float8_128 z16_2 = __dlc_int2float_rn(*zero);
    float8_128 z16 = v_f32_sub_b(z16_1,z16_2);

    *z1z16_0 = z1 - z16_2;
    // *z1z16_0 = __dlc_int2float_rn(z1);

    *z1z16_1 = z16;

    float8_128 y1 = 1.0f;
    float8_128 y16 =  0.0625f;

    *y1y16_0 = y1;
    *y1y16_1 =  y16;
}

inline float8_128 fp(float8_128 v) {
    int8_128 i = *(int8_128*)(&v);
    i = i & 0xffffe000;
    return *(float8_128*)(&i);
}

inline void dequant_4bit_8_gptq(/*__attribute__((address_space(2)))*/ int8_128* q_0,
                                /*__attribute__((address_space(2)))*/ float8_128* dq_0,
                                /*__attribute__((address_space(2)))*/ float8_128* dq_1,
                                /*__attribute__((address_space(2)))*/ float8_128* dq_2,
                                /*__attribute__((address_space(2)))*/ float8_128* dq_3,
                                /*__attribute__((address_space(2)))*/ float8_128* dq_4,
                                /*__attribute__((address_space(2)))*/ float8_128* dq_5,
                                /*__attribute__((address_space(2)))*/ float8_128* dq_6,
                                /*__attribute__((address_space(2)))*/ float8_128* dq_7,
                                /*__attribute__((address_space(2)))*/ float8_128* z1z16_0,
                                /*__attribute__((address_space(2)))*/ float8_128* z1z16_1,
                                /*__attribute__((address_space(2)))*/ float8_128* y1y16_0,
                                /*__attribute__((address_space(2)))*/ float8_128* y1y16_1

                                                    ) {
  int8_128 qa = *q_0;
//   // Print("qa: %h\n",qa);
//   int8_128 q0 = ((qa & 0x000f000f) | c0);  // half2( q[0]      + 1024, q[1]      + 1024 )
  int8_128 q0_i = ((qa & 0xf) << 13) | 0x44800000;
  float8_128 q0 = *(float8_128*)(&q0_i);
  int8_128 q1_i =  (((qa >> 16) & 0xf) << 13) | 0x44800000;
  float8_128 q1 = *(float8_128*)(&q1_i);

//   int8_128 q1 = ((qa & 0x00f000f0) | c0);  // half2( q[2] * 16 + 1024, q[3] * 16 + 1024 )
  int8_128 q2_i = ((qa & 0xf0) << 13) | 0x44800000;
  float8_128 q2 = *(float8_128*)(&q2_i);
  int8_128 q3_i = (((qa >> 16) & 0xf0) << 13) | 0x44800000;
  float8_128 q3 = *(float8_128*)(&q3_i);


  qa = v_u32_shr(qa, v_u32_move_i(8));

//   int8_128 q2 = ((qa & 0x000f000f) | c0);  // half2( q[4]      + 1024, q[5]      + 1024 )
//   int8_128 q3 = ((qa & 0x00f000f0) | c0);  // half2( q[6] * 16 + 1024, q[7] * 16 + 1024 )

  int8_128 q4_i = ((qa & 0xf) << 13) | 0x44800000;
  float8_128 q4 = *(float8_128*)(&q4_i);
  int8_128 q5_i = (((qa >> 16) & 0xf) << 13) | 0x44800000;
  float8_128 q5 = *(float8_128*)(&q5_i);

  int8_128 q6_i = ((qa & 0xf0) << 13) | 0x44800000;
  float8_128 q6 = *(float8_128*)(&q6_i);
  int8_128 q7_i = (((qa >> 16) & 0xf0) << 13) | 0x44800000;
  float8_128 q7 = *(float8_128*)(&q7_i);

// if(get_device_id() == 0){
//   // Print("q0_i:%f \n",q0);
//   // Print("q1_i:%f \n",q1);
//   // Print("q2_i:%f \n",q2);
//   // Print("q3_i:%f \n",q3);
//   // Print("q4_i:%f \n",q4);
//   // Print("q5_i:%f \n",q5);
//   // Print("q6_i:%f \n",q6);
//   // Print("q7_i:%f \n",q7);
// }



//   if (scaled) {
    //乘法转float用

    // dq[0] = __hfma2(q0.as_half2, y1y16[0],
    //                 z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)
    // dq[1] = __hfma2(q1.as_half2, y1y16[1],
    //                 z1z16[1]);  // half2( q[2] * s - z * s, q[3] * s - z * s)
    // dq[2] = __hfma2(q2.as_half2, y1y16[0], z1z16[0]);
    // dq[3] = __hfma2(q3.as_half2, y1y16[1], z1z16[1]);

    // float8_128 q0_f_low = dlc_half2float(q0);
    // float8_128 q0_f_high = dlc_half2float(v_u32_shr(q0,v_u32_move_i(16)));

    // *dq[0] = v_f32_add_b(v_f32_mul_b(q0_f_low, *y1y16[0]),*z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)
    // *dq[1] = v_f32_add_b(v_f32_mul_b(q0_f_high, *y1y16[0]),*z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)


    // float8_128 q1_f_low = dlc_half2float(q1);
    // float8_128 q1_f_high = dlc_half2float(v_u32_shr(q1,v_u32_move_i(16)));
    // *dq[2] = v_f32_add_b(v_f32_mul_b(q1_f_low, *y1y16[1]),*z1z16[1]);               // half2( q[2] * s - z * s, q[3] * s - z * s)
    // *dq[3] = v_f32_add_b(v_f32_mul_b(q1_f_high, *y1y16[1]),*z1z16[1]);               // half2( q[2] * s - z * s, q[3] * s - z * s)


    // float8_128 q2_f_low = dlc_half2float(q2);
    // float8_128 q2_f_high = dlc_half2float(v_u32_shr(q2,v_u32_move_i(16)));
    // *dq[4] = v_f32_add_b(v_f32_mul_b(q2_f_low, *y1y16[0]), *z1z16[0]);
    // *dq[5] = v_f32_add_b(v_f32_mul_b(q2_f_high, *y1y16[0]), *z1z16[0]);


    // float8_128 q3_f_low = dlc_half2float(q3);
    // float8_128 q3_f_high = dlc_half2float(v_u32_shr(q3,v_u32_move_i(16)));
    // *dq[6] = v_f32_add_b(v_f32_mul_b(q3_f_low, *y1y16[1]), *z1z16[1]);
    // *dq[7] = v_f32_add_b(v_f32_mul_b(q3_f_high, *y1y16[1]), *z1z16[1]);


//   } else {

    // dq[0] = __hadd2(q0.as_half2, z1z16[0]);  // half2( q[0] - z, q[1] - z )
    // dq[1] = __hfma2(q1.as_half2, y1y16[1],
    //                 z1z16[1]);               // half2( q[2] - z, q[3] - z )
    // dq[2] = __hadd2(q2.as_half2, z1z16[0]);  // half2( q[4] - z, q[5] - z )
    // dq[3] = __hfma2(q3.as_half2, y1y16[1],
    //                 z1z16[1]);  // half2( q[6] - z, q[7] - z )

    *dq_0 = v_f32_add_b(q0,*z1z16_0);  // half2( q[0] - z, q[1] - z )
    *dq_1 = v_f32_add_b(q1,*z1z16_0);  // half2( q[0] - z, q[1] - z )
    

    *dq_2 = v_f32_add_b(v_f32_mul_b(q2, *y1y16_1),*z1z16_1);               // half2( q[2] - z, q[3] - z )
    *dq_3 = v_f32_add_b(v_f32_mul_b(q3, *y1y16_1),*z1z16_1);               // half2( q[2] - z, q[3] - z )

    *dq_4 = v_f32_add_b(q4, *z1z16_0);  // half2( q[4] - z, q[5] - z )
    *dq_5 = v_f32_add_b(q5, *z1z16_0);  // half2( q[4] - z, q[5] - z )

    *dq_6 = v_f32_add_b(v_f32_mul_b(q6, *y1y16_1),*z1z16_1);  // half2( q[6] - z, q[7] - z )
    *dq_7 = v_f32_add_b(v_f32_mul_b(q7, *y1y16_1),*z1z16_1);  // half2( q[6] - z, q[7] - z )


    // *dq_0 = fp(*dq_0);
    // *dq_1 = fp(*dq_1);
    // *dq_2 = fp(*dq_2);
    // *dq_3 = fp(*dq_3);
    // *dq_4 = fp(*dq_4);
    // *dq_5 = fp(*dq_5);
    // *dq_6 = fp(*dq_6);
    // *dq_7 = fp(*dq_7);
    // }
}

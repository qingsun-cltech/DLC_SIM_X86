#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#ifndef __LDST_H_X86__
#define __LDST_H_X86__


#include "math.h"

inline float8_128 vmix(float8_128 a, float8_128 b) {
    int8_128 ia = *(int8_128*)(&a);
    int8_128 ib = *(int8_128*)(&b);
    int8_128 ic = ia | ib;
    return *(float8_128*)(&ic);
}

inline float8_128 load8_128_stride2(const int off, const int st, SIM_X86::tensor t) {
    float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, 0b11110000);
    float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, 0b00001111);
    return vmix(a, b);
}

inline int8_128 load8_128_stride2_i(const int off, const int st, SIM_X86::tensor t) {
    int8_128 a = v_i32_ld_tnsr(off, t, st, 0b11110000);
    int8_128 b = v_i32_ld_tnsr(off, t, st, 0b00001111);
    return (a | b);
}

inline float8_128 load8_128_stride4(const int off, const int st, SIM_X86::tensor t) {
    float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, 0b11000000);
    float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, 0b00110000);
    float8_128 ab = vmix(a, b);
    float8_128 c = v_f32_ld_tnsr_st_msk(off, t, st, 0b00001100);
    float8_128 d = v_f32_ld_tnsr_st_msk(off, t, st, 0b00000011);
    float8_128 cd = vmix(c, d);
    return vmix(ab, cd);
}

inline float8_128 load8_128_stride8(const int off, const int st, SIM_X86::tensor t) {
    float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, 0b10000000);
    float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, 0b01000000);
    float8_128 ab = vmix(a, b);
    float8_128 c = v_f32_ld_tnsr_st_msk(off, t, st, 0b00100000);
    float8_128 d = v_f32_ld_tnsr_st_msk(off, t, st, 0b00010000);
    float8_128 cd = vmix(c, d);
    float8_128 abcd = vmix(ab, cd);
    float8_128 e = v_f32_ld_tnsr_st_msk(off, t, st, 0b00001000);
    float8_128 f = v_f32_ld_tnsr_st_msk(off, t, st, 0b00000100);
    float8_128 ef = vmix(e, f);
    float8_128 g = v_f32_ld_tnsr_st_msk(off, t, st, 0b00000010);
    float8_128 h = v_f32_ld_tnsr_st_msk(off, t, st, 0b00000001);
    float8_128 gh = vmix(g, h);
    float8_128 efgh = vmix(ef, gh);
    return vmix(abcd, efgh);
}

inline float8_128 load8_128_stride(const int off, const int st, SIM_X86::tensor t) {
    if ((st & 1) == 1) {
        return v_f32_ld_tnsr_st(off, t, st);
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        return load8_128_stride2(off, st, t);
    } else if ((st & 7) == 4) {
        return load8_128_stride4(off, st, t);
    } else {
        return load8_128_stride8(off, st, t);
    }
}

inline float8_128 load8_128_stride2_with_ldmask(const int off, const int st, const int ldmask, SIM_X86::tensor t)
{
    float8_128 res = 0;
    if(ldmask & 0b11110000){
        float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b11110000);
        res = vmix(a, res);
    }
    if(ldmask & 0b00001111){
        float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00001111);
        res = vmix(b, res);
    }
    return res;
}

inline float8_128 load8_128_stride4_with_ldmask(const int off, const int st, const int ldmask, SIM_X86::tensor t)
{
    float8_128 res = 0;
    if(ldmask & 0b11000000){
        float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b11000000);
        res = vmix(a, res);
    }
    if(ldmask & 0b00110000){
        float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00110000);
        res = vmix(b, res);
    }
    if(ldmask & 0b00001100){
        float8_128 c = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00001100);
        res = vmix(c, res);
    }
    if(ldmask & 0b00000011){
        float8_128 d = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00000011);
        res = vmix(d, res);
    }
    return res;
}

inline float8_128 load8_128_stride8_with_ldmask(const int off, const int st, const int ldmask, SIM_X86::tensor t)
{
    float8_128 res = 0;
    if(ldmask & 0b10000000){
        float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b10000000);
        res = vmix(a, res);
    }
    if(ldmask & 0b01000000){
        float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b01000000);
        res = vmix(b, res);
    }
    if(ldmask & 0b00100000){
        float8_128 c = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00100000);
        res = vmix(c, res);
    }
    if(ldmask & 0b00010000){
        float8_128 d = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00010000);
        res = vmix(d, res);
    }
    if(ldmask & 0b00001000){
        float8_128 e = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00001000);
        res = vmix(e, res);
    }
    if(ldmask & 0b00000100){
        float8_128 f = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00000100);
        res = vmix(f, res);
    }
    if(ldmask & 0b00000010){
        float8_128 g = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00000010);
        res = vmix(g, res);
    }
    if(ldmask & 0b00000001){
        float8_128 h = v_f32_ld_tnsr_st_msk(off, t, st, ldmask & 0b00000001);
        res = vmix(h, res);
    }
    return res;
}

inline float8_128 load8_128_stride_with_ldmask(const int off, const int st, const int ldmask, SIM_X86::tensor t)
{
    if ((st & 1) == 1)
    {
        return v_f32_ld_tnsr_st_msk(off, t, st, ldmask);
    }
    else if ((st & 7) == 2 || (st & 7) == 6)
    {
        return load8_128_stride2_with_ldmask(off, st, ldmask, t);
    }
    else if ((st & 7) == 4)
    {
        return load8_128_stride4_with_ldmask(off, st, ldmask, t);
    }
    else
    {
        return load8_128_stride8_with_ldmask(off, st, ldmask, t);
    }
}

inline int8_128 load8_128_stride2_with_ldmask_i(const int off, const int st, const int ldmask, SIM_X86::tensor t)
{
    int8_128 res = 0;
    if(ldmask & 0b11110000){
        int8_128 a = v_i32_ld_tnsr(off, t, st, ldmask & 0b11110000);
        res = v_s32_add(a, res);
    }
    if(ldmask & 0b00001111){
        int8_128 b = v_i32_ld_tnsr(off, t, st, ldmask & 0b00001111);
        res = v_s32_add(b, res);
    }
    return res;
}

inline int8_128 load8_128_stride4_with_ldmask_i(const int off, const int st, const int ldmask, SIM_X86::tensor t)
{
    int8_128 res = 0;
    if(ldmask & 0b11000000){
        int8_128 a = v_i32_ld_tnsr(off, t, st, ldmask & 0b11000000);
        res = v_s32_add(a, res);
    }
    if(ldmask & 0b00110000){
        int8_128 b = v_i32_ld_tnsr(off, t, st, ldmask & 0b00110000);
        res = v_s32_add(b, res);
    }
    if(ldmask & 0b00001100){
        int8_128 c = v_i32_ld_tnsr(off, t, st, ldmask & 0b00001100);
        res = v_s32_add(c, res);
    }
    if(ldmask & 0b00000011){
        int8_128 d = v_i32_ld_tnsr(off, t, st, ldmask & 0b00000011);
        res = v_s32_add(d, res);
    }
    return res;
}

inline int8_128 load8_128_stride8_with_ldmask_i(const int off, const int st, const int ldmask, SIM_X86::tensor t)
{
    int8_128 res = 0;
    if(ldmask & 0b10000000){
        int8_128 a = v_i32_ld_tnsr(off, t, st, ldmask & 0b10000000);
        res = v_s32_add(a, res);
    }
    if(ldmask & 0b01000000){
        int8_128 b = v_i32_ld_tnsr(off, t, st, ldmask & 0b01000000);
        res = v_s32_add(b, res);
    }
    if(ldmask & 0b00100000){
        int8_128 c = v_i32_ld_tnsr(off, t, st, ldmask & 0b00100000);
        res = v_s32_add(c, res);
    }
    if(ldmask & 0b00010000){
        int8_128 d = v_i32_ld_tnsr(off, t, st, ldmask & 0b00010000);
        res = v_s32_add(d, res);
    }
    if(ldmask & 0b00001000){
        int8_128 e = v_i32_ld_tnsr(off, t, st, ldmask & 0b00001000);
        res = v_s32_add(e, res);
    }
    if(ldmask & 0b00000100){
        int8_128 f = v_i32_ld_tnsr(off, t, st, ldmask & 0b00000100);
        res = v_s32_add(f, res);
    }
    if(ldmask & 0b00000010){
        int8_128 g = v_i32_ld_tnsr(off, t, st, ldmask & 0b00000010);
        res = v_s32_add(g, res);
    }
    if(ldmask & 0b00000001){
        int8_128 h = v_i32_ld_tnsr(off, t, st, ldmask & 0b00000001);
        res = v_s32_add(h, res);
    }
    return res;
}

inline int8_128 load8_128_stride_with_ldmask_i(const int off, const int st, const int ldmask, SIM_X86::tensor t)
{
    if ((st & 1) == 1)
    {
        return v_i32_ld_tnsr(off, t, st, ldmask);
    }
    else if ((st & 7) == 2 || (st & 7) == 6)
    {
        return load8_128_stride2_with_ldmask_i(off, st, ldmask, t);
    }
    else if ((st & 7) == 4)
    {
        return load8_128_stride4_with_ldmask_i(off, st, ldmask, t);
    }
    else
    {
        return load8_128_stride8_with_ldmask_i(off, st, ldmask, t);
    }
}

inline float8_128 load8_128_stride2_ldmk(const int off, const int st, SIM_X86::tensor t, int ldmk) {
    float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, 0b11110000 & ldmk);
    float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, 0b00001111 & ldmk);
    return vmix(a, b);
}

inline float8_128 load8_128_stride4_ldmk(const int off, const int st, SIM_X86::tensor t, int ldmk) {
    float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, 0b11000000 & ldmk);
    float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, 0b00110000 & ldmk);
    float8_128 ab = vmix(a, b);
    float8_128 c = v_f32_ld_tnsr_st_msk(off, t, st, 0b00001100 & ldmk);
    float8_128 d = v_f32_ld_tnsr_st_msk(off, t, st, 0b00000011 & ldmk);
    float8_128 cd = vmix(c, d);
    return vmix(ab, cd);
}

inline float8_128 load8_128_stride8_ldmk(const int off, const int st, SIM_X86::tensor t, int ldmk) {
    float8_128 a = v_f32_ld_tnsr_st_msk(off, t, st, 0b10000000 & ldmk);
    float8_128 b = v_f32_ld_tnsr_st_msk(off, t, st, 0b01000000 & ldmk);
    float8_128 ab = vmix(a, b);
    float8_128 c = v_f32_ld_tnsr_st_msk(off, t, st, 0b00100000 & ldmk);
    float8_128 d = v_f32_ld_tnsr_st_msk(off, t, st, 0b00010000 & ldmk);
    float8_128 cd = vmix(c, d);
    float8_128 abcd = vmix(ab, cd);
    float8_128 e = v_f32_ld_tnsr_st_msk(off, t, st, 0b00001000 & ldmk);
    float8_128 f = v_f32_ld_tnsr_st_msk(off, t, st, 0b00000100 & ldmk);
    float8_128 ef = vmix(e, f);
    float8_128 g = v_f32_ld_tnsr_st_msk(off, t, st, 0b00000010 & ldmk);
    float8_128 h = v_f32_ld_tnsr_st_msk(off, t, st, 0b00000001 & ldmk);
    float8_128 gh = vmix(g, h);
    float8_128 efgh = vmix(ef, gh);
    return vmix(abcd, efgh);
}

inline float8_128 load8_128_stride_ldmk(const int off, const int st, SIM_X86::tensor t, int ldmk) {
    if ((st & 1) == 1) {
        return v_f32_ld_tnsr_st_msk(off, t, st, ldmk);
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        return load8_128_stride2_ldmk(off, st, t, ldmk);
    } else if ((st & 7) == 4) {
        return load8_128_stride4_ldmk(off, st, t, ldmk);
    } else {
        return load8_128_stride8_ldmk(off, st, t, ldmk);
    }
}

inline float8_128 load8_128_stride2_wh(const int off, const int st, SIM_X86::tensor t, int ldmk, int src_weight) {
    int8_128 core_id = v_u32_and(get_core_id(), v_u32_move_i(0x7f));
    bool8_128 vmask = v_s32_cmp(LSEQ, core_id, src_weight);
    float8_128 a = v_ld_vmsk(off, t, st, 0b11110000 & ldmk, vmask);
    float8_128 b = v_ld_vmsk(off, t, st, 0b00001111 & ldmk, vmask);
    return vmix(a, b);
}

inline float8_128 load8_128_stride4_wh(const int off, const int st, SIM_X86::tensor t, int ldmk, int src_weight) {
    int8_128 core_id = v_u32_and(get_core_id(), v_u32_move_i(0x7f));
    bool8_128 vmask = v_s32_cmp(LSEQ, core_id, src_weight);
    float8_128 a = v_ld_vmsk(off, t, st, 0b11000000 & ldmk, vmask);
    float8_128 b = v_ld_vmsk(off, t, st, 0b00110000 & ldmk, vmask);
    float8_128 ab = vmix(a, b);
    float8_128 c = v_ld_vmsk(off, t, st, 0b00001100 & ldmk, vmask);
    float8_128 d = v_ld_vmsk(off, t, st, 0b00000011 & ldmk, vmask);
    float8_128 cd = vmix(c, d);
    return vmix(ab, cd);
}

inline float8_128 load8_128_stride8_wh(const int off, const int st, SIM_X86::tensor t, int ldmk, int src_weight) {
    int8_128 core_id = v_u32_and(get_core_id(), v_u32_move_i(0x7f));
    bool8_128 vmask = v_s32_cmp(LSEQ, core_id, src_weight);
    float8_128 a = v_ld_vmsk(off, t, st, 0b10000000 & ldmk, vmask);
    float8_128 b = v_ld_vmsk(off, t, st, 0b01000000 & ldmk, vmask);
    float8_128 ab = vmix(a, b);
    float8_128 c = v_ld_vmsk(off, t, st, 0b00100000 & ldmk, vmask);
    float8_128 d = v_ld_vmsk(off, t, st, 0b00010000 & ldmk, vmask);
    float8_128 cd = vmix(c, d);
    float8_128 abcd = vmix(ab, cd);
    float8_128 e = v_ld_vmsk(off, t, st, 0b00001000 & ldmk, vmask);
    float8_128 f = v_ld_vmsk(off, t, st, 0b00000100 & ldmk, vmask);
    float8_128 ef = vmix(e, f);
    float8_128 g = v_ld_vmsk(off, t, st, 0b00000010 & ldmk, vmask);
    float8_128 h = v_ld_vmsk(off, t, st, 0b00000001 & ldmk, vmask);
    float8_128 gh = vmix(g, h);
    float8_128 efgh = vmix(ef, gh);
    return vmix(abcd, efgh);
}

inline float8_128 load8_128_stride_wh(const int off, const int st, SIM_X86::tensor t, int ldmk, int src_weight) {
    if ((st & 1) == 1) {
        int8_128 core_id = v_u32_and(get_core_id(), v_u32_move_i(0x7f));
        bool8_128 vmask = v_s32_cmp(LSEQ, core_id, src_weight);
        return v_ld_vmsk(off, t, st, ldmk, vmask);
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        return load8_128_stride2_wh(off, st, t, ldmk, src_weight);
    } else if ((st & 7) == 4) {
        return load8_128_stride4_wh(off, st, t, ldmk, src_weight);
    } else {
        return load8_128_stride8_wh(off, st, t, ldmk, src_weight);
    }
}

inline void store8_128_stride2(const int off, const int st, SIM_X86::tensor t, float8_128 data) {
    v_f32_st_tnsr_st_msk(off, t, st, 0b11110000, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00001111, data);
}

inline void store8_128_stride4(const int off, const int st, SIM_X86::tensor t, float8_128 data) {
    v_f32_st_tnsr_st_msk(off, t, st, 0b11000000, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00110000, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00001100, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00000011, data);
}

inline void store8_128_stride8(const int off, const int st, SIM_X86::tensor t, float8_128 data) {
    v_f32_st_tnsr_st_msk(off, t, st, 0b10000000, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b01000000, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00100000, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00010000, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00001000, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00000100, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00000010, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00000001, data);
}

inline void store8_128_stride(const int off, const int st, SIM_X86::tensor t, float8_128 data) {
    if ((st & 1) == 1) {
        v_f32_st_tnsr_st(off, t, st, data);
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        store8_128_stride2(off, st, t, data);
    } else if ((st & 7) == 4) {
        store8_128_stride4(off, st, t, data);
    } else {
        store8_128_stride8(off, st, t, data);
    }
}

inline void store8_128_stride2_i(const int off, const int st, SIM_X86::tensor t, int8_128 data) {
    v_st_generic(off, t, st, 0b11110000, data);
    v_st_generic(off, t, st, 0b00001111, data);
}

inline void store8_128_stride4_i(const int off, const int st, SIM_X86::tensor t, int8_128 data) {
    v_st_generic(off, t, st, 0b11000000, data);
    v_st_generic(off, t, st, 0b00110000, data);
    v_st_generic(off, t, st, 0b00001100, data);
    v_st_generic(off, t, st, 0b00000011, data);
}

inline void store8_128_stride8_i(const int off, const int st, SIM_X86::tensor t, int8_128 data) {
    v_st_generic(off, t, st, 0b10000000, data);
    v_st_generic(off, t, st, 0b01000000, data);
    v_st_generic(off, t, st, 0b00100000, data);
    v_st_generic(off, t, st, 0b00010000, data);
    v_st_generic(off, t, st, 0b00001000, data);
    v_st_generic(off, t, st, 0b00000100, data);
    v_st_generic(off, t, st, 0b00000010, data);
    v_st_generic(off, t, st, 0b00000001, data);
}
inline void store8_128_stride_i(const int off, const int st, SIM_X86::tensor t, int8_128 data) {
    if ((st & 1) == 1) {
        v_st_generic(off, t, st, data);
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        store8_128_stride2_i(off, st, t, data);
    } else if ((st & 7) == 4) {
        store8_128_stride4_i(off, st, t, data);
    } else {
        store8_128_stride8_i(off, st, t, data);
    }
}

inline void store8_128_stride2_with_stmask(const int off, const int st, const int stmask, SIM_X86::tensor t, float8_128 data)
{   
    if(stmask & 0b11110000){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b11110000, data);
    }
    if(stmask & 0b00001111){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00001111, data);
    }
}

inline void store8_128_stride4_with_stmask(const int off, const int st, const int stmask, SIM_X86::tensor t, float8_128 data)
{
    if(stmask & 0b11000000){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b11000000, data);
    }
    if(stmask & 0b00110000){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00110000, data);
    }
    if(stmask & 0b00001100){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00001100, data);
    }
    if(stmask & 0b00000011){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00000011, data);
    }
}

inline void store8_128_stride8_with_stmask(const int off, const int st, const int stmask, SIM_X86::tensor t, float8_128 data)
{   
    if(stmask & 0b10000000){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b10000000, data);
    }
    if(stmask & 0b01000000){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b01000000, data);
    }
    if(stmask & 0b00100000){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00100000, data);
    }
    if(stmask & 0b00010000){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00010000, data);
    }
    if(stmask & 0b00001000){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00001000, data);
    }
    if(stmask & 0b00000100){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00000100, data);
    }
    if(stmask & 0b00000010){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00000010, data);
    }
    if(stmask & 0b00000001){
        v_f32_st_tnsr_st_msk(off, t, st, stmask & 0b00000001, data);
    }
}

inline void store8_128_stride_with_stmask(const int off, const int st, const int stmask, SIM_X86::tensor t, float8_128 data)
{
    if ((st & 1) == 1)
    {
        v_f32_st_tnsr_st_msk(off, t, st, stmask, data);
    }
    else if ((st & 7) == 2 || (st & 7) == 6)
    {
        store8_128_stride2_with_stmask(off, st, stmask, t, data);
    }
    else if ((st & 7) == 4)
    {
        store8_128_stride4_with_stmask(off, st, stmask, t, data);
    }
    else
    {
        store8_128_stride8_with_stmask(off, st, stmask, t, data);
    }
}

inline void store8_128_stride2_with_stmask_i(const int off, const int st, const int stmask, SIM_X86::tensor t, int8_128 data)
{   
    if(stmask & 0b11110000){
        v_st_generic(off, t, st, stmask & 0b11110000, data);
    }
    if(stmask & 0b00001111){
        v_st_generic(off, t, st, stmask & 0b00001111, data);
    }
}

inline void store8_128_stride4_with_stmask_i(const int off, const int st, const int stmask, SIM_X86::tensor t, int8_128 data)
{
    if(stmask & 0b11000000){
        v_st_generic(off, t, st, stmask & 0b11000000, data);
    }
    if(stmask & 0b00110000){
        v_st_generic(off, t, st, stmask & 0b00110000, data);
    }
    if(stmask & 0b00001100){
        v_st_generic(off, t, st, stmask & 0b00001100, data);
    }
    if(stmask & 0b00000011){
        v_st_generic(off, t, st, stmask & 0b00000011, data);
    }
}

inline void store8_128_stride8_with_stmask_i(const int off, const int st, const int stmask, SIM_X86::tensor t, int8_128 data)
{   
    if(stmask & 0b10000000){
        v_st_generic(off, t, st, stmask & 0b10000000, data);
    }
    if(stmask & 0b01000000){
        v_st_generic(off, t, st, stmask & 0b01000000, data);
    }
    if(stmask & 0b00100000){
        v_st_generic(off, t, st, stmask & 0b00100000, data);
    }
    if(stmask & 0b00010000){
        v_st_generic(off, t, st, stmask & 0b00010000, data);
    }
    if(stmask & 0b00001000){
        v_st_generic(off, t, st, stmask & 0b00001000, data);
    }
    if(stmask & 0b00000100){
        v_st_generic(off, t, st, stmask & 0b00000100, data);
    }
    if(stmask & 0b00000010){
        v_st_generic(off, t, st, stmask & 0b00000010, data);
    }
    if(stmask & 0b00000001){
        v_st_generic(off, t, st, stmask & 0b00000001, data);
    }
}

inline void store8_128_stride_with_stmask_i(const int off, const int st, const int stmask, SIM_X86::tensor t, int8_128 data)
{
    if ((st & 1) == 1)
    {
        v_st_generic(off, t, st, stmask, data);
    }
    else if ((st & 7) == 2 || (st & 7) == 6)
    {
        store8_128_stride2_with_stmask_i(off, st, stmask, t, data);
    }
    else if ((st & 7) == 4)
    {
        store8_128_stride4_with_stmask_i(off, st, stmask, t, data);
    }
    else
    {
        store8_128_stride8_with_stmask_i(off, st, stmask, t, data);
    }
}

inline void store8_128_stride2_with_stmask_cmem(const int off, const int st, const int stmask, SIM_X86::tensor t, float8_128 data)
{   
    if(stmask & 0b11110000){
        v_f32_fxc_store(off, t, st, stmask & 0b11110000, data);
    }
    if(stmask & 0b00001111){
        v_f32_fxc_store(off, t, st, stmask & 0b00001111, data);
    }
}
inline void store8_128_stride4_with_stmask_cmem(const int off, const int st, const int stmask, SIM_X86::tensor t, float8_128 data)
{
    if(stmask & 0b11000000){
        v_f32_fxc_store(off, t, st, stmask & 0b11000000, data);
    }
    if(stmask & 0b00110000){
        v_f32_fxc_store(off, t, st, stmask & 0b00110000, data);
    }
    if(stmask & 0b00001100){
        v_f32_fxc_store(off, t, st, stmask & 0b00001100, data);
    }
    if(stmask & 0b00000011){
        v_f32_fxc_store(off, t, st, stmask & 0b00000011, data);
    }
}
inline void store8_128_stride8_with_stmask_cmem(const int off, const int st, const int stmask, SIM_X86::tensor t, float8_128 data)
{   
    if(stmask & 0b10000000){
        v_f32_fxc_store(off, t, st, stmask & 0b10000000, data);
    }
    if(stmask & 0b01000000){
        v_f32_fxc_store(off, t, st, stmask & 0b01000000, data);
    }
    if(stmask & 0b00100000){
        v_f32_fxc_store(off, t, st, stmask & 0b00100000, data);
    }
    if(stmask & 0b00010000){
        v_f32_fxc_store(off, t, st, stmask & 0b00010000, data);
    }
    if(stmask & 0b00001000){
        v_f32_fxc_store(off, t, st, stmask & 0b00001000, data);
    }
    if(stmask & 0b00000100){
        v_f32_fxc_store(off, t, st, stmask & 0b00000100, data);
    }
    if(stmask & 0b00000010){
        v_f32_fxc_store(off, t, st, stmask & 0b00000010, data);
    }
    if(stmask & 0b00000001){
        v_f32_fxc_store(off, t, st, stmask & 0b00000001, data);
    }
}

inline void store8_128_stride_with_stmask_cmem(const int off, const int st, const int stmask, SIM_X86::tensor t, float8_128 data)
{
    if ((st & 1) == 1)
    {
        v_f32_fxc_store(off, t, st, stmask, data);
    }
    else if ((st & 7) == 2 || (st & 7) == 6)
    {
        store8_128_stride2_with_stmask_cmem(off, st, stmask, t, data);
    }
    else if ((st & 7) == 4)
    {
        store8_128_stride4_with_stmask_cmem(off, st, stmask, t, data);
    }
    else
    {
        store8_128_stride8_with_stmask_cmem(off, st, stmask, t, data);
    }
}

inline void store8_128_stride2_stmk(const int off, const int st, SIM_X86::tensor t, float8_128 data, int stmk) {
    v_f32_st_tnsr_st_msk(off, t, st, 0b11110000 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00001111 & stmk, data);
}

inline void store8_128_stride4_stmk(const int off, const int st, SIM_X86::tensor t, float8_128 data, int stmk) {
    v_f32_st_tnsr_st_msk(off, t, st, 0b11000000 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00110000 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00001100 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00000011 & stmk, data);
}

inline void store8_128_stride8_stmk(const int off, const int st, SIM_X86::tensor t, float8_128 data, int stmk) {
    v_f32_st_tnsr_st_msk(off, t, st, 0b10000000 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b01000000 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00100000 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00010000 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00001000 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00000100 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00000010 & stmk, data);
    v_f32_st_tnsr_st_msk(off, t, st, 0b00000001 & stmk, data);
}

inline void store8_128_stride_stmk(const int off, const int st, SIM_X86::tensor t, float8_128 data, int stmk) {
    if ((st & 1) == 1) {
        v_f32_st_tnsr_st_msk(off, t, st, stmk, data);
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        store8_128_stride2_stmk(off, st, t, data, stmk);
    } else if ((st & 7) == 4) {
        store8_128_stride4_stmk(off, st, t, data, stmk);
    } else {
        store8_128_stride8_stmk(off, st, t, data, stmk);
    }
}

inline void store8_128_stride2_stmk_i(const int off, const int st, SIM_X86::tensor t, int8_128 data, int stmk) {
    v_st_generic(off, t, st, 0b11110000 & stmk, data);
    v_st_generic(off, t, st, 0b00001111 & stmk, data);
}

inline void store8_128_stride4_stmk_i(const int off, const int st, SIM_X86::tensor t, int8_128 data, int stmk) {
    v_st_generic(off, t, st, 0b11000000 & stmk, data);
    v_st_generic(off, t, st, 0b00110000 & stmk, data);
    v_st_generic(off, t, st, 0b00001100 & stmk, data);
    v_st_generic(off, t, st, 0b00000011 & stmk, data);
}

inline void store8_128_stride8_stmk_i(const int off, const int st, SIM_X86::tensor t, int8_128 data, int stmk) {
    v_st_generic(off, t, st, 0b10000000 & stmk, data);
    v_st_generic(off, t, st, 0b01000000 & stmk, data);
    v_st_generic(off, t, st, 0b00100000 & stmk, data);
    v_st_generic(off, t, st, 0b00010000 & stmk, data);
    v_st_generic(off, t, st, 0b00001000 & stmk, data);
    v_st_generic(off, t, st, 0b00000100 & stmk, data);
    v_st_generic(off, t, st, 0b00000010 & stmk, data);
    v_st_generic(off, t, st, 0b00000001 & stmk, data);
}

inline void store8_128_stride_stmk_i(const int off, const int st, SIM_X86::tensor t, int8_128 data, int stmk) {
    if ((st & 1) == 1) {
        v_st_generic(off, t, st, stmk, data);
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        store8_128_stride2_stmk_i(off, st, t, data, stmk);
    } else if ((st & 7) == 4) {
        store8_128_stride4_stmk_i(off, st, t, data, stmk);
    } else {
        store8_128_stride8_stmk_i(off, st, t, data, stmk);
    }
}

inline void store8_128_stride2_mk(const int off, const int st, SIM_X86::tensor t, float8_128 data, int stmk,
                                  bool8_128 vmsk) {
    v_st_vmsk(off, t, st, 0b11110000 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00001111 & stmk, vmsk, data);
}

inline void store8_128_stride4_mk(const int off, const int st, SIM_X86::tensor t, float8_128 data, int stmk,
                                  bool8_128 vmsk) {
    v_st_vmsk(off, t, st, 0b11000000 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00110000 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00001100 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00000011 & stmk, vmsk, data);
}

inline void store8_128_stride8_mk(const int off, const int st, SIM_X86::tensor t, float8_128 data, int stmk,
                                  bool8_128 vmsk) {
    v_st_vmsk(off, t, st, 0b10000000 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b01000000 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00100000 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00010000 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00001000 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00000100 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00000010 & stmk, vmsk, data);
    v_st_vmsk(off, t, st, 0b00000001 & stmk, vmsk, data);
}

inline void store8_128_stride_mk(const int off, const int st, SIM_X86::tensor t, float8_128 data, int stmk,
                                 bool8_128 vmsk) {
    if ((st & 1) == 1) {
        v_st_vmsk(off, t, st, stmk, vmsk, data);
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        store8_128_stride2_mk(off, st, t, data, stmk, vmsk);
    } else if ((st & 7) == 4) {
        store8_128_stride4_mk(off, st, t, data, stmk, vmsk);
    } else {
        store8_128_stride8_mk(off, st, t, data, stmk, vmsk);
    }
}

inline float128_128 load128_128(const int off, SIM_X86::tensor t) {
    float8_128 a[16];
    for (int i = 0; i < 16; i++) {
        a[i] = v_f32_ld_tnsr_b(off + i * 32, t);
    }
    return v_concat_16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
                       a[14], a[15]);
}

inline float128_128 load128_128_stride(const int off, const int st, SIM_X86::tensor t) {
    float8_128 a[16];
    if ((st & 1) == 1) {
        for (int i = 0; i < 16; i++) {
            a[i] = v_f32_ld_tnsr_st(off + i * st * 32, t, st);
        }
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        for (int i = 0; i < 16; i++) {
            a[i] = load8_128_stride2(off + i * st * 32, st, t);
        }
    } else if ((st & 7) == 4) {
        a[0] = load8_128_stride4(off + 0 * st * 32, st, t);
        a[1] = load8_128_stride4(off + 1 * st * 32, st, t);
        a[2] = load8_128_stride4(off + 2 * st * 32, st, t);
        a[3] = load8_128_stride4(off + 3 * st * 32, st, t);
        a[4] = load8_128_stride4(off + 4 * st * 32, st, t);
        a[5] = load8_128_stride4(off + 5 * st * 32, st, t);
        a[6] = load8_128_stride4(off + 6 * st * 32, st, t);
        a[7] = load8_128_stride4(off + 7 * st * 32, st, t);
        a[8] = load8_128_stride4(off + 8 * st * 32, st, t);
        a[9] = load8_128_stride4(off + 9 * st * 32, st, t);
        a[10] = load8_128_stride4(off + 10 * st * 32, st, t);
        a[11] = load8_128_stride4(off + 11 * st * 32, st, t);
        a[12] = load8_128_stride4(off + 12 * st * 32, st, t);
        a[13] = load8_128_stride4(off + 13 * st * 32, st, t);
        a[14] = load8_128_stride4(off + 14 * st * 32, st, t);
        a[15] = load8_128_stride4(off + 15 * st * 32, st, t);
    } else {
        a[0] = load8_128_stride8(off + 0 * st * 32, st, t);
        a[1] = load8_128_stride8(off + 1 * st * 32, st, t);
        a[2] = load8_128_stride8(off + 2 * st * 32, st, t);
        a[3] = load8_128_stride8(off + 3 * st * 32, st, t);
        a[4] = load8_128_stride8(off + 4 * st * 32, st, t);
        a[5] = load8_128_stride8(off + 5 * st * 32, st, t);
        a[6] = load8_128_stride8(off + 6 * st * 32, st, t);
        a[7] = load8_128_stride8(off + 7 * st * 32, st, t);
        a[8] = load8_128_stride8(off + 8 * st * 32, st, t);
        a[9] = load8_128_stride8(off + 9 * st * 32, st, t);
        a[10] = load8_128_stride8(off + 10 * st * 32, st, t);
        a[11] = load8_128_stride8(off + 11 * st * 32, st, t);
        a[12] = load8_128_stride8(off + 12 * st * 32, st, t);
        a[13] = load8_128_stride8(off + 13 * st * 32, st, t);
        a[14] = load8_128_stride8(off + 14 * st * 32, st, t);
        a[15] = load8_128_stride8(off + 15 * st * 32, st, t);
    }
    return v_concat_16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
                       a[14], a[15]);
}

//可以设置h的长度
inline float128_128 load128_128_stride_h(const int off, const int st, const int h, SIM_X86::tensor t)
{
    float8_128  __attribute__((address_space(2))) a[16];
    
    //case1: complete 8*128
    const int com_8_128_num = h/8;
    //case2: incomplete 8*128
    const int incom_8_128_num = h%8;
    const int ld_mask = pre_exp2(incom_8_128_num);
    //case3：0
    if ((st & 1) == 1)
    {
        for (int i = 0; i < 16; i++)
        {   
            if(i < com_8_128_num){
                a[i] = v_f32_ld_tnsr_st(off + i * st * 32, t, st);
            }else if(i == com_8_128_num && incom_8_128_num){
                a[i] = v_f32_ld_tnsr_st_msk(off + i * st * 32, t, st, ld_mask);  
            }else{
                a[i] = 0;
            }
        }
    }
    else if ((st & 7) == 2 || (st & 7) == 6)
    {
        for (int i = 0; i < 16; i++)
        {   
            if(i < com_8_128_num){
                a[i] = load8_128_stride2(off + i * st * 32, st, t);
            }else if(i == com_8_128_num && incom_8_128_num){
                a[i] = load8_128_stride2_with_ldmask(off + i * st * 32, st, ld_mask, t);  
            }else{
                a[i] = 0;
            }
        }
    }
    else if ((st & 7) == 4)
    {
        for (int i = 0; i < 16; i++)
        {   
            if(i < com_8_128_num){
                a[i] = load8_128_stride4(off + i * st * 32, st, t);
            }else if(i == com_8_128_num && incom_8_128_num){
                a[i] = load8_128_stride4_with_ldmask(off + i * st * 32, st, ld_mask, t);  
            }else{
                a[i] = 0;
            }
        }
    }
    else
    {
        for (int i = 0; i < 16; i++)
        {   
            if(i < com_8_128_num){
                a[i] = load8_128_stride8(off + i * st * 32, st, t);
            }else if(i == com_8_128_num && incom_8_128_num){
                a[i] = load8_128_stride8_with_ldmask(off + i * st * 32, st, ld_mask, t);  
            }else{
                a[i] = 0;
            }
        }
    }
    return v_concat_16(
        a[0], a[1], a[2], a[3],
        a[4], a[5], a[6], a[7],
        a[8], a[9], a[10], a[11],
        a[12], a[13], a[14], a[15]);
}

//_mh、_mw都是未align到128的长度
//row、col是目前要取的128 * 128的起始点坐标，向下为row坐标，向右为col坐标，mat的左上角为[0, 0]
inline float128_128 load128_128from_matrix_line_layout_wh(
    SIM_X86::tensor mat, const int _mh, const int _mw, const int row, const int col){
    int mw = (_mw + 127) & 0xffffff80;
    int off = (row * mw + col) / 32;
    const int st = mw / 128;
    float8_128  __attribute__((address_space(2))) a[16];
    int h = min(_mh - row, 128);
    int w = min(_mw - col, 128);
    //case1: complete 8*128
    const int com_8_128_num = h/8;
    //case2: incomplete 8*128
    const int incom_8_128_num = h%8;
    const int ld_mask = pre_exp2(incom_8_128_num);
    //case3：0
    for (int i = 0; i < 16; i++)
    {   
        if(i < com_8_128_num){
            a[i] = load8_128_stride_wh(off + i * st * 32, st, mat, 255, w);
        }else if(i == com_8_128_num && incom_8_128_num){
            a[i] = load8_128_stride_wh(off + i * st * 32, st, mat, ld_mask, w);
        }else{
            a[i] = 0;
        }
    }
    return v_concat_16(
        a[0], a[1], a[2], a[3],
        a[4], a[5], a[6], a[7],
        a[8], a[9], a[10], a[11],
        a[12], a[13], a[14], a[15]);
}

inline float8_128 load8_128from_matrix_line_layout_wh(
    SIM_X86::tensor mat, const int _mh, const int _mw, const int row, const int col){
    int mw = (_mw + 127) & 0xffffff80;
    int off = (row * mw + col) / 32;
    const int st = mw / 128;
    int h = min(_mh - row, 8);
    const int ld_mask = pre_exp2(h);
    int w = min(_mw - col, 128);
    return load8_128_stride_wh(off, st, mat, ld_mask, w);
}

inline void store128_128(const int off, SIM_X86::tensor t, float128_128 data) {
    v_f32_st_tnsr_b(off + 0 * 32, t, sub_vector(data, 0));
    v_f32_st_tnsr_b(off + 1 * 32, t, sub_vector(data, 1));
    v_f32_st_tnsr_b(off + 2 * 32, t, sub_vector(data, 2));
    v_f32_st_tnsr_b(off + 3 * 32, t, sub_vector(data, 3));
    v_f32_st_tnsr_b(off + 4 * 32, t, sub_vector(data, 4));
    v_f32_st_tnsr_b(off + 5 * 32, t, sub_vector(data, 5));
    v_f32_st_tnsr_b(off + 6 * 32, t, sub_vector(data, 6));
    v_f32_st_tnsr_b(off + 7 * 32, t, sub_vector(data, 7));
    v_f32_st_tnsr_b(off + 8 * 32, t, sub_vector(data, 8));
    v_f32_st_tnsr_b(off + 9 * 32, t, sub_vector(data, 9));
    v_f32_st_tnsr_b(off + 10 * 32, t, sub_vector(data, 10));
    v_f32_st_tnsr_b(off + 11 * 32, t, sub_vector(data, 11));
    v_f32_st_tnsr_b(off + 12 * 32, t, sub_vector(data, 12));
    v_f32_st_tnsr_b(off + 13 * 32, t, sub_vector(data, 13));
    v_f32_st_tnsr_b(off + 14 * 32, t, sub_vector(data, 14));
    v_f32_st_tnsr_b(off + 15 * 32, t, sub_vector(data, 15));
}

inline void store128_128_stride(const int off, const int st, SIM_X86::tensor t, float128_128 data) {
    if ((st & 1) == 1) {
        v_f32_st_tnsr_st(off + 0 * st * 32, t, st, sub_vector(data, 0));
        v_f32_st_tnsr_st(off + 1 * st * 32, t, st, sub_vector(data, 1));
        v_f32_st_tnsr_st(off + 2 * st * 32, t, st, sub_vector(data, 2));
        v_f32_st_tnsr_st(off + 3 * st * 32, t, st, sub_vector(data, 3));
        v_f32_st_tnsr_st(off + 4 * st * 32, t, st, sub_vector(data, 4));
        v_f32_st_tnsr_st(off + 5 * st * 32, t, st, sub_vector(data, 5));
        v_f32_st_tnsr_st(off + 6 * st * 32, t, st, sub_vector(data, 6));
        v_f32_st_tnsr_st(off + 7 * st * 32, t, st, sub_vector(data, 7));
        v_f32_st_tnsr_st(off + 8 * st * 32, t, st, sub_vector(data, 8));
        v_f32_st_tnsr_st(off + 9 * st * 32, t, st, sub_vector(data, 9));
        v_f32_st_tnsr_st(off + 10 * st * 32, t, st, sub_vector(data, 10));
        v_f32_st_tnsr_st(off + 11 * st * 32, t, st, sub_vector(data, 11));
        v_f32_st_tnsr_st(off + 12 * st * 32, t, st, sub_vector(data, 12));
        v_f32_st_tnsr_st(off + 13 * st * 32, t, st, sub_vector(data, 13));
        v_f32_st_tnsr_st(off + 14 * st * 32, t, st, sub_vector(data, 14));
        v_f32_st_tnsr_st(off + 15 * st * 32, t, st, sub_vector(data, 15));
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        store8_128_stride2(off + 0 * st * 32, st, t, sub_vector(data, 0));
        store8_128_stride2(off + 1 * st * 32, st, t, sub_vector(data, 1));
        store8_128_stride2(off + 2 * st * 32, st, t, sub_vector(data, 2));
        store8_128_stride2(off + 3 * st * 32, st, t, sub_vector(data, 3));
        store8_128_stride2(off + 4 * st * 32, st, t, sub_vector(data, 4));
        store8_128_stride2(off + 5 * st * 32, st, t, sub_vector(data, 5));
        store8_128_stride2(off + 6 * st * 32, st, t, sub_vector(data, 6));
        store8_128_stride2(off + 7 * st * 32, st, t, sub_vector(data, 7));
        store8_128_stride2(off + 8 * st * 32, st, t, sub_vector(data, 8));
        store8_128_stride2(off + 9 * st * 32, st, t, sub_vector(data, 9));
        store8_128_stride2(off + 10 * st * 32, st, t, sub_vector(data, 10));
        store8_128_stride2(off + 11 * st * 32, st, t, sub_vector(data, 11));
        store8_128_stride2(off + 12 * st * 32, st, t, sub_vector(data, 12));
        store8_128_stride2(off + 13 * st * 32, st, t, sub_vector(data, 13));
        store8_128_stride2(off + 14 * st * 32, st, t, sub_vector(data, 14));
        store8_128_stride2(off + 15 * st * 32, st, t, sub_vector(data, 15));
    } else if ((st & 7) == 4) {
        store8_128_stride4(off + 0 * st * 32, st, t, sub_vector(data, 0));
        store8_128_stride4(off + 1 * st * 32, st, t, sub_vector(data, 1));
        store8_128_stride4(off + 2 * st * 32, st, t, sub_vector(data, 2));
        store8_128_stride4(off + 3 * st * 32, st, t, sub_vector(data, 3));
        store8_128_stride4(off + 4 * st * 32, st, t, sub_vector(data, 4));
        store8_128_stride4(off + 5 * st * 32, st, t, sub_vector(data, 5));
        store8_128_stride4(off + 6 * st * 32, st, t, sub_vector(data, 6));
        store8_128_stride4(off + 7 * st * 32, st, t, sub_vector(data, 7));
        store8_128_stride4(off + 8 * st * 32, st, t, sub_vector(data, 8));
        store8_128_stride4(off + 9 * st * 32, st, t, sub_vector(data, 9));
        store8_128_stride4(off + 10 * st * 32, st, t, sub_vector(data, 10));
        store8_128_stride4(off + 11 * st * 32, st, t, sub_vector(data, 11));
        store8_128_stride4(off + 12 * st * 32, st, t, sub_vector(data, 12));
        store8_128_stride4(off + 13 * st * 32, st, t, sub_vector(data, 13));
        store8_128_stride4(off + 14 * st * 32, st, t, sub_vector(data, 14));
        store8_128_stride4(off + 15 * st * 32, st, t, sub_vector(data, 15));
    } else {
        store8_128_stride8(off + 0 * st * 32, st, t, sub_vector(data, 0));
        store8_128_stride8(off + 1 * st * 32, st, t, sub_vector(data, 1));
        store8_128_stride8(off + 2 * st * 32, st, t, sub_vector(data, 2));
        store8_128_stride8(off + 3 * st * 32, st, t, sub_vector(data, 3));
        store8_128_stride8(off + 4 * st * 32, st, t, sub_vector(data, 4));
        store8_128_stride8(off + 5 * st * 32, st, t, sub_vector(data, 5));
        store8_128_stride8(off + 6 * st * 32, st, t, sub_vector(data, 6));
        store8_128_stride8(off + 7 * st * 32, st, t, sub_vector(data, 7));
        store8_128_stride8(off + 8 * st * 32, st, t, sub_vector(data, 8));
        store8_128_stride8(off + 9 * st * 32, st, t, sub_vector(data, 9));
        store8_128_stride8(off + 10 * st * 32, st, t, sub_vector(data, 10));
        store8_128_stride8(off + 11 * st * 32, st, t, sub_vector(data, 11));
        store8_128_stride8(off + 12 * st * 32, st, t, sub_vector(data, 12));
        store8_128_stride8(off + 13 * st * 32, st, t, sub_vector(data, 13));
        store8_128_stride8(off + 14 * st * 32, st, t, sub_vector(data, 14));
        store8_128_stride8(off + 15 * st * 32, st, t, sub_vector(data, 15));
    }
}
inline void store128_128_stride_h(const int off, const int st, const int h, SIM_X86::tensor t, float128_128 data) {
    //case1: complete 8*128
    const int com_8_128_num = h/8;
    //case2: incomplete 8*128
    const int incom_8_128_num = h%8;
    const int st_mask = pre_exp2(incom_8_128_num);

    float8_128  __attribute__((address_space(2))) a[16];
    a[0] = sub_vector(data, 0);
    a[1] = sub_vector(data, 1);
    a[2] = sub_vector(data, 2);
    a[3] = sub_vector(data, 3);
    a[4] = sub_vector(data, 4);
    a[5] = sub_vector(data, 5);
    a[6] = sub_vector(data, 6);
    a[7] = sub_vector(data, 7);
    a[8] = sub_vector(data, 8);
    a[9] = sub_vector(data, 9);
    a[10] = sub_vector(data, 10);
    a[11] = sub_vector(data, 11);
    a[12] = sub_vector(data, 12);
    a[13] = sub_vector(data, 13);
    a[14] = sub_vector(data, 14);
    a[15] = sub_vector(data, 15);
    if ((st & 1) == 1) {
        int i = 0;
        for(; i < com_8_128_num; i++){
            v_f32_st_tnsr_st(off + i * st * 32, t, st, a[i]);
        }
        if(incom_8_128_num){
            v_f32_st_tnsr_st_msk(off + i * st * 32, t, st, st_mask, a[i]);
        }
    } else if ((st & 7) == 2 || (st & 7) == 6) {
        int i = 0;
        for(; i < com_8_128_num; i++){
            store8_128_stride2(off + i * st * 32, st, t, a[i]);
        }
        if(incom_8_128_num){
            store8_128_stride_with_stmask(off + i * st * 32, st, st_mask, t, a[i]);
        }
    } else if ((st & 7) == 4) {
        int i = 0;
       for(; i < com_8_128_num; i++){
            store8_128_stride4(off + i * st * 32, st, t, a[i]);
        }
        if(incom_8_128_num){
            store8_128_stride_with_stmask(off + i * st * 32, st, st_mask, t, a[i]);
        }
    } else {
        int i = 0;
       for(; i < com_8_128_num; i++){
            store8_128_stride8(off + i * st * 32, st, t, a[i]);
        }
        if(incom_8_128_num){
            store8_128_stride_with_stmask(off + i * st * 32, st, st_mask, t, a[i]);
        }
    }
}

inline float8_128 load8_128_stride_h(const int off, const int st, const int h, SIM_X86::tensor t){
    int ld_mask = pre_exp2(h);
    return load8_128_stride_with_ldmask(off, st, ld_mask, t);
}

inline void store8_128_stride_h(const int off, const int st, const int h, SIM_X86::tensor t, float8_128 data){
    int st_mask = pre_exp2(h);
    store8_128_stride_with_stmask(off, st, st_mask, t, data);
}

#define DEF_LOAD_STORE(unitH, unitW, offCalcExpr, layoutName)                                                \
    inline float##unitH##_##unitW load##unitH##_##unitW##from_matrix_##layoutName##_with_offset(             \
        const int offset, SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col) {            \
        const int _unitH = unitH;                                                                            \
        const int _unitW = unitW;                                                                            \
        return load##unitH##_##unitW##_stride(offset + (offCalcExpr), mw, mat);                              \
    }                                                                                                        \
    inline float##unitH##_##unitW load##unitH##_##unitW##from_matrix_##layoutName##_srch(                    \
        SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col, const int h) {                 \
        const int _unitH = unitH;                                                                            \
        const int _unitW = unitW;                                                                            \
        return load##unitH##_##unitW##_stride_h((offCalcExpr), mw, h, mat);                                  \
    }                                                                                                        \
    inline float##unitH##_##unitW load##unitH##_##unitW##from_matrix_##layoutName(                           \
        SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col) {                              \
        return load##unitH##_##unitW##from_matrix_##layoutName##_with_offset(0, mat, mh, mw, row, col);      \
    }                                                                                                        \
    inline float##unitH##_##unitW load##unitH##_##unitW##from_trans_matrix_##layoutName##_with_offset(       \
        const int offset, SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col) {            \
        return load##unitH##_##unitW##from_matrix_##layoutName##_with_offset(offset, mat, mw, mh, col, row); \
    }                                                                                                        \
    inline float##unitH##_##unitW load##unitH##_##unitW##from_trans_matrix_##layoutName(                     \
        SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col) {                              \
        return load##unitH##_##unitW##from_trans_matrix_##layoutName##_with_offset(0, mat, mh, mw, row,      \
                                                                                   col);                     \
    }                                                                                                        \
    inline void store##unitH##_##unitW##to_matrix_##layoutName##_with_offset(                                \
        const int offset, SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col,              \
        float##unitH##_##unitW data) {                                                                       \
        const int _unitH = unitH;                                                                            \
        const int _unitW = unitW;                                                                            \
        return store##unitH##_##unitW##_stride(offset + (offCalcExpr), mw, mat, data);                       \
    }                                                                                                        \
    inline void store##unitH##_##unitW##to_matrix_##layoutName##_srch(                                       \
        SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col,                                \
        const int h, float##unitH##_##unitW data) {                                                          \
        const int _unitH = unitH;                                                                            \
        const int _unitW = unitW;                                                                            \
        return store##unitH##_##unitW##_stride_h((offCalcExpr), mw, h, mat, data);                           \
    }                                                                                                        \
    inline void store##unitH##_##unitW##to_matrix_##layoutName(                                              \
        SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col, float##unitH##_##unitW data) { \
        return store##unitH##_##unitW##to_matrix_##layoutName##_with_offset(0, mat, mh, mw, row, col, data); \
    }                                                                                                        \
    inline void store##unitH##_##unitW##to_trans_matrix_##layoutName##_with_offset(                          \
        const int offset, SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col,              \
        float##unitH##_##unitW data) {                                                                       \
        return store##unitH##_##unitW##to_matrix_##layoutName##_with_offset(offset, mat, mw, mh, col, row,   \
                                                                            data);                           \
    }                                                                                                        \
    inline void store##unitH##_##unitW##to_trans_matrix_##layoutName(                                        \
        SIM_X86::tensor mat, const int mh, const int mw, const int row, const int col, float##unitH##_##unitW data) { \
        return store##unitH##_##unitW##to_trans_matrix_##layoutName##_with_offset(0, mat, mh, mw, row, col,  \
                                                                                  data);                     \
    }

DEF_LOAD_STORE(128, 128, row *mw *_unitH *_unitW / 32 + col * _unitW / 32, line_layout)
DEF_LOAD_STORE(8, 128, row *mw *_unitH *_unitW / 32 + col * _unitW / 32, line_layout)
DEF_LOAD_STORE(128, 128, (row * mw + col) * _unitH * _unitW / 32, block_layout)
DEF_LOAD_STORE(8, 128, (row * mw + col) * _unitH * _unitW / 32, block_layout)

#undef DEF_LOAD_STORE

#endif
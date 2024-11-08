#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// #include "typehint.h"
#include "math.h"
#include "libdevice.h"

inline float8_128 $F(int8_128 a) {
    float8_128 result0;
    asm volatile("{V0@(pr0) %[res] = mov.u32 %[input];}" : [res] "=x"(result0) : [input] "x"(a) :);
    return result0;
}

inline int8_128 $S(float8_128 a) {
    int8_128 result0;
    asm volatile("{V0@(pr0) %[res] = mov.u32 %[input];}" : [res] "=x"(result0) : [input] "x"(a) :);
    return result0;
}

inline int8_128 subs(int8_128 a, int8_128 b) {
    int8_128 result0;
    asm volatile("{V0@(pr0) %[res] = sub.s32 %[a], %[b];}" : [res] "=x"(result0) : [a] "x"(a), [b] "x"(b) :);
    return result0;
}

inline int8_128 _S(int a) { return v_u32_move_i(a); }

inline float8_128 _F(float a) { return v_u32_move_f(a); }

inline int8_128 oru(int8_128 a, int8_128 b) {
    int8_128 result0;
    asm volatile("{V0@(pr0) %[res] = or.u32 %[a], %[b];}" : [res] "=x"(result0) : [a] "x"(a), [b] "x"(b) :);
    return result0;
}

inline int8_128 swap01(int8_128 a) { return v_u32_xor(a, _S(1)); }

/*修改为int*/
inline int8_128 maxu(int8_128 a, int8_128 b) {

    return __dlc_umax(a, b);
}

inline int8_128 minu(int8_128 a, int8_128 b) {
        return __dlc_umin(a, b);
}

// arguments are [value, permute reg, mti_select, mode]
// mode {normal = 0, sublanes = 1, bytes = 2}
inline int8_128 m_u32_perm(int8_128 a, int8_128 b, int c, int d) { return $S(m_f32_perm($F(a), b, c, d)); }

/*修改为int*/

inline int8_128 cmp_exchange_1_i(int8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();               // vr0
    int8_128 ex1 = swap01(cid);                 // vr1
    int8_128 nex1 = m_u32_perm(n, ex1, 0, 0); // vr1
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_s32_cmp(GT, n, nex1); // vmsk7
    int8_128 s = minu(n, nex1);            // vr2
    int8_128 g = maxu(n, nex1);            // vr3
    int8_128 idxs = v_s32_sel(cmp1, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp1, idx1, *idx);
    bool8_128 m = v_s32_cmp(NEQ, v_u32_and(cid, _S(1)), _S(1)); // vmsk7
    int8_128 cex1 = v_s32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline int8_128 cmp_exchange_1_i_stable(int8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();               // vr0
    int8_128 ex1 = swap01(cid);                 // vr1
    int8_128 nex1 = m_u32_perm(n, ex1, 0, 0); // vr1
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_s32_cmp(GT, n, nex1); // vmsk7
    bool8_128 cmp2 = v_s32_cmp(LSEQ, nex1, n);
    // (a, b) cmp (b, a), if a != b, so mask always has one true and one false
    // while if a == b, mask will be all true or false
    // so we choose n > nex1 in even pos, and nex1 <= n in odd pos
    // to pervert all true or all false
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, *idx, idx1);
    cmp1 = v_s32_cmp(EQ,
                     v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                               v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                     v_u32_move_i(1));
    int8_128 s = minu(n, nex1); // vr2
    int8_128 g = maxu(n, nex1); // vr3
    int8_128 idxs = v_s32_sel(cmp1, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp1, idx1, *idx);
    bool8_128 m = v_s32_cmp(NEQ, v_u32_and(cid, _S(1)), _S(1)); // vmsk7
    int8_128 cex1 = v_s32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

/*
    Prevent LLVM to optimize
*/
inline int8_128 shru(int8_128 a, int8_128 b) {
    int8_128 result0;
    asm volatile("{V0@(pr0) %[res] = shr.u32 %[a], %[b];}" : [res] "=x"(result0) : [a] "x"(a), [b] "x"(b) :);
    return result0;
}

inline int8_128 andu(int8_128 a, int8_128 b) {
    int8_128 result0;
    asm volatile("{V0@(pr0) %[res] = and.u32 %[a], %[b];}" : [res] "=x"(result0) : [a] "x"(a), [b] "x"(b) :);
    return result0;
}

/*修改为int*/
inline int8_128 cmp_exchange_p2_b_i(int p, int8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p - 1));
    int8_128 r = andu(cid, _S((1 << (p - 1)) - 1));
    int8_128 ex1 = oru(v_u32_shl(swap01(k), _S(p - 1)), r);
    int8_128 nex1 = m_u32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp = v_s32_cmp(GT, n, nex1);
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    int8_128 s = minu(n, nex1);
    int8_128 g = maxu(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << (p)) - 1)), _S(1 << (p - 1)));
    int8_128 cex1 = v_s32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline int8_128 cmp_exchange_p2_b_i_stable(int p, int8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p - 1));
    int8_128 r = andu(cid, _S((1 << (p - 1)) - 1));
    int8_128 ex1 = oru(v_u32_shl(swap01(k), _S(p - 1)), r);
    int8_128 nex1 = m_u32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_s32_cmp(GT, n, nex1);
    bool8_128 cmp2 = v_s32_cmp(LSEQ, nex1, n);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, *idx, idx1);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    int8_128 s = minu(n, nex1);
    int8_128 g = maxu(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << (p)) - 1)), _S(1 << (p - 1)));
    int8_128 cex1 = v_s32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

/*修改为int*/
inline int8_128 cmp_exchange_p2_i(int p, int8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p));
    int8_128 r = andu(cid, _S((1 << p) - 1));
    int8_128 ex1 = oru(v_u32_shl(k, _S(p)), subs(_S((1 << p) - 1), r));
    int8_128 nex1 = m_u32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp = v_s32_cmp(GT, n, nex1);
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    int8_128 s = minu(n, nex1);
    int8_128 g = maxu(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << p) - 1)), _S(1 << (p - 1)));
    int8_128 cex1 = v_s32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline int8_128 cmp_exchange_p2_i_stable(int p, int8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p));
    int8_128 r = andu(cid, _S((1 << p) - 1));
    int8_128 ex1 = oru(v_u32_shl(k, _S(p)), subs(_S((1 << p) - 1), r));
    int8_128 nex1 = m_u32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_s32_cmp(GT, n, nex1);
    bool8_128 cmp2 = v_s32_cmp(LSEQ, nex1, n);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, *idx, idx1);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    int8_128 s = minu(n, nex1);
    int8_128 g = maxu(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << p) - 1)), _S(1 << (p - 1)));
    int8_128 cex1 = v_s32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}


/*修改为int*/
inline int8_128 sort128_i(int8_128 n, int8_128 *idx) {
    n = cmp_exchange_1_i(n, idx);

    n = cmp_exchange_p2_i(2, n, idx);
    n = cmp_exchange_1_i(n, idx);

    n = cmp_exchange_p2_i(3, n, idx);
    n = cmp_exchange_p2_b_i(2, n, idx);
    n = cmp_exchange_1_i(n, idx);

    n = cmp_exchange_p2_i(4, n, idx);
    n = cmp_exchange_p2_b_i(3, n, idx);
    n = cmp_exchange_p2_b_i(2, n, idx);
    n = cmp_exchange_1_i(n, idx);

    n = cmp_exchange_p2_i(5, n, idx);
    n = cmp_exchange_p2_b_i(4, n, idx);
    n = cmp_exchange_p2_b_i(3, n, idx);
    n = cmp_exchange_p2_b_i(2, n, idx);
    n = cmp_exchange_1_i(n, idx);

    n = cmp_exchange_p2_i(6, n, idx);
    n = cmp_exchange_p2_b_i(5, n, idx);
    n = cmp_exchange_p2_b_i(4, n, idx);
    n = cmp_exchange_p2_b_i(3, n, idx);
    n = cmp_exchange_p2_b_i(2, n, idx);
    n = cmp_exchange_1_i(n, idx);

    n = cmp_exchange_p2_i(7, n, idx);
    n = cmp_exchange_p2_b_i(6, n, idx);
    n = cmp_exchange_p2_b_i(5, n, idx);
    n = cmp_exchange_p2_b_i(4, n, idx);
    n = cmp_exchange_p2_b_i(3, n, idx);
    n = cmp_exchange_p2_b_i(2, n, idx);
    n = cmp_exchange_1_i(n, idx);
    return n;
}
inline int8_128 sort128_i_stable(int8_128 n, int8_128 *idx) {
    n = cmp_exchange_1_i_stable(n, idx);

    n = cmp_exchange_p2_i_stable(2, n, idx);
    n = cmp_exchange_1_i_stable(n, idx);

    n = cmp_exchange_p2_i_stable(3, n, idx);
    n = cmp_exchange_p2_b_i_stable(2, n, idx);
    n = cmp_exchange_1_i_stable(n, idx);

    n = cmp_exchange_p2_i_stable(4, n, idx);
    n = cmp_exchange_p2_b_i_stable(3, n, idx);
    n = cmp_exchange_p2_b_i_stable(2, n, idx);
    n = cmp_exchange_1_i_stable(n, idx);

    n = cmp_exchange_p2_i_stable(5, n, idx);
    n = cmp_exchange_p2_b_i_stable(4, n, idx);
    n = cmp_exchange_p2_b_i_stable(3, n, idx);
    n = cmp_exchange_p2_b_i_stable(2, n, idx);
    n = cmp_exchange_1_i_stable(n, idx);

    n = cmp_exchange_p2_i_stable(6, n, idx);
    n = cmp_exchange_p2_b_i_stable(5, n, idx);
    n = cmp_exchange_p2_b_i_stable(4, n, idx);
    n = cmp_exchange_p2_b_i_stable(3, n, idx);
    n = cmp_exchange_p2_b_i_stable(2, n, idx);
    n = cmp_exchange_1_i_stable(n, idx);

    n = cmp_exchange_p2_i_stable(7, n, idx);
    n = cmp_exchange_p2_b_i_stable(6, n, idx);
    n = cmp_exchange_p2_b_i_stable(5, n, idx);
    n = cmp_exchange_p2_b_i_stable(4, n, idx);
    n = cmp_exchange_p2_b_i_stable(3, n, idx);
    n = cmp_exchange_p2_b_i_stable(2, n, idx);
    n = cmp_exchange_1_i_stable(n, idx);
    return n;
}


/*修改为int*/
inline int8_128 sort128_b_i(int8_128 n, int8_128 *idx) {
    n = cmp_exchange_p2_b_i(7, n, idx);
    n = cmp_exchange_p2_b_i(6, n, idx);
    n = cmp_exchange_p2_b_i(5, n, idx);
    n = cmp_exchange_p2_b_i(4, n, idx);
    n = cmp_exchange_p2_b_i(3, n, idx);
    n = cmp_exchange_p2_b_i(2, n, idx);
    n = cmp_exchange_1_i(n, idx);
    return n;
}
inline int8_128 sort128_b_i_stable(int8_128 n, int8_128 *idx) {
    n = cmp_exchange_p2_b_i_stable(7, n, idx);
    n = cmp_exchange_p2_b_i_stable(6, n, idx);
    n = cmp_exchange_p2_b_i_stable(5, n, idx);
    n = cmp_exchange_p2_b_i_stable(4, n, idx);
    n = cmp_exchange_p2_b_i_stable(3, n, idx);
    n = cmp_exchange_p2_b_i_stable(2, n, idx);
    n = cmp_exchange_1_i_stable(n, idx);
    return n;
}


/*修改为int*/
inline int8_128 re128_i(int8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 p = subs(_S(127), cid);
    *idx = m_u32_perm(*idx, p, 0, 0);
    return m_u32_perm(n, p, 0, 0);
}



inline int8_128 v_u32_rotate(int8_128 a, int b) { return $S(v_row_rotate($F(a), b)); }

/*修改为int*/
inline int8_128 sort256_i(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = sort128_i(n, idx);
    int8_128 i01234567 = *idx;
    int8_128 n12345670 =v_u32_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 n70123456 =v_u32_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n11335577 = v_s32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    int8_128 n00224466 = v_s32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    n11335577 = re128_i(n11335577, &i11335577);
    bool8_128 cmp = v_s32_cmp(GT, n11335577, n00224466);
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    int8_128 s = minu(n11335577, n00224466);
    int8_128 g = maxu(n11335577, n00224466);
    int8_128 r = v_s32_sel(mk, s, g);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i(r, idx);
}

inline int8_128 sort256_i_stable(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = sort128_i_stable(n, idx);
    int8_128 i01234567 = *idx;
    int8_128 n12345670 = v_u32_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 n70123456 = v_u32_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n11335577 = v_s32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    int8_128 n00224466 = v_s32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    n11335577 = re128_i(n11335577, &i11335577);
    bool8_128 cmp1 = v_s32_cmp(GT, n11335577, n00224466);
    bool8_128 cmp2 = v_s32_cmp(LSEQ, n00224466, n11335577);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i11335577, i00224466);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    int8_128 s = minu(n11335577, n00224466);
    int8_128 g = maxu(n11335577, n00224466);
    int8_128 r = v_s32_sel(mk, s, g);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i_stable(r, idx);
}

/*修改为int*/

inline int8_128 sort256_b_i(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    int8_128 n12345670 =v_u32_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 n70123456 =v_u32_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n11335577 = v_s32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    int8_128 n00224466 = v_s32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    bool8_128 cmp = v_s32_cmp(GT, n11335577, n00224466);
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    int8_128 s = minu(n11335577, n00224466);
    int8_128 g = maxu(n11335577, n00224466);
    int8_128 r = v_s32_sel(mk, s, g);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i(r, idx);
}

inline int8_128 sort256_b_i_stable(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    int8_128 n12345670 = v_u32_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 n70123456 = v_u32_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n11335577 = v_s32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    int8_128 n00224466 = v_s32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    bool8_128 cmp1 = v_s32_cmp(GT, n11335577, n00224466);
    bool8_128 cmp2 = v_s32_cmp(LSEQ, n00224466, n11335577);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i11335577, i00224466);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    int8_128 s = minu(n11335577, n00224466);
    int8_128 g = maxu(n11335577, n00224466);
    int8_128 r = v_s32_sel(mk, s, g);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i_stable(r, idx);
}


/*修改为int*/
inline int8_128 sort512_i(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = sort256_i(n, idx);
    int8_128 i01234567 = *idx;
    int8_128 n12345670 =v_u32_rotate(n01234567, 0);
    int8_128 n23456701 =v_u32_rotate(n12345670, 0);
    int8_128 n70123456 =v_u32_rotate(n01234567, 1);
    int8_128 n67012345 =v_u32_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    int8_128 n23236767 = v_s32_sel(mk, n01234567, n23456701);
    int8_128 n01014545 = v_s32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    int8_128 n32367672 =v_u32_rotate(n23236767, 0);
    int8_128 n72323676 =v_u32_rotate(n23236767, 1);
    int8_128 i32367672 = v_u32_rotate(i23236767, 0);
    int8_128 i72323676 = v_u32_rotate(i23236767, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n32327676 = v_s32_sel(mk2, n32367672, n72323676);
    int8_128 i32327676 = v_s32_sel(mk2, i32367672, i72323676);
    n32327676 = re128_i(n32327676, &i32327676);
    bool8_128 cmp = v_s32_cmp(GT, n01014545, n32327676);
    int8_128 is = v_s32_sel(cmp, i01014545, i32327676);
    int8_128 ig = v_s32_sel(cmp, i32327676, i01014545);
    int8_128 g = maxu(n01014545, n32327676);
    int8_128 s = minu(n01014545, n32327676);
    int8_128 r = v_s32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i(r, idx);
}

inline int8_128 sort512_i_stable(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = sort256_i_stable(n, idx);
    int8_128 i01234567 = *idx;
    int8_128 n12345670 = v_u32_rotate(n01234567, 0);
    int8_128 n23456701 = v_u32_rotate(n12345670, 0);
    int8_128 n70123456 = v_u32_rotate(n01234567, 1);
    int8_128 n67012345 = v_u32_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    int8_128 n23236767 = v_s32_sel(mk, n01234567, n23456701);
    int8_128 n01014545 = v_s32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    int8_128 n32367672 = v_u32_rotate(n23236767, 0);
    int8_128 n72323676 = v_u32_rotate(n23236767, 1);
    int8_128 i32367672 = v_u32_rotate(i23236767, 0);
    int8_128 i72323676 = v_u32_rotate(i23236767, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n32327676 = v_s32_sel(mk2, n32367672, n72323676);
    int8_128 i32327676 = v_s32_sel(mk2, i32367672, i72323676);
    n32327676 = re128_i(n32327676, &i32327676);
    bool8_128 cmp1 = v_s32_cmp(GT, n01014545, n32327676);
    bool8_128 cmp2 = v_s32_cmp(LSEQ, n32327676, n01014545);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i01014545, i32327676);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i01014545, i32327676);
    int8_128 ig = v_s32_sel(cmp, i32327676, i01014545);
    int8_128 g = maxu(n01014545, n32327676);
    int8_128 s = minu(n01014545, n32327676);
    int8_128 r = v_s32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i_stable(r, idx);
}


/*修改为int*/
inline int8_128 sort512_b_i(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    int8_128 n12345670 =v_u32_rotate(n01234567, 0);
    int8_128 n23456701 =v_u32_rotate(n12345670, 0);
    int8_128 n70123456 =v_u32_rotate(n01234567, 1);
    int8_128 n67012345 =v_u32_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    int8_128 n23236767 = v_s32_sel(mk, n01234567, n23456701);
    int8_128 n01014545 = v_s32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    bool8_128 cmp = v_s32_cmp(GT, n01014545, n23236767);
    int8_128 is = v_s32_sel(cmp, i01014545, i23236767);
    int8_128 ig = v_s32_sel(cmp, i23236767, i01014545);
    int8_128 g = maxu(n01014545, n23236767);
    int8_128 s = minu(n01014545, n23236767);
    int8_128 r = v_s32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i(r, idx);
}

inline int8_128 sort512_b_i_stable(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    int8_128 n12345670 = v_u32_rotate(n01234567, 0);
    int8_128 n23456701 = v_u32_rotate(n12345670, 0);
    int8_128 n70123456 = v_u32_rotate(n01234567, 1);
    int8_128 n67012345 = v_u32_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    int8_128 n23236767 = v_s32_sel(mk, n01234567, n23456701);
    int8_128 n01014545 = v_s32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    bool8_128 cmp1 = v_s32_cmp(GT, n01014545, n23236767);
    bool8_128 cmp2 = v_s32_cmp(LSEQ, n23236767, n01014545);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i01014545, i23236767);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i01014545, i23236767);
    int8_128 ig = v_s32_sel(cmp, i23236767, i01014545);
    int8_128 g = maxu(n01014545, n23236767);
    int8_128 s = minu(n01014545, n23236767);
    int8_128 r = v_s32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i_stable(r, idx);
}


/*修改为int*/
inline int8_128 sort1024_i(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = sort512_i(n, idx);
    int8_128 i01234567 = *idx;
    int8_128 n12345670 =v_u32_rotate(n01234567, 0);
    int8_128 n23456701 =v_u32_rotate(n12345670, 0);
    int8_128 n34567012 =v_u32_rotate(n23456701, 0);
    int8_128 n45670123 =v_u32_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    int8_128 n45674567 = v_s32_sel(mk, n01234567, n45670123);
    int8_128 n01230123 = v_s32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    int8_128 n56745674 =v_u32_rotate(n45674567, 0);
    int8_128 n74567456 =v_u32_rotate(n45674567, 1);
    int8_128 i56745674 = v_u32_rotate(i45674567, 0);
    int8_128 i74567456 = v_u32_rotate(i45674567, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n54765476 = v_s32_sel(mk2, n56745674, n74567456);
    int8_128 i54765476 = v_s32_sel(mk2, i56745674, i74567456);
    int8_128 n47654765 =v_u32_rotate(n54765476, 0);
    int8_128 n76547654 =v_u32_rotate(n47654765, 0);
    int8_128 i47654765 = v_u32_rotate(i54765476, 0);
    int8_128 i76547654 = v_u32_rotate(i47654765, 0);
    n76547654 = re128_i(n76547654, &i76547654);
    bool8_128 cmp = v_s32_cmp(GT, n76547654, n01230123);
    int8_128 is = v_s32_sel(cmp, i76547654, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i76547654);
    int8_128 g = maxu(n76547654, n01230123);
    int8_128 s = minu(n76547654, n01230123);
    int8_128 r = v_s32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i(r, idx);
}

inline int8_128 sort1024_i_stable(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = sort512_i_stable(n, idx);
    int8_128 i01234567 = *idx;
    int8_128 n12345670 = v_u32_rotate(n01234567, 0);
    int8_128 n23456701 = v_u32_rotate(n12345670, 0);
    int8_128 n34567012 = v_u32_rotate(n23456701, 0);
    int8_128 n45670123 = v_u32_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    int8_128 n45674567 = v_s32_sel(mk, n01234567, n45670123);
    int8_128 n01230123 = v_s32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    int8_128 n56745674 = v_u32_rotate(n45674567, 0);
    int8_128 n74567456 = v_u32_rotate(n45674567, 1);
    int8_128 i56745674 = v_u32_rotate(i45674567, 0);
    int8_128 i74567456 = v_u32_rotate(i45674567, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n54765476 = v_s32_sel(mk2, n56745674, n74567456);
    int8_128 i54765476 = v_s32_sel(mk2, i56745674, i74567456);
    int8_128 n47654765 = v_u32_rotate(n54765476, 0);
    int8_128 n76547654 = v_u32_rotate(n47654765, 0);
    int8_128 i47654765 = v_u32_rotate(i54765476, 0);
    int8_128 i76547654 = v_u32_rotate(i47654765, 0);
    n76547654 = re128_i(n76547654, &i76547654);
    bool8_128 cmp1 = v_s32_cmp(GT, n76547654, n01230123);
    bool8_128 cmp2 = v_s32_cmp(LSEQ, n01230123, n76547654);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i76547654, i01230123);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i76547654, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i76547654);
    int8_128 g = maxu(n76547654, n01230123);
    int8_128 s = minu(n76547654, n01230123);
    int8_128 r = v_s32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i_stable(r, idx);
}


/*修改为int*/
inline int8_128 sort1024_b_i(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    int8_128 n12345670 =v_u32_rotate(n01234567, 0);
    int8_128 n23456701 =v_u32_rotate(n12345670, 0);
    int8_128 n34567012 =v_u32_rotate(n23456701, 0);
    int8_128 n45670123 =v_u32_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    int8_128 n45674567 = v_s32_sel(mk, n01234567, n45670123);
    int8_128 n01230123 = v_s32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    bool8_128 cmp = v_s32_cmp(GT, n45674567, n01230123);
    int8_128 is = v_s32_sel(cmp, i45674567, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i45674567);
    int8_128 g = maxu(n45674567, n01230123);
    int8_128 s = minu(n45674567, n01230123);
    int8_128 r = v_s32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i(r, idx);
}

inline int8_128 sort1024_b_i_stable(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    int8_128 n12345670 = v_u32_rotate(n01234567, 0);
    int8_128 n23456701 = v_u32_rotate(n12345670, 0);
    int8_128 n34567012 = v_u32_rotate(n23456701, 0);
    int8_128 n45670123 = v_u32_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    int8_128 n45674567 = v_s32_sel(mk, n01234567, n45670123);
    int8_128 n01230123 = v_s32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    bool8_128 cmp1 = v_s32_cmp(GT, n45674567, n01230123);
    bool8_128 cmp2 = v_s32_cmp(LSEQ, n01230123, n45674567);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i45674567, i01230123);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i45674567, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i45674567);
    int8_128 g = maxu(n45674567, n01230123);
    int8_128 s = minu(n45674567, n01230123);
    int8_128 r = v_s32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i_stable(r, idx);
}


inline int nearx2(int n) {
    n = n - 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

inline int clz(int n) {
    int r;
    asm volatile("{ S0@(pr0) %[res] = count.u32 %[input]; }" : [res] "=r"(r) : [input] "r"(n) :);
    return r;
}

inline int lg2(int n) { return 31 - clz(n); }

/*修改为int*/
inline void cmp_exchange_1024_p0_i(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        int8_128 v = $S(v_f32_ld_tnsr_b(i / 32, d));
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_i(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, $F(v));
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}

inline void cmp_exchange_1024_p0_i_stable(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        int8_128 v =$S(v_f32_ld_tnsr_b(i / 32, d));
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_i_stable(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, $F(v));
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}


/*修改为int*/
inline void cmp_exchange_1024_pb0_i(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        int8_128 v = $S(v_f32_ld_tnsr_b(i / 32, d));
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_b_i(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, $F(v));
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}

inline void cmp_exchange_1024_pb0_i_stable(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        int8_128 v =$S(v_f32_ld_tnsr_b(i / 32, d));
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_b_i_stable(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, $F(v));
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}



/*修改为int*/
inline int8_128 re1024_i(int8_128 n, int8_128 *idx) {
    int8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    int8_128 n12345670 =v_u32_rotate(n01234567, 0);
    int8_128 n23456701 =v_u32_rotate(n12345670, 0);
    int8_128 n70123456 =v_u32_rotate(n01234567, 1);
    int8_128 n67012345 =v_u32_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    int8_128 n23016745 = v_s32_sel(mk, n67012345, n23456701);
    int8_128 i23016745 = v_s32_sel(mk, i67012345, i23456701);
    int8_128 n30167452 =v_u32_rotate(n23016745, 0);
    int8_128 n52301674 =v_u32_rotate(n23016745, 1);
    int8_128 i30167452 = v_u32_rotate(i23016745, 0);
    int8_128 i52301674 = v_u32_rotate(i23016745, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n32107654 = v_s32_sel(mk2, n30167452, n52301674);
    int8_128 i32107654 = v_s32_sel(mk2, i30167452, i52301674);
    int8_128 n21076543 =v_u32_rotate(n32107654, 0);
    int8_128 n10765432 =v_u32_rotate(n21076543, 0);
    int8_128 n07654321 =v_u32_rotate(n10765432, 0);
    int8_128 n76543210 =v_u32_rotate(n07654321, 0);
    int8_128 i21076543 = v_u32_rotate(i32107654, 0);
    int8_128 i10765432 = v_u32_rotate(i21076543, 0);
    int8_128 i07654321 = v_u32_rotate(i10765432, 0);
    int8_128 i76543210 = v_u32_rotate(i07654321, 0);
    *idx = i76543210;
    return re128_i(n76543210, idx);
}


inline void cmp_exchange_1024_p_i(int swapl, int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    // k * W + n -> swap01(k) * W + (W - 1 - n)
    int g = (lenp2 / 1024) >> swapl;
    int w = 1 << swapl;
    int hw = w / 2;
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < hw; j++) {
            int laddr = (i * 2) * hw + j;
            int raddr = (i * 2 + 1) * hw + (hw - 1 - j);
            int8_128 l = $S(v_f32_ld_tnsr_b(laddr * 32, d));
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            int8_128 r = $S(v_f32_ld_tnsr_b(raddr * 32, d));
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            r = re1024_i(r, &rx);
            bool8_128 cmp = v_s32_cmp(GT, l, r);
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, $F(minu(l, r)));
            v_f32_st_tnsr_b(raddr * 32, d, $F(maxu(l, r)));
            v_f32_st_tnsr_b(laddr * 32, idx, $F(is));
            v_f32_st_tnsr_b(raddr * 32, idx, $F(ig));
        }
    }
}

inline void cmp_exchange_1024_p_i_stable(int swapl, int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    // k * W + n -> swap01(k) * W + (W - 1 - n)
    int g = (lenp2 / 1024) >> swapl;
    int w = 1 << swapl;
    int hw = w / 2;
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < hw; j++) {
            int laddr = (i * 2) * hw + j;
            int raddr = (i * 2 + 1) * hw + (hw - 1 - j);
            int8_128 l =$S(v_f32_ld_tnsr_b(laddr * 32, d));
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            int8_128 r =$S(v_f32_ld_tnsr_b(raddr * 32, d));
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            r = re1024_i(r, &rx);
            bool8_128 cmp1 = v_s32_cmp(GT, l, r);
            bool8_128 cmp2 = v_s32_cmp(LSEQ, r, l);
            bool8_128 cmp3 = cmp1 ^ cmp2;
            bool8_128 cmp4 = v_s32_cmp(GT, lx, rx);
            bool8_128 cmp = v_s32_cmp(EQ,
                                      v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                                v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                                      v_u32_move_i(1));
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, $F(minu(l, r)));
            v_f32_st_tnsr_b(raddr * 32, d, $F(maxu(l, r)));
            v_f32_st_tnsr_b(laddr * 32, idx, $F(is));
            v_f32_st_tnsr_b(raddr * 32, idx, $F(ig));
        }
    }
}


inline void cmp_exchange_1024_pb_i(int swapl, int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    // k * W + n -> swap01(k) * W + n
    int g = (lenp2 / 1024) >> swapl;
    int w = 1 << swapl;
    int hw = w / 2;
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < hw; j++) {
            int laddr = (i * 2) * hw + j;
            int raddr = (i * 2 + 1) * hw + j;
            int8_128 l = $S(v_f32_ld_tnsr_b(laddr * 32, d));
            int8_128 r = $S(v_f32_ld_tnsr_b(raddr * 32, d));
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            bool8_128 cmp = v_s32_cmp(GT, l, r);
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, $F(minu(l, r)));
            v_f32_st_tnsr_b(raddr * 32, d, $F(maxu(l, r)));
            v_f32_st_tnsr_b(laddr * 32, idx, $F(is));
            v_f32_st_tnsr_b(raddr * 32, idx, $F(ig));
        }
    }
}

inline void cmp_exchange_1024_pb_i_stable(int swapl, int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    // k * W + n -> swap01(k) * W + n
    int g = (lenp2 / 1024) >> swapl;
    int w = 1 << swapl;
    int hw = w / 2;
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < hw; j++) {
            int laddr = (i * 2) * hw + j;
            int raddr = (i * 2 + 1) * hw + j;
            int8_128 l =$S(v_f32_ld_tnsr_b(laddr * 32, d));
            int8_128 r =$S(v_f32_ld_tnsr_b(raddr * 32, d));
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            bool8_128 cmp1 = v_s32_cmp(GT, l, r);
            bool8_128 cmp2 = v_s32_cmp(LSEQ, r, l);
            bool8_128 cmp3 = cmp1 ^ cmp2;
            bool8_128 cmp4 = v_s32_cmp(GT, lx, rx);
            bool8_128 cmp = v_s32_cmp(EQ,
                                      v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                                v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                                      v_u32_move_i(1));
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, $F(minu(l, r)));
            v_f32_st_tnsr_b(raddr * 32, d, $F(maxu(l, r)));
            v_f32_st_tnsr_b(laddr * 32, idx, $F(is));
            v_f32_st_tnsr_b(raddr * 32, idx, $F(ig));
        }
    }
}




inline void sort_i(SIM_X86::tensor d, SIM_X86::tensor idx, int lenp2) {
    int lg = lg2(lenp2 / 1024);

    cmp_exchange_1024_p0_i(lenp2, d, idx); // 0

    for (int i = 1; i <= lg; i++) {
        cmp_exchange_1024_p_i(i, lenp2, d, idx); // 1
        for (int j = i - 1; j > 0; j--) {
            cmp_exchange_1024_pb_i(j, lenp2, d, idx);
        }
        cmp_exchange_1024_pb0_i(lenp2, d, idx);
    }
}

inline void sort_i_stable(SIM_X86::tensor d, SIM_X86::tensor idx, int lenp2) {
    int lg = lg2(lenp2 / 1024);

    cmp_exchange_1024_p0_i_stable(lenp2, d, idx); // 0

    for (int i = 1; i <= lg; i++) {
        cmp_exchange_1024_p_i_stable(i, lenp2, d, idx); // 1
        for (int j = i - 1; j > 0; j--) {
            cmp_exchange_1024_pb_i_stable(j, lenp2, d, idx);
        }
        cmp_exchange_1024_pb0_i_stable(lenp2, d, idx);
    }
}


inline int8_128 re128(int8_128 n) {
    int8_128 cid = get_core_id();
    return m_u32_perm(n, subs(_S(127), cid), 0, 0);
}

inline int8_128 re1024(int8_128 n) {
    int8_128 n01234567 = n;
    int8_128 n12345670 =v_u32_rotate(n01234567, 0);
    int8_128 n23456701 =v_u32_rotate(n12345670, 0);
    int8_128 n70123456 =v_u32_rotate(n01234567, 1);
    int8_128 n67012345 =v_u32_rotate(n70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    int8_128 n23016745 = v_s32_sel(mk, n67012345, n23456701);
    int8_128 n30167452 =v_u32_rotate(n23016745, 0);
    int8_128 n52301674 =v_u32_rotate(n23016745, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    int8_128 n32107654 = v_s32_sel(mk2, n30167452, n52301674);
    int8_128 n21076543 =v_u32_rotate(n32107654, 0);
    int8_128 n10765432 =v_u32_rotate(n21076543, 0);
    int8_128 n07654321 =v_u32_rotate(n10765432, 0);
    int8_128 n76543210 =v_u32_rotate(n07654321, 0);
    return re128(n76543210);
}
inline void sort_line_i(SIM_X86::tensor a, SIM_X86::tensor tmp, SIM_X86::tensor tmpidx, SIM_X86::tensor b, SIM_X86::tensor idx, int len, bool ascending) {
    int8_128 padding_v = ascending ? _S(0x7fffffff) : _S(-2147483648);
    int l2 = max(1024, nearx2(len));
    int fcpyl = len & 0xfffffc00;
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, v_f32_ld_tnsr_b(i / 32, a));
    }
    int rest = len - fcpyl;
    int mk = (1 << ((rest + 127) / 128)) - 1;
    if (rest > 0) {
        int8_128 v = $S(v_f32_ld_tnsr_st_msk(fcpyl / 32, a, 1, mk));
        bool8_128 m = v_s32_cmp(LS, get_core_id(), _S(rest));
        v = v_s32_sel(m, padding_v, v);
        v_f32_st_tnsr_b(fcpyl / 32, tmp, $F(v));
    }
    int pad = (len + 1023) & 0xfffffc00;
    for (int i = pad; i < l2; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, $F(padding_v));
    }
    for (int i = 0; i < l2; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmpidx, $F(v_s32_add(get_core_id(), _S(i))));
    }
    sort_i(tmp, tmpidx, l2);
    if (!ascending) {
        int l21024 = l2 / 1024;
        for (int i = 0; i < l21024 / 2; i++) {
            int laddr = i;
            int raddr = l21024 - 1 - i;
            int8_128 l = $S(v_f32_ld_tnsr_b(laddr * 32, tmp));
            int8_128 r = $S(v_f32_ld_tnsr_b(raddr * 32, tmp));
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, tmpidx));
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, tmpidx));
            l = re1024(l);
            r = re1024(r);
            lx = re1024(lx);
            rx = re1024(rx);
            v_f32_st_tnsr_b(laddr * 32, tmp, $F(r));
            v_f32_st_tnsr_b(raddr * 32, tmp, $F(l));
            v_f32_st_tnsr_b(laddr * 32, tmpidx, $F(rx));
            v_f32_st_tnsr_b(raddr * 32, tmpidx, $F(lx));
        }
        if (l21024 == 1) {
            int8_128 l = $S(v_f32_ld_tnsr_b(0, tmp));
            int8_128 lx = $S(v_f32_ld_tnsr_b(0, tmpidx));
            l = re1024(l);
            lx = re1024(lx);
            v_f32_st_tnsr_b(0, tmp, $F(l));
            v_f32_st_tnsr_b(0, tmpidx, $F(lx));
        }
    }
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, b, v_f32_ld_tnsr_b(i / 32, tmp));
        v_f32_st_tnsr_b(i / 32, idx, v_f32_ld_tnsr_b(i / 32, tmpidx));
    }
    if (rest > 0) {
        int8_128 v = $S(v_f32_ld_tnsr_b(fcpyl / 32, tmp));
        v_f32_st_tnsr_st_msk(fcpyl / 32, b, 1, mk, $F(v));
        int8_128 vi = $S(v_f32_ld_tnsr_b(fcpyl / 32, tmpidx));
        v_f32_st_tnsr_st_msk(fcpyl / 32, idx, 1, mk, $F(vi));
    }
}
inline void sort_line_i_stable(SIM_X86::tensor a, SIM_X86::tensor tmp, SIM_X86::tensor tmpidx, SIM_X86::tensor b, SIM_X86::tensor idx, int len,
                               bool ascending) {
    // int8_128 padding_v = ascending ? _F(3.40282e+38f) : _F(-3.40282e+38f);
    int8_128 padding_v = ascending ? _S(0x7fffffff) : _S(-2147483648);

    int l2 = max(1024, nearx2(len));
    int fcpyl = len & 0xfffffc00;
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, v_f32_ld_tnsr_b(i / 32, a));
    }
    int rest = len - fcpyl;
    int mk = (1 << ((rest + 127) / 128)) - 1;
    if (rest > 0) {
        int8_128 v = $S(v_f32_ld_tnsr_st_msk(fcpyl / 32, a, 1, mk));
        bool8_128 m = v_s32_cmp(LS, get_core_id(), _S(rest));
        v = v_s32_sel(m, padding_v, v);
        v_f32_st_tnsr_b(fcpyl / 32, tmp, $F(v));
    }
    int pad = (len + 1023) & 0xfffffc00;
    for (int i = pad; i < l2; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, $F(padding_v));
    }
    for (int i = 0; i < l2; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmpidx, $F(v_s32_add(get_core_id(), _S(i))));
    }
    sort_i_stable(tmp, tmpidx, l2);
    if (!ascending) {
        int l21024 = l2 / 1024;
        for (int i = 0; i < l21024 / 2; i++) {
            int laddr = i;
            int raddr = l21024 - 1 - i;
            int8_128 l =$S(v_f32_ld_tnsr_b(laddr * 32, tmp));
            int8_128 r =$S(v_f32_ld_tnsr_b(raddr * 32, tmp));
            int8_128 lx =$S(v_f32_ld_tnsr_b(laddr * 32, tmpidx));
            int8_128 rx =$S(v_f32_ld_tnsr_b(raddr * 32, tmpidx));
            l = re1024(l);
            r = re1024(r);
            lx = re1024(lx);
            rx = re1024(rx);
            v_f32_st_tnsr_b(laddr * 32, tmp,$F(r));
            v_f32_st_tnsr_b(raddr * 32, tmp, $F(l));
            v_f32_st_tnsr_b(laddr * 32, tmpidx, $F(rx));
            v_f32_st_tnsr_b(raddr * 32, tmpidx, $F(lx));
        }
        if (l21024 == 1) {
            int8_128 l =$S(v_f32_ld_tnsr_b(0, tmp));
            int8_128 lx =$S(v_f32_ld_tnsr_b(0, tmpidx));
            l = re1024(l);
            lx = re1024(lx);
            v_f32_st_tnsr_b(0, tmp, $F(l));
            v_f32_st_tnsr_b(0, tmpidx, $F(lx));
        }
    }
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, b, v_f32_ld_tnsr_b(i / 32, tmp));
        v_f32_st_tnsr_b(i / 32, idx, v_f32_ld_tnsr_b(i / 32, tmpidx));
    }
    if (rest > 0) {
        int8_128 v =$S(v_f32_ld_tnsr_b(fcpyl / 32, tmp));
        v_f32_st_tnsr_st_msk(fcpyl / 32, b, 1, mk, $F(v));
        int8_128 vi =$S(v_f32_ld_tnsr_b(fcpyl / 32, tmpidx));
        v_f32_st_tnsr_st_msk(fcpyl / 32, idx, 1, mk, $F(vi));
    }
}


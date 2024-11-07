#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

// FIXME: desend sort is just reverse the result of ascend sort


#include "math.h"

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

inline int8_128 subs(int8_128 a, int8_128 b) { return a - b; }

inline int8_128 _S(int a) { return v_u32_move_i(a); }

inline float8_128 _F(float a) { return v_u32_move_f(a); }

inline int8_128 oru(int8_128 a, int8_128 b) { return a | b; }

inline int8_128 swap01(int8_128 a) { return v_u32_xor(a, _S(1)); }

inline float8_128 maxf(float8_128 a, float8_128 b) {
    float8_128 result0;
    asm volatile("{V0@(pr0) %[res] = max.f32 %[a], %[b];}" : [res] "=x"(result0) : [a] "x"(a), [b] "x"(b) :);
    return result0;
}

inline float8_128 minf(float8_128 a, float8_128 b) {
    float8_128 result0;
    asm volatile("{V0@(pr0) %[res] = min.f32 %[a], %[b];}" : [res] "=x"(result0) : [a] "x"(a), [b] "x"(b) :);
    return result0;
}

// arguments are [value, permute reg, mti_select, mode]
// mode {normal = 0, sublanes = 1, bytes = 2}
inline int8_128 m_u32_perm(int8_128 a, int8_128 b, int c, int d) { return $S(m_f32_perm($F(a), b, c, d)); }

inline float8_128 cmp_exchange_1(float8_128 n) {
    float8_128 res;
    asm volatile("{ V0@(pr0) vr0 = vcoreid; }"
                 "{ V0@(pr0) vr1 = or.u32 vr0, r48; V1@(pr0) vr2 = and.u32 vr0, r48; }"
                 "{ V0@(pr0) vr1 = xor.u32 vr1, vr2; }"
                 "{ MTI@(pr0) pcr<0> = setperm vr1; }"
                 "{ MTI@(pr0) trf<0> = pmt %[n]; }"
                 "{ MTR@(pr0) vr1 = pop trf<0>; }"
                 "{ V0@(pr0) vr2 = min.f32 %[n], vr1; V1@(pr0) vr3 = max.f32 vr1, %[n]; }"
                 "{ V0@(pr0) vr1 = and.u32 vr0, r48; }"
                 "{ V0@(pr0) vmsk7 = neq.s32 vr1, r48; }"
                 "{ V0@(pr0) %[res] = sel vmsk7 vr3, vr2; }"
                 : [res] "=x"(res)
                 : [n] "x"(n)
                 : "vr0", "vr1", "vr2", "vr3", "vmsk7");
    return res;
    // int8_128 cid = get_core_id(); // vr0
    // int8_128 ex1 = swap01(cid); // vr1
    // float8_128 nex1 = m_f32_perm(n, ex1, 0, 0); // vr1
    // bool8_128 cmp1 = v_f32_cmp(GT, n, nex1); // vmsk7
    // float8_128 s = v_f32_sel(cmp1, n, nex1); // vr2
    // float8_128 g = v_f32_sel(cmp1, nex1, n); // vr3
    // bool8_128 m = v_s32_cmp(NEQ, v_u32_and(cid, _S(1)), _S(1)); // vmsk7
    // float8_128 cex1 = v_f32_sel(m, g, s);
    // return cex1;
}

inline float8_128 cmp_exchange_1_i(float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();               // vr0
    int8_128 ex1 = swap01(cid);                 // vr1
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0); // vr1
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_f32_cmp(GT, n, nex1); // vmsk7
    float8_128 s = minf(n, nex1);            // vr2
    float8_128 g = maxf(n, nex1);            // vr3
    int8_128 idxs = v_s32_sel(cmp1, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp1, idx1, *idx);
    bool8_128 m = v_s32_cmp(NEQ, v_u32_and(cid, _S(1)), _S(1)); // vmsk7
    float8_128 cex1 = v_f32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline float8_128 cmp_exchange_1_i_stable(float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();               // vr0
    int8_128 ex1 = swap01(cid);                 // vr1
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0); // vr1
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_f32_cmp(GT, n, nex1); // vmsk7
    bool8_128 cmp2 = v_f32_cmp(LSEQ, nex1, n);
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
    float8_128 s = minf(n, nex1); // vr2
    float8_128 g = maxf(n, nex1); // vr3
    int8_128 idxs = v_s32_sel(cmp1, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp1, idx1, *idx);
    bool8_128 m = v_s32_cmp(NEQ, v_u32_and(cid, _S(1)), _S(1)); // vmsk7
    float8_128 cex1 = v_f32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline float8_128 cmp_exchange_1_i_stable_mode0(float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();               // vr0
    int8_128 ex1 = swap01(cid);                 // vr1
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0); // vr1
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_f32_cmp(LS, n, nex1); // vmsk7
    bool8_128 cmp2 = v_f32_cmp(GTEQ, nex1, n);
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
    float8_128 s = minf(n, nex1); // vr2
    float8_128 g = maxf(n, nex1); // vr3
    int8_128 idxs = v_s32_sel(cmp1, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp1, idx1, *idx);
    bool8_128 m = v_s32_cmp(NEQ, v_u32_and(cid, _S(1)), _S(1)); // vmsk7
    float8_128 cex1 = v_f32_sel(m, s, g);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

/*
    Prevent LLVM to optimize
*/
inline int8_128 shru(int8_128 a, int8_128 b) { return a >> b; }

inline int8_128 andu(int8_128 a, int8_128 b) { return a & b; }

inline float8_128 cmp_exchange_p2_b(int p, float8_128 n) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p - 1));
    int8_128 r = andu(cid, _S((1 << (p - 1)) - 1));
    int8_128 ex1 = oru(v_u32_shl(swap01(k), _S(p - 1)), r);
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0);
    float8_128 s = minf(n, nex1);
    float8_128 g = maxf(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << (p)) - 1)), _S(1 << (p - 1)));
    float8_128 cex1 = v_f32_sel(m, g, s);
    return cex1;
}

inline float8_128 cmp_exchange_p2_b_i(int p, float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p - 1));
    int8_128 r = andu(cid, _S((1 << (p - 1)) - 1));
    int8_128 ex1 = oru(v_u32_shl(swap01(k), _S(p - 1)), r);
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp = v_f32_cmp(GT, n, nex1);
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    float8_128 s = minf(n, nex1);
    float8_128 g = maxf(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << (p)) - 1)), _S(1 << (p - 1)));
    float8_128 cex1 = v_f32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline float8_128 cmp_exchange_p2_b_i_stable(int p, float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p - 1));
    int8_128 r = andu(cid, _S((1 << (p - 1)) - 1));
    int8_128 ex1 = oru(v_u32_shl(swap01(k), _S(p - 1)), r);
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_f32_cmp(GT, n, nex1);
    bool8_128 cmp2 = v_f32_cmp(LSEQ, nex1, n);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, *idx, idx1);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    float8_128 s = minf(n, nex1);
    float8_128 g = maxf(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << (p)) - 1)), _S(1 << (p - 1)));
    float8_128 cex1 = v_f32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline float8_128 cmp_exchange_p2_b_i_stable_mode0(int p, float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p - 1));
    int8_128 r = andu(cid, _S((1 << (p - 1)) - 1));
    int8_128 ex1 = oru(v_u32_shl(swap01(k), _S(p - 1)), r);
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_f32_cmp(LS, n, nex1);
    bool8_128 cmp2 = v_f32_cmp(GTEQ, nex1, n);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, *idx, idx1);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    float8_128 s = minf(n, nex1);
    float8_128 g = maxf(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << (p)) - 1)), _S(1 << (p - 1)));
    float8_128 cex1 = v_f32_sel(m, s, g);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline float8_128 cmp_exchange_p2(int p, float8_128 n) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p));
    int8_128 r = andu(cid, _S((1 << p) - 1));
    int8_128 ex1 = oru(v_u32_shl(k, _S(p)), subs(_S((1 << p) - 1), r));
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0);
    float8_128 s = minf(n, nex1);
    float8_128 g = maxf(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << p) - 1)), _S(1 << (p - 1)));
    float8_128 cex1 = v_f32_sel(m, g, s);
    return cex1;
}

inline float8_128 cmp_exchange_p2_i(int p, float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p));
    int8_128 r = andu(cid, _S((1 << p) - 1));
    int8_128 ex1 = oru(v_u32_shl(k, _S(p)), subs(_S((1 << p) - 1), r));
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp = v_f32_cmp(GT, n, nex1);
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    float8_128 s = minf(n, nex1);
    float8_128 g = maxf(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << p) - 1)), _S(1 << (p - 1)));
    float8_128 cex1 = v_f32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline float8_128 cmp_exchange_p2_i_stable(int p, float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p));
    int8_128 r = andu(cid, _S((1 << p) - 1));
    int8_128 ex1 = oru(v_u32_shl(k, _S(p)), subs(_S((1 << p) - 1), r));
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_f32_cmp(GT, n, nex1);
    bool8_128 cmp2 = v_f32_cmp(LSEQ, nex1, n);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, *idx, idx1);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    float8_128 s = minf(n, nex1);
    float8_128 g = maxf(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << p) - 1)), _S(1 << (p - 1)));
    float8_128 cex1 = v_f32_sel(m, g, s);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline float8_128 cmp_exchange_p2_i_stable_mode0(int p, float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 k = shru(cid, _S(p));
    int8_128 r = andu(cid, _S((1 << p) - 1));
    int8_128 ex1 = oru(v_u32_shl(k, _S(p)), subs(_S((1 << p) - 1), r));
    float8_128 nex1 = m_f32_perm(n, ex1, 0, 0);
    int8_128 idx1 = m_u32_perm(*idx, ex1, 0, 0);
    bool8_128 cmp1 = v_f32_cmp(LS, n, nex1);
    bool8_128 cmp2 = v_f32_cmp(GTEQ, nex1, n);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, *idx, idx1);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 idxs = v_s32_sel(cmp, *idx, idx1);
    int8_128 idxg = v_s32_sel(cmp, idx1, *idx);
    float8_128 s = minf(n, nex1);
    float8_128 g = maxf(n, nex1);
    bool8_128 m = v_s32_cmp(LS, andu(cid, _S((1 << p) - 1)), _S(1 << (p - 1)));
    float8_128 cex1 = v_f32_sel(m, s, g);
    *idx = v_s32_sel(m, idxg, idxs);
    return cex1;
}

inline float8_128 sort128(float8_128 n) {
    n = cmp_exchange_1(n);

    n = cmp_exchange_p2(2, n);
    n = cmp_exchange_1(n);

    n = cmp_exchange_p2(3, n);
    n = cmp_exchange_p2_b(2, n);
    n = cmp_exchange_1(n);

    n = cmp_exchange_p2(4, n);
    n = cmp_exchange_p2_b(3, n);
    n = cmp_exchange_p2_b(2, n);
    n = cmp_exchange_1(n);

    n = cmp_exchange_p2(5, n);
    n = cmp_exchange_p2_b(4, n);
    n = cmp_exchange_p2_b(3, n);
    n = cmp_exchange_p2_b(2, n);
    n = cmp_exchange_1(n);

    n = cmp_exchange_p2(6, n);
    n = cmp_exchange_p2_b(5, n);
    n = cmp_exchange_p2_b(4, n);
    n = cmp_exchange_p2_b(3, n);
    n = cmp_exchange_p2_b(2, n);
    n = cmp_exchange_1(n);

    n = cmp_exchange_p2(7, n);
    n = cmp_exchange_p2_b(6, n);
    n = cmp_exchange_p2_b(5, n);
    n = cmp_exchange_p2_b(4, n);
    n = cmp_exchange_p2_b(3, n);
    n = cmp_exchange_p2_b(2, n);
    n = cmp_exchange_1(n);
    return n;
}

inline float8_128 sort128_i(float8_128 n, int8_128 *idx) {
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

inline float8_128 sort128_i_stable(float8_128 n, int8_128 *idx) {
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

inline float8_128 sort128_i_stable_mode0(float8_128 n, int8_128 *idx) {
    n = cmp_exchange_1_i_stable_mode0(n, idx);

    n = cmp_exchange_p2_i_stable_mode0(2, n, idx);
    n = cmp_exchange_1_i_stable_mode0(n, idx);

    n = cmp_exchange_p2_i_stable_mode0(3, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(2, n, idx);
    n = cmp_exchange_1_i_stable_mode0(n, idx);

    n = cmp_exchange_p2_i_stable_mode0(4, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(3, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(2, n, idx);
    n = cmp_exchange_1_i_stable_mode0(n, idx);

    n = cmp_exchange_p2_i_stable_mode0(5, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(4, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(3, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(2, n, idx);
    n = cmp_exchange_1_i_stable_mode0(n, idx);

    n = cmp_exchange_p2_i_stable_mode0(6, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(5, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(4, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(3, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(2, n, idx);
    n = cmp_exchange_1_i_stable_mode0(n, idx);

    n = cmp_exchange_p2_i_stable_mode0(7, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(6, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(5, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(4, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(3, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(2, n, idx);
    n = cmp_exchange_1_i_stable_mode0(n, idx);
    return n;
}

inline float8_128 sort128_b(float8_128 n) {
    n = cmp_exchange_p2_b(7, n);
    n = cmp_exchange_p2_b(6, n);
    n = cmp_exchange_p2_b(5, n);
    n = cmp_exchange_p2_b(4, n);
    n = cmp_exchange_p2_b(3, n);
    n = cmp_exchange_p2_b(2, n);
    n = cmp_exchange_1(n);
    return n;
}

inline float8_128 sort128_b_i(float8_128 n, int8_128 *idx) {
    n = cmp_exchange_p2_b_i(7, n, idx);
    n = cmp_exchange_p2_b_i(6, n, idx);
    n = cmp_exchange_p2_b_i(5, n, idx);
    n = cmp_exchange_p2_b_i(4, n, idx);
    n = cmp_exchange_p2_b_i(3, n, idx);
    n = cmp_exchange_p2_b_i(2, n, idx);
    n = cmp_exchange_1_i(n, idx);
    return n;
}

inline float8_128 sort128_b_i_stable(float8_128 n, int8_128 *idx) {
    n = cmp_exchange_p2_b_i_stable(7, n, idx);
    n = cmp_exchange_p2_b_i_stable(6, n, idx);
    n = cmp_exchange_p2_b_i_stable(5, n, idx);
    n = cmp_exchange_p2_b_i_stable(4, n, idx);
    n = cmp_exchange_p2_b_i_stable(3, n, idx);
    n = cmp_exchange_p2_b_i_stable(2, n, idx);
    n = cmp_exchange_1_i_stable(n, idx);
    return n;
}

inline float8_128 sort128_b_i_stable_mode0(float8_128 n, int8_128 *idx) {

    n = cmp_exchange_p2_b_i_stable_mode0(7, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(6, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(5, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(4, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(3, n, idx);
    n = cmp_exchange_p2_b_i_stable_mode0(2, n, idx);
    n = cmp_exchange_1_i_stable_mode0(n, idx);
    return n;
}

inline float8_128 re128(float8_128 n) {
    int8_128 cid = get_core_id();
    return m_f32_perm(n, subs(_S(127), cid), 0, 0);
}

inline float8_128 re128_i(float8_128 n, int8_128 *idx) {
    int8_128 cid = get_core_id();
    int8_128 p = subs(_S(127), cid);
    *idx = m_u32_perm(*idx, p, 0, 0);
    return m_f32_perm(n, p, 0, 0);
}

inline float8_128 sort256(float8_128 n) {
    float8_128 n01234567 = sort128(n);
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n11335577 = v_f32_sel(mk, n12345670, n01234567);
    float8_128 n00224466 = v_f32_sel(mk, n01234567, n70123456);
    n11335577 = re128(n11335577);
    float8_128 s = minf(n11335577, n00224466);
    float8_128 g = maxf(n11335577, n00224466);
    float8_128 r = v_f32_sel(mk, s, g);
    return sort128_b(r);
}

inline int8_128 v_u32_rotate(int8_128 a, int b) { return $S(v_row_rotate($F(a), b)); }

inline float8_128 sort256_i(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort128_i(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n11335577 = v_f32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    float8_128 n00224466 = v_f32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    n11335577 = re128_i(n11335577, &i11335577);
    bool8_128 cmp = v_f32_cmp(GT, n11335577, n00224466);
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    float8_128 s = minf(n11335577, n00224466);
    float8_128 g = maxf(n11335577, n00224466);
    float8_128 r = v_f32_sel(mk, s, g);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i(r, idx);
}

inline float8_128 sort256_i_stable(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort128_i_stable(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n11335577 = v_f32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    float8_128 n00224466 = v_f32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    n11335577 = re128_i(n11335577, &i11335577);
    bool8_128 cmp1 = v_f32_cmp(GT, n11335577, n00224466);
    bool8_128 cmp2 = v_f32_cmp(LSEQ, n00224466, n11335577);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i11335577, i00224466);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    float8_128 s = minf(n11335577, n00224466);
    float8_128 g = maxf(n11335577, n00224466);
    float8_128 r = v_f32_sel(mk, s, g);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i_stable(r, idx);
}

inline float8_128 sort256_i_stable_mode0(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort128_i_stable_mode0(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n11335577 = v_f32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    float8_128 n00224466 = v_f32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    n11335577 = re128_i(n11335577, &i11335577);
    bool8_128 cmp1 = v_f32_cmp(LS, n11335577, n00224466);
    bool8_128 cmp2 = v_f32_cmp(GTEQ, n00224466, n11335577);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i11335577, i00224466);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    float8_128 s = minf(n11335577, n00224466);
    float8_128 g = maxf(n11335577, n00224466);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i_stable_mode0(r, idx);
}

inline float8_128 sort256_b(float8_128 n) {
    float8_128 n01234567 = n;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n11335577 = v_f32_sel(mk, n12345670, n01234567);
    float8_128 n00224466 = v_f32_sel(mk, n01234567, n70123456);
    float8_128 s = minf(n11335577, n00224466);
    float8_128 g = maxf(n11335577, n00224466);
    float8_128 r = v_f32_sel(mk, s, g);
    return sort128_b(r);
}

inline float8_128 sort256_b_i(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n11335577 = v_f32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    float8_128 n00224466 = v_f32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    bool8_128 cmp = v_f32_cmp(GT, n11335577, n00224466);
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    float8_128 s = minf(n11335577, n00224466);
    float8_128 g = maxf(n11335577, n00224466);
    float8_128 r = v_f32_sel(mk, s, g);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i(r, idx);
}

inline float8_128 sort256_b_i_stable(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n11335577 = v_f32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    float8_128 n00224466 = v_f32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    bool8_128 cmp1 = v_f32_cmp(GT, n11335577, n00224466);
    bool8_128 cmp2 = v_f32_cmp(LSEQ, n00224466, n11335577);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i11335577, i00224466);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    float8_128 s = minf(n11335577, n00224466);
    float8_128 g = maxf(n11335577, n00224466);
    float8_128 r = v_f32_sel(mk, s, g);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i_stable(r, idx);
}

inline float8_128 sort256_b_i_stable_mode0(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    bool8_128 mk = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n11335577 = v_f32_sel(mk, n12345670, n01234567);
    int8_128 i11335577 = v_s32_sel(mk, i12345670, i01234567);
    float8_128 n00224466 = v_f32_sel(mk, n01234567, n70123456);
    int8_128 i00224466 = v_s32_sel(mk, i01234567, i70123456);
    bool8_128 cmp1 = v_f32_cmp(LS, n11335577, n00224466);
    bool8_128 cmp2 = v_f32_cmp(GTEQ, n00224466, n11335577);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i11335577, i00224466);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i11335577, i00224466);
    int8_128 ig = v_s32_sel(cmp, i00224466, i11335577);
    float8_128 s = minf(n11335577, n00224466);
    float8_128 g = maxf(n11335577, n00224466);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, is, ig);
    return sort128_b_i_stable_mode0(r, idx);
}

inline float8_128 sort512(float8_128 n) {
    float8_128 n01234567 = sort256(n);
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23236767 = v_f32_sel(mk, n01234567, n23456701);
    float8_128 n01014545 = v_f32_sel(mk, n67012345, n01234567);
    float8_128 n32367672 = v_row_rotate(n23236767, 0);
    float8_128 n72323676 = v_row_rotate(n23236767, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n32327676 = v_f32_sel(mk2, n32367672, n72323676);
    n32327676 = re128(n32327676);
    float8_128 g = maxf(n01014545, n32327676);
    float8_128 s = minf(n01014545, n32327676);
    float8_128 r = v_f32_sel(mk, g, s);
    return sort256_b(r);
}

inline float8_128 sort512_i(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort256_i(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23236767 = v_f32_sel(mk, n01234567, n23456701);
    float8_128 n01014545 = v_f32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    float8_128 n32367672 = v_row_rotate(n23236767, 0);
    float8_128 n72323676 = v_row_rotate(n23236767, 1);
    int8_128 i32367672 = v_u32_rotate(i23236767, 0);
    int8_128 i72323676 = v_u32_rotate(i23236767, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n32327676 = v_f32_sel(mk2, n32367672, n72323676);
    int8_128 i32327676 = v_s32_sel(mk2, i32367672, i72323676);
    n32327676 = re128_i(n32327676, &i32327676);
    bool8_128 cmp = v_f32_cmp(GT, n01014545, n32327676);
    int8_128 is = v_s32_sel(cmp, i01014545, i32327676);
    int8_128 ig = v_s32_sel(cmp, i32327676, i01014545);
    float8_128 g = maxf(n01014545, n32327676);
    float8_128 s = minf(n01014545, n32327676);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i(r, idx);
}

inline float8_128 sort512_i_stable(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort256_i_stable(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23236767 = v_f32_sel(mk, n01234567, n23456701);
    float8_128 n01014545 = v_f32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    float8_128 n32367672 = v_row_rotate(n23236767, 0);
    float8_128 n72323676 = v_row_rotate(n23236767, 1);
    int8_128 i32367672 = v_u32_rotate(i23236767, 0);
    int8_128 i72323676 = v_u32_rotate(i23236767, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n32327676 = v_f32_sel(mk2, n32367672, n72323676);
    int8_128 i32327676 = v_s32_sel(mk2, i32367672, i72323676);
    n32327676 = re128_i(n32327676, &i32327676);
    bool8_128 cmp1 = v_f32_cmp(GT, n01014545, n32327676);
    bool8_128 cmp2 = v_f32_cmp(LSEQ, n32327676, n01014545);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i01014545, i32327676);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i01014545, i32327676);
    int8_128 ig = v_s32_sel(cmp, i32327676, i01014545);
    float8_128 g = maxf(n01014545, n32327676);
    float8_128 s = minf(n01014545, n32327676);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i_stable(r, idx);
}

inline float8_128 sort512_i_stable_mode0(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort256_i_stable_mode0(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23236767 = v_f32_sel(mk, n01234567, n23456701);
    float8_128 n01014545 = v_f32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    float8_128 n32367672 = v_row_rotate(n23236767, 0);
    float8_128 n72323676 = v_row_rotate(n23236767, 1);
    int8_128 i32367672 = v_u32_rotate(i23236767, 0);
    int8_128 i72323676 = v_u32_rotate(i23236767, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n32327676 = v_f32_sel(mk2, n32367672, n72323676);
    int8_128 i32327676 = v_s32_sel(mk2, i32367672, i72323676);
    n32327676 = re128_i(n32327676, &i32327676);
    bool8_128 cmp1 = v_f32_cmp(LS, n01014545, n32327676);
    bool8_128 cmp2 = v_f32_cmp(GTEQ, n32327676, n01014545);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i01014545, i32327676);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i01014545, i32327676);
    int8_128 ig = v_s32_sel(cmp, i32327676, i01014545);
    float8_128 g = maxf(n01014545, n32327676);
    float8_128 s = minf(n01014545, n32327676);
    float8_128 r = v_f32_sel(mk, s, g);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i_stable_mode0(r, idx);
}

inline float8_128 sort512_b(float8_128 n) {
    float8_128 n01234567 = n;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23236767 = v_f32_sel(mk, n01234567, n23456701);
    float8_128 n01014545 = v_f32_sel(mk, n67012345, n01234567);
    float8_128 g = maxf(n01014545, n23236767);
    float8_128 s = minf(n01014545, n23236767);
    float8_128 r = v_f32_sel(mk, g, s);
    return sort256_b(r);
}

inline float8_128 sort512_b_i(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23236767 = v_f32_sel(mk, n01234567, n23456701);
    float8_128 n01014545 = v_f32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    bool8_128 cmp = v_f32_cmp(GT, n01014545, n23236767);
    int8_128 is = v_s32_sel(cmp, i01014545, i23236767);
    int8_128 ig = v_s32_sel(cmp, i23236767, i01014545);
    float8_128 g = maxf(n01014545, n23236767);
    float8_128 s = minf(n01014545, n23236767);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i(r, idx);
}

inline float8_128 sort512_b_i_stable(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23236767 = v_f32_sel(mk, n01234567, n23456701);
    float8_128 n01014545 = v_f32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    bool8_128 cmp1 = v_f32_cmp(GT, n01014545, n23236767);
    bool8_128 cmp2 = v_f32_cmp(LSEQ, n23236767, n01014545);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i01014545, i23236767);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i01014545, i23236767);
    int8_128 ig = v_s32_sel(cmp, i23236767, i01014545);
    float8_128 g = maxf(n01014545, n23236767);
    float8_128 s = minf(n01014545, n23236767);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i_stable(r, idx);
}

inline float8_128 sort512_b_i_stable_mode0(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23236767 = v_f32_sel(mk, n01234567, n23456701);
    float8_128 n01014545 = v_f32_sel(mk, n67012345, n01234567);
    int8_128 i23236767 = v_s32_sel(mk, i01234567, i23456701);
    int8_128 i01014545 = v_s32_sel(mk, i67012345, i01234567);
    bool8_128 cmp1 = v_f32_cmp(LS, n01014545, n23236767);
    bool8_128 cmp2 = v_f32_cmp(GTEQ, n23236767, n01014545);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i01014545, i23236767);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i01014545, i23236767);
    int8_128 ig = v_s32_sel(cmp, i23236767, i01014545);
    float8_128 g = maxf(n01014545, n23236767);
    float8_128 s = minf(n01014545, n23236767);
    float8_128 r = v_f32_sel(mk, s, g);
    *idx = v_s32_sel(mk, ig, is);
    return sort256_b_i_stable_mode0(r, idx);
}

inline float8_128 sort1024(float8_128 n) {
    float8_128 n01234567 = sort512(n);
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n34567012 = v_row_rotate(n23456701, 0);
    float8_128 n45670123 = v_row_rotate(n34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    float8_128 n45674567 = v_f32_sel(mk, n01234567, n45670123);
    float8_128 n01230123 = v_f32_sel(mk, n45670123, n01234567);
    float8_128 n56745674 = v_row_rotate(n45674567, 0);
    float8_128 n74567456 = v_row_rotate(n45674567, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n54765476 = v_f32_sel(mk2, n56745674, n74567456);
    float8_128 n47654765 = v_row_rotate(n54765476, 0);
    float8_128 n76547654 = v_row_rotate(n47654765, 0);
    n76547654 = re128(n76547654);
    float8_128 g = maxf(n76547654, n01230123);
    float8_128 s = minf(n76547654, n01230123);
    float8_128 r = v_f32_sel(mk, g, s);
    return sort512_b(r);
}

inline float8_128 sort1024_i(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort512_i(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n34567012 = v_row_rotate(n23456701, 0);
    float8_128 n45670123 = v_row_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    float8_128 n45674567 = v_f32_sel(mk, n01234567, n45670123);
    float8_128 n01230123 = v_f32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    float8_128 n56745674 = v_row_rotate(n45674567, 0);
    float8_128 n74567456 = v_row_rotate(n45674567, 1);
    int8_128 i56745674 = v_u32_rotate(i45674567, 0);
    int8_128 i74567456 = v_u32_rotate(i45674567, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n54765476 = v_f32_sel(mk2, n56745674, n74567456);
    int8_128 i54765476 = v_s32_sel(mk2, i56745674, i74567456);
    float8_128 n47654765 = v_row_rotate(n54765476, 0);
    float8_128 n76547654 = v_row_rotate(n47654765, 0);
    int8_128 i47654765 = v_u32_rotate(i54765476, 0);
    int8_128 i76547654 = v_u32_rotate(i47654765, 0);
    n76547654 = re128_i(n76547654, &i76547654);
    bool8_128 cmp = v_f32_cmp(GT, n76547654, n01230123);
    int8_128 is = v_s32_sel(cmp, i76547654, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i76547654);
    float8_128 g = maxf(n76547654, n01230123);
    float8_128 s = minf(n76547654, n01230123);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i(r, idx);
}

inline float8_128 sort1024_i_stable(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort512_i_stable(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n34567012 = v_row_rotate(n23456701, 0);
    float8_128 n45670123 = v_row_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    float8_128 n45674567 = v_f32_sel(mk, n01234567, n45670123);
    float8_128 n01230123 = v_f32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    float8_128 n56745674 = v_row_rotate(n45674567, 0);
    float8_128 n74567456 = v_row_rotate(n45674567, 1);
    int8_128 i56745674 = v_u32_rotate(i45674567, 0);
    int8_128 i74567456 = v_u32_rotate(i45674567, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n54765476 = v_f32_sel(mk2, n56745674, n74567456);
    int8_128 i54765476 = v_s32_sel(mk2, i56745674, i74567456);
    float8_128 n47654765 = v_row_rotate(n54765476, 0);
    float8_128 n76547654 = v_row_rotate(n47654765, 0);
    int8_128 i47654765 = v_u32_rotate(i54765476, 0);
    int8_128 i76547654 = v_u32_rotate(i47654765, 0);
    n76547654 = re128_i(n76547654, &i76547654);
    bool8_128 cmp1 = v_f32_cmp(GT, n76547654, n01230123);
    bool8_128 cmp2 = v_f32_cmp(LSEQ, n01230123, n76547654);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i76547654, i01230123);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i76547654, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i76547654);
    float8_128 g = maxf(n76547654, n01230123);
    float8_128 s = minf(n76547654, n01230123);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i_stable(r, idx);
}

inline float8_128 sort1024_i_stable_mode0(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = sort512_i_stable_mode0(n, idx);
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n34567012 = v_row_rotate(n23456701, 0);
    float8_128 n45670123 = v_row_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    float8_128 n45674567 = v_f32_sel(mk, n01234567, n45670123);
    float8_128 n01230123 = v_f32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    float8_128 n56745674 = v_row_rotate(n45674567, 0);
    float8_128 n74567456 = v_row_rotate(n45674567, 1);
    int8_128 i56745674 = v_u32_rotate(i45674567, 0);
    int8_128 i74567456 = v_u32_rotate(i45674567, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n54765476 = v_f32_sel(mk2, n56745674, n74567456);
    int8_128 i54765476 = v_s32_sel(mk2, i56745674, i74567456);
    float8_128 n47654765 = v_row_rotate(n54765476, 0);
    float8_128 n76547654 = v_row_rotate(n47654765, 0);
    int8_128 i47654765 = v_u32_rotate(i54765476, 0);
    int8_128 i76547654 = v_u32_rotate(i47654765, 0);
    n76547654 = re128_i(n76547654, &i76547654);
    bool8_128 cmp1 = v_f32_cmp(LS, n76547654, n01230123);
    bool8_128 cmp2 = v_f32_cmp(GTEQ, n01230123, n76547654);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i76547654, i01230123);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i76547654, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i76547654);
    float8_128 g = maxf(n76547654, n01230123);
    float8_128 s = minf(n76547654, n01230123);
    float8_128 r = v_f32_sel(mk, s, g);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i_stable_mode0(r, idx);
}

inline float8_128 sort1024_b(float8_128 n) {
    float8_128 n01234567 = n;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n34567012 = v_row_rotate(n23456701, 0);
    float8_128 n45670123 = v_row_rotate(n34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    float8_128 n45674567 = v_f32_sel(mk, n01234567, n45670123);
    float8_128 n01230123 = v_f32_sel(mk, n45670123, n01234567);
    float8_128 g = maxf(n45674567, n01230123);
    float8_128 s = minf(n45674567, n01230123);
    float8_128 r = v_f32_sel(mk, g, s);
    return sort512_b(r);
}

inline float8_128 sort1024_b_i(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n34567012 = v_row_rotate(n23456701, 0);
    float8_128 n45670123 = v_row_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    float8_128 n45674567 = v_f32_sel(mk, n01234567, n45670123);
    float8_128 n01230123 = v_f32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    bool8_128 cmp = v_f32_cmp(GT, n45674567, n01230123);
    int8_128 is = v_s32_sel(cmp, i45674567, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i45674567);
    float8_128 g = maxf(n45674567, n01230123);
    float8_128 s = minf(n45674567, n01230123);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i(r, idx);
}

inline float8_128 sort1024_b_i_stable(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n34567012 = v_row_rotate(n23456701, 0);
    float8_128 n45670123 = v_row_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    float8_128 n45674567 = v_f32_sel(mk, n01234567, n45670123);
    float8_128 n01230123 = v_f32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    bool8_128 cmp1 = v_f32_cmp(GT, n45674567, n01230123);
    bool8_128 cmp2 = v_f32_cmp(LSEQ, n01230123, n45674567);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i45674567, i01230123);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i45674567, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i45674567);
    float8_128 g = maxf(n45674567, n01230123);
    float8_128 s = minf(n45674567, n01230123);
    float8_128 r = v_f32_sel(mk, g, s);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i_stable(r, idx);
}

inline float8_128 sort1024_b_i_stable_mode0(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n34567012 = v_row_rotate(n23456701, 0);
    float8_128 n45670123 = v_row_rotate(n34567012, 0);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i34567012 = v_u32_rotate(i23456701, 0);
    int8_128 i45670123 = v_u32_rotate(i34567012, 0);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(7)), _S(4));
    float8_128 n45674567 = v_f32_sel(mk, n01234567, n45670123);
    float8_128 n01230123 = v_f32_sel(mk, n45670123, n01234567);
    int8_128 i45674567 = v_s32_sel(mk, i01234567, i45670123);
    int8_128 i01230123 = v_s32_sel(mk, i45670123, i01234567);
    bool8_128 cmp1 = v_f32_cmp(LS, n45674567, n01230123);
    bool8_128 cmp2 = v_f32_cmp(GTEQ, n01230123, n45674567);
    bool8_128 cmp3 = cmp1 ^ cmp2;
    bool8_128 cmp4 = v_s32_cmp(GT, i45674567, i01230123);
    bool8_128 cmp = v_s32_cmp(EQ,
                              v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                        v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                              v_u32_move_i(1));
    int8_128 is = v_s32_sel(cmp, i45674567, i01230123);
    int8_128 ig = v_s32_sel(cmp, i01230123, i45674567);
    float8_128 g = maxf(n45674567, n01230123);
    float8_128 s = minf(n45674567, n01230123);
    float8_128 r = v_f32_sel(mk, s, g);
    *idx = v_s32_sel(mk, ig, is);
    return sort512_b_i_stable_mode0(r, idx);
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

inline void cmp_exchange_1024_p0(int lenp2, SIM_X86::tensor d) {
    for (int i = 0; i < lenp2; i += 1024) {
        float8_128 v = v_f32_ld_tnsr_b(i / 32, d);
        v = sort1024(v);
        v_f32_st_tnsr_b(i / 32, d, v);
    }
}

inline void cmp_exchange_1024_p0_i(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        float8_128 v = v_f32_ld_tnsr_b(i / 32, d);
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_i(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, v);
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}

inline void cmp_exchange_1024_p0_i_stable(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        float8_128 v = v_f32_ld_tnsr_b(i / 32, d);
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_i_stable(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, v);
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}

inline void cmp_exchange_1024_p0_i_stable_mode0(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        float8_128 v = v_f32_ld_tnsr_b(i / 32, d);
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_i_stable_mode0(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, v);
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}

inline void cmp_exchange_1024_pb0(int lenp2, SIM_X86::tensor d) {
    for (int i = 0; i < lenp2; i += 1024) {
        float8_128 v = v_f32_ld_tnsr_b(i / 32, d);
        v = sort1024_b(v);
        v_f32_st_tnsr_b(i / 32, d, v);
    }
}

inline void cmp_exchange_1024_pb0_i(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        float8_128 v = v_f32_ld_tnsr_b(i / 32, d);
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_b_i(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, v);
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}

inline void cmp_exchange_1024_pb0_i_stable(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        float8_128 v = v_f32_ld_tnsr_b(i / 32, d);
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_b_i_stable(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, v);
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}

inline void cmp_exchange_1024_pb0_i_stable_mode0(int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    for (int i = 0; i < lenp2; i += 1024) {
        float8_128 v = v_f32_ld_tnsr_b(i / 32, d);
        int8_128 ix = $S(v_f32_ld_tnsr_b(i / 32, idx));
        v = sort1024_b_i_stable_mode0(v, &ix);
        v_f32_st_tnsr_b(i / 32, d, v);
        v_f32_st_tnsr_b(i / 32, idx, $F(ix));
    }
}

inline float8_128 re1024(float8_128 n) {
    float8_128 n01234567 = n;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23016745 = v_f32_sel(mk, n67012345, n23456701);
    float8_128 n30167452 = v_row_rotate(n23016745, 0);
    float8_128 n52301674 = v_row_rotate(n23016745, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n32107654 = v_f32_sel(mk2, n30167452, n52301674);
    float8_128 n21076543 = v_row_rotate(n32107654, 0);
    float8_128 n10765432 = v_row_rotate(n21076543, 0);
    float8_128 n07654321 = v_row_rotate(n10765432, 0);
    float8_128 n76543210 = v_row_rotate(n07654321, 0);
    return re128(n76543210);
}

inline float8_128 re1024_i(float8_128 n, int8_128 *idx) {
    float8_128 n01234567 = n;
    int8_128 i01234567 = *idx;
    float8_128 n12345670 = v_row_rotate(n01234567, 0);
    float8_128 n23456701 = v_row_rotate(n12345670, 0);
    float8_128 n70123456 = v_row_rotate(n01234567, 1);
    float8_128 n67012345 = v_row_rotate(n70123456, 1);
    int8_128 i12345670 = v_u32_rotate(i01234567, 0);
    int8_128 i23456701 = v_u32_rotate(i12345670, 0);
    int8_128 i70123456 = v_u32_rotate(i01234567, 1);
    int8_128 i67012345 = v_u32_rotate(i70123456, 1);
    bool8_128 mk = v_s32_cmp(LS, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(3)), _S(2));
    float8_128 n23016745 = v_f32_sel(mk, n67012345, n23456701);
    int8_128 i23016745 = v_s32_sel(mk, i67012345, i23456701);
    float8_128 n30167452 = v_row_rotate(n23016745, 0);
    float8_128 n52301674 = v_row_rotate(n23016745, 1);
    int8_128 i30167452 = v_u32_rotate(i23016745, 0);
    int8_128 i52301674 = v_u32_rotate(i23016745, 1);
    bool8_128 mk2 = v_s32_cmp(EQ, v_u32_and(v_u32_shr(get_core_id(), _S(7)), _S(1)), _S(1));
    float8_128 n32107654 = v_f32_sel(mk2, n30167452, n52301674);
    int8_128 i32107654 = v_s32_sel(mk2, i30167452, i52301674);
    float8_128 n21076543 = v_row_rotate(n32107654, 0);
    float8_128 n10765432 = v_row_rotate(n21076543, 0);
    float8_128 n07654321 = v_row_rotate(n10765432, 0);
    float8_128 n76543210 = v_row_rotate(n07654321, 0);
    int8_128 i21076543 = v_u32_rotate(i32107654, 0);
    int8_128 i10765432 = v_u32_rotate(i21076543, 0);
    int8_128 i07654321 = v_u32_rotate(i10765432, 0);
    int8_128 i76543210 = v_u32_rotate(i07654321, 0);
    *idx = i76543210;
    return re128_i(n76543210, idx);
}

inline void cmp_exchange_1024_p(int swapl, int lenp2, SIM_X86::tensor d) {
    // k * W + n -> swap01(k) * W + (W - 1 - n)
    int g = (lenp2 / 1024) >> swapl;
    int w = 1 << swapl;
    int hw = w / 2;
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < hw; j++) {
            int laddr = (i * 2) * hw + j;
            int raddr = (i * 2 + 1) * hw + (hw - 1 - j);
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, d);
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, d);
            r = re1024(r);
            v_f32_st_tnsr_b(laddr * 32, d, minf(l, r));
            v_f32_st_tnsr_b(raddr * 32, d, maxf(l, r));
        }
    }
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
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, d);
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, d);
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            r = re1024_i(r, &rx);
            bool8_128 cmp = v_f32_cmp(GT, l, r);
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, minf(l, r));
            v_f32_st_tnsr_b(raddr * 32, d, maxf(l, r));
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
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, d);
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, d);
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            r = re1024_i(r, &rx);
            bool8_128 cmp1 = v_f32_cmp(GT, l, r);
            bool8_128 cmp2 = v_f32_cmp(LSEQ, r, l);
            bool8_128 cmp3 = cmp1 ^ cmp2;
            bool8_128 cmp4 = v_s32_cmp(GT, lx, rx);
            bool8_128 cmp = v_s32_cmp(EQ,
                                      v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                                v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                                      v_u32_move_i(1));
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, minf(l, r));
            v_f32_st_tnsr_b(raddr * 32, d, maxf(l, r));
            v_f32_st_tnsr_b(laddr * 32, idx, $F(is));
            v_f32_st_tnsr_b(raddr * 32, idx, $F(ig));
        }
    }
}

inline void cmp_exchange_1024_p_i_stable_mode0(int swapl, int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    // k * W + n -> swap01(k) * W + (W - 1 - n)
    int g = (lenp2 / 1024) >> swapl;
    int w = 1 << swapl;
    int hw = w / 2;
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < hw; j++) {
            int laddr = (i * 2) * hw + j;
            int raddr = (i * 2 + 1) * hw + (hw - 1 - j);
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, d);
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, d);
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            r = re1024_i(r, &rx);
            bool8_128 cmp1 = v_f32_cmp(LS, l, r);
            bool8_128 cmp2 = v_f32_cmp(GTEQ, r, l);
            bool8_128 cmp3 = cmp1 ^ cmp2;
            bool8_128 cmp4 = v_s32_cmp(GT, lx, rx);
            bool8_128 cmp = v_s32_cmp(EQ,
                                      v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                                v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                                      v_u32_move_i(1));
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, maxf(l, r));
            v_f32_st_tnsr_b(raddr * 32, d, minf(l, r));
            v_f32_st_tnsr_b(laddr * 32, idx, $F(is));
            v_f32_st_tnsr_b(raddr * 32, idx, $F(ig));
        }
    }
}

inline void cmp_exchange_1024_pb(int swapl, int lenp2, SIM_X86::tensor d) {
    // k * W + n -> swap01(k) * W + n
    int g = (lenp2 / 1024) >> swapl;
    int w = 1 << swapl;
    int hw = w / 2;
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < hw; j++) {
            int laddr = (i * 2) * hw + j;
            int raddr = (i * 2 + 1) * hw + j;
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, d);
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, d);
            v_f32_st_tnsr_b(laddr * 32, d, minf(l, r));
            v_f32_st_tnsr_b(raddr * 32, d, maxf(l, r));
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
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, d);
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, d);
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            bool8_128 cmp = v_f32_cmp(GT, l, r);
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, minf(l, r));
            v_f32_st_tnsr_b(raddr * 32, d, maxf(l, r));
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
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, d);
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, d);
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            bool8_128 cmp1 = v_f32_cmp(GT, l, r);
            bool8_128 cmp2 = v_f32_cmp(LSEQ, r, l);
            bool8_128 cmp3 = cmp1 ^ cmp2;
            bool8_128 cmp4 = v_s32_cmp(GT, lx, rx);
            bool8_128 cmp = v_s32_cmp(EQ,
                                      v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                                v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                                      v_u32_move_i(1));
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, minf(l, r));
            v_f32_st_tnsr_b(raddr * 32, d, maxf(l, r));
            v_f32_st_tnsr_b(laddr * 32, idx, $F(is));
            v_f32_st_tnsr_b(raddr * 32, idx, $F(ig));
        }
    }
}

inline void cmp_exchange_1024_pb_i_stable_mode0(int swapl, int lenp2, SIM_X86::tensor d, SIM_X86::tensor idx) {
    // k * W + n -> swap01(k) * W + n
    int g = (lenp2 / 1024) >> swapl;
    int w = 1 << swapl;
    int hw = w / 2;
    for (int i = 0; i < g; i++) {
        for (int j = 0; j < hw; j++) {
            int laddr = (i * 2) * hw + j;
            int raddr = (i * 2 + 1) * hw + j;
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, d);
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, d);
            int8_128 lx = $S(v_f32_ld_tnsr_b(laddr * 32, idx));
            int8_128 rx = $S(v_f32_ld_tnsr_b(raddr * 32, idx));
            bool8_128 cmp1 = v_f32_cmp(LS, l, r);
            bool8_128 cmp2 = v_f32_cmp(GTEQ, r, l);
            bool8_128 cmp3 = cmp1 ^ cmp2;
            bool8_128 cmp4 = v_s32_cmp(GT, lx, rx);
            bool8_128 cmp = v_s32_cmp(EQ,
                                      v_s32_sel(cmp3, v_s32_sel(cmp1, v_u32_move_i(0), v_u32_move_i(1)),
                                                v_s32_sel(cmp4, v_u32_move_i(0), v_u32_move_i(1))),
                                      v_u32_move_i(1));
            int8_128 is = v_s32_sel(cmp, lx, rx);
            int8_128 ig = v_s32_sel(cmp, rx, lx);
            v_f32_st_tnsr_b(laddr * 32, d, maxf(l, r));
            v_f32_st_tnsr_b(raddr * 32, d, minf(l, r));
            v_f32_st_tnsr_b(laddr * 32, idx, $F(is));
            v_f32_st_tnsr_b(raddr * 32, idx, $F(ig));
        }
    }
}

inline void sort(SIM_X86::tensor d, int lenp2) {
    int lg = lg2(lenp2 / 1024);

    cmp_exchange_1024_p0(lenp2, d); // 0

    for (int i = 1; i <= lg; i++) {
        cmp_exchange_1024_p(i, lenp2, d); // 1
        for (int j = i - 1; j > 0; j--) {
            cmp_exchange_1024_pb(j, lenp2, d);
        }
        cmp_exchange_1024_pb0(lenp2, d);
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

inline void sort_i_stable_mode0(SIM_X86::tensor d, SIM_X86::tensor idx, int lenp2) {
    int lg = lg2(lenp2 / 1024);

    cmp_exchange_1024_p0_i_stable_mode0(lenp2, d, idx); // 0

    for (int i = 1; i <= lg; i++) {
        cmp_exchange_1024_p_i_stable_mode0(i, lenp2, d, idx); // 1
        for (int j = i - 1; j > 0; j--) {
            cmp_exchange_1024_pb_i_stable_mode0(j, lenp2, d, idx);
        }
        cmp_exchange_1024_pb0_i_stable_mode0(lenp2, d, idx);
    }
}

inline void sort_line(SIM_X86::tensor a, SIM_X86::tensor tmp, SIM_X86::tensor b, int len, bool ascending) {
    float8_128 padding_v = ascending ? _F(3.40282e+38f) : _F(-3.40282e+38f);
    int l2 = max(1024, nearx2(len));
    int fcpyl = len & 0xfffffc00;
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, v_f32_ld_tnsr_b(i / 32, a));
    }
    int rest = len - fcpyl;
    int mk = (1 << ((rest + 127) / 128)) - 1;
    if (rest > 0) {
        float8_128 v = v_f32_ld_tnsr_st_msk(fcpyl / 32, a, 1, mk);
        bool8_128 m = v_s32_cmp(LS, get_core_id(), _S(rest));
        v = v_f32_sel(m, padding_v, v);
        v_f32_st_tnsr_b(fcpyl / 32, tmp, v);
    }
    int pad = (len + 1023) & 0xfffffc00;
    for (int i = pad; i < l2; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, padding_v);
    }
    sort(tmp, l2);
    if (!ascending) {
        int l21024 = l2 / 1024;
        for (int i = 0; i < l21024 / 2; i++) {
            int laddr = i;
            int raddr = l21024 - 1 - i;
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, tmp);
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, tmp);
            l = re1024(l);
            r = re1024(r);
            v_f32_st_tnsr_b(laddr * 32, tmp, r);
            v_f32_st_tnsr_b(raddr * 32, tmp, l);
        }
        if (l21024 == 1) {
            float8_128 l = v_f32_ld_tnsr_b(0, tmp);
            l = re1024(l);
            v_f32_st_tnsr_b(0, tmp, l);
        }
    }
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, b, v_f32_ld_tnsr_b(i / 32, tmp));
    }
    if (rest > 0) {
        float8_128 v = v_f32_ld_tnsr_b(fcpyl / 32, tmp);
        v_f32_st_tnsr_st_msk(fcpyl / 32, b, 1, mk, v);
    }
}

inline void sort_line_i(SIM_X86::tensor a, SIM_X86::tensor tmp, SIM_X86::tensor tmpidx, SIM_X86::tensor b, SIM_X86::tensor idx, int len, bool ascending) {
    float8_128 padding_v = ascending ? _F(3.40282e+38f) : _F(-3.40282e+38f);
    int l2 = max(1024, nearx2(len));
    int fcpyl = len & 0xfffffc00;
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, v_f32_ld_tnsr_b(i / 32, a));
    }
    int rest = len - fcpyl;
    int mk = (1 << ((rest + 127) / 128)) - 1;
    if (rest > 0) {
        float8_128 v = v_f32_ld_tnsr_st_msk(fcpyl / 32, a, 1, mk);
        bool8_128 m = v_s32_cmp(LS, get_core_id(), _S(rest));
        v = v_f32_sel(m, padding_v, v);
        v_f32_st_tnsr_b(fcpyl / 32, tmp, v);
    }
    int pad = (len + 1023) & 0xfffffc00;
    for (int i = pad; i < l2; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, padding_v);
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
            float8_128 l = v_f32_ld_tnsr_b(laddr * 32, tmp);
            float8_128 r = v_f32_ld_tnsr_b(raddr * 32, tmp);
            float8_128 lx = v_f32_ld_tnsr_b(laddr * 32, tmpidx);
            float8_128 rx = v_f32_ld_tnsr_b(raddr * 32, tmpidx);
            l = re1024(l);
            r = re1024(r);
            lx = re1024(lx);
            rx = re1024(rx);
            v_f32_st_tnsr_b(laddr * 32, tmp, r);
            v_f32_st_tnsr_b(raddr * 32, tmp, l);
            v_f32_st_tnsr_b(laddr * 32, tmpidx, rx);
            v_f32_st_tnsr_b(raddr * 32, tmpidx, lx);
        }
        if (l21024 == 1) {
            float8_128 l = v_f32_ld_tnsr_b(0, tmp);
            float8_128 lx = v_f32_ld_tnsr_b(0, tmpidx);
            l = re1024(l);
            lx = re1024(lx);
            v_f32_st_tnsr_b(0, tmp, l);
            v_f32_st_tnsr_b(0, tmpidx, lx);
        }
    }
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, b, v_f32_ld_tnsr_b(i / 32, tmp));
        v_f32_st_tnsr_b(i / 32, idx, v_f32_ld_tnsr_b(i / 32, tmpidx));
    }
    if (rest > 0) {
        float8_128 v = v_f32_ld_tnsr_b(fcpyl / 32, tmp);
        v_f32_st_tnsr_st_msk(fcpyl / 32, b, 1, mk, v);
        float8_128 vi = v_f32_ld_tnsr_b(fcpyl / 32, tmpidx);
        v_f32_st_tnsr_st_msk(fcpyl / 32, idx, 1, mk, vi);
    }
}

inline void sort_line_i_stable(SIM_X86::tensor a, SIM_X86::tensor tmp, SIM_X86::tensor tmpidx, SIM_X86::tensor b, SIM_X86::tensor idx, int len,
                               bool ascending) {
    float8_128 padding_v = ascending ? _F(3.40282e+38f) : _F(-3.40282e+38f);
    int l2 = max(1024, nearx2(len));
    int fcpyl = len / 1024 * 1024;
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, v_f32_ld_tnsr_b(i / 32, a));
    }
    int rest = len % 1024;
    int mk = (1 << ((rest + 127) / 128)) - 1;
    if (rest > 0) {
        float8_128 v = v_f32_ld_tnsr_st_msk(fcpyl / 32, a, 1, mk);
        bool8_128 m = v_s32_cmp(LS, get_core_id(), _S(rest));
        v = v_f32_sel(m, padding_v, v);
        v_f32_st_tnsr_b(fcpyl / 32, tmp, v);
    }
    int pad = (len + 1023) / 1024 * 1024;
    for (int i = pad; i < l2; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmp, padding_v);
    }
    for (int i = 0; i < l2; i += 1024) {
        v_f32_st_tnsr_b(i / 32, tmpidx, $F(v_s32_add(get_core_id(), _S(i))));
    }
    if (ascending) {
        sort_i_stable(tmp, tmpidx, l2);
    } else {
        sort_i_stable_mode0(tmp, tmpidx, l2);
    }
    for (int i = 0; i < fcpyl; i += 1024) {
        v_f32_st_tnsr_b(i / 32, b, v_f32_ld_tnsr_b(i / 32, tmp));
        v_f32_st_tnsr_b(i / 32, idx, v_f32_ld_tnsr_b(i / 32, tmpidx));
    }
    if (rest > 0) {
        float8_128 v = v_f32_ld_tnsr_b(fcpyl / 32, tmp);
        v_f32_st_tnsr_st_msk(fcpyl / 32, b, 1, mk, v);
        float8_128 vi = v_f32_ld_tnsr_b(fcpyl / 32, tmpidx);
        v_f32_st_tnsr_st_msk(fcpyl / 32, idx, 1, mk, vi);
    }
}

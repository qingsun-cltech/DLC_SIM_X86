#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"



#include "permute.h"
#include "libdevice.h"
#include ".././libdevice/fdiv_rn_without_unary.h"

#define BILINEAR 0
#define NEAREST 1
#define BICUBIC 2
#define ZEROS 0
#define BORDER 1
#define REFLECTION 2

/**
 * Author: QingSun
 * Print single value in vreg by idx[0 ~ 1023], type: 0 -> float, 1 -> int
*/
inline void PrintVregSingle(float8_128 vreg, const int idx, const int type) {
  int subcore_rotate_count = idx / 128;
  int m_rotate_count = idx - subcore_rotate_count * 128;

  for (int i = 0; i < subcore_rotate_count; ++i) {
    vreg = v_row_rotate(vreg, 0);
  }
  m_rotate(vreg, -m_rotate_count, true);

  if (type == 0) {
    Print("vreg = %f\n", vreg[0]);
  } else {
    Print("vreg = %d\n", vreg[0]);
  }
}

/**
 * cpu和cuda上：float2int, rz, rd, rn, ru, 对 inf, -inf, nan, -nan, 都 = -2147483648
*/
inline int8_128 float2int_clean_invalid(const float8_128 a, const int8_128 b) {
  int8_128 nan = __dlc_isnanf(a);
  int8_128 inf = __dlc_isinff(a);

  bool8_128 nan_flag = v_s32_cmp(NEQ, nan, 0);
  bool8_128 inf_flag = v_s32_cmp(NEQ, inf, 0);
  
  int8_128 re = b;
  re = v_s32_sel(nan_flag, re, -2147483648);
  re = v_s32_sel(inf_flag, re, -2147483648);

  return re;
}

/* ((A + 2) * x - (A + 3)) * x * x + 1 */
inline float8_128 cubic_convolution1(float8_128 x, float8_128 A) {
  float8_128 A_2 = v_f32_add_b(A, 2.0f);
  float8_128 A_3 = v_f32_add_b(A, 3.0f);
  float8_128 A_2_x = v_f32_mul_b(A_2, x);
  float8_128 A_2_x_A_3 = v_f32_sub_b(A_2_x, A_3);
  float8_128 x_x = v_f32_mul_b(x, x);
  float8_128 A_2_x_A_3_x_x = v_f32_mul_b(A_2_x_A_3, x_x);
  
  return v_f32_add_b(A_2_x_A_3_x_x, 1.0f);
}

/* ((A * x - 5 * A) * x + 8 * A) * x - 4 * A */
inline float8_128 cubic_convolution2(float8_128 x, float8_128 A) {
  float8_128 A_x = v_f32_mul_b(A, x);
  float8_128 A_4 = v_f32_mul_b(A, 4.0f);
  float8_128 A_5 = v_f32_mul_b(A, 5.0f);
  float8_128 A_8 = v_f32_mul_b(A, 8.0f);
  float8_128 A_x_A_5 = v_f32_sub_b(A_x, A_5);
  float8_128 A_x_A_5_x = v_f32_mul_b(A_x_A_5, x);
  float8_128 A_x_A_5_x_A_8 = v_f32_add_b(A_x_A_5_x, A_8);
  float8_128 A_x_A_5_x_A_8_x = v_f32_mul_b(A_x_A_5_x_A_8, x);

	return v_f32_sub_b(A_x_A_5_x_A_8_x, A_4);
}

inline void get_cubic_upsample_coefficients(
    float8_128 __attribute__((address_space(2))) *coeffs, float8_128 t) {
	float8_128 A = -0.75f;

	float8_128 x1 = t;
	coeffs[0] = cubic_convolution2(v_f32_add_b(x1, 1.0f), A);
	coeffs[1] = cubic_convolution1(x1, A);

	// opposite coefficients
	float8_128 x2 = 1.0f - t;
	coeffs[2] = cubic_convolution1(x2, A);
  coeffs[3] = cubic_convolution2(v_f32_add_b(x2, 1.0f), A);
}

inline void get_cubic_upsampling_coefficients(
    float8_128 __attribute__((address_space(2))) *coeffs, float8_128 t) {
  float8_128 A = -0.75f;

  float8_128 x1 = t;
	coeffs[0] = cubic_convolution2(v_f32_add_b(x1, 1.0f), A);
	coeffs[1] = cubic_convolution1(x1, A);

	// opposite coefficients
	float8_128 x2 = 1.0f - t;
	coeffs[2] = cubic_convolution1(x2, A);
  coeffs[3] = cubic_convolution2(v_f32_add_b(x2, 1.0f), A);
}

inline void get_cubic_coefficients_grad(
    float8_128 __attribute__((address_space(2))) *coeffs, float8_128 t) {
  float8_128 A = -0.75f;
  float8_128 x;

  x = v_f32_add_b(-t, -1.0f);
  // coeffs[0] = (-3.0f * A * x - 10.0f * A) * x - 8.0f * A;
  if (1) {
    float8_128 A_3 = v_f32_mul_b(A, -3.0f);
    float8_128 A_3_x = v_f32_mul_b(A_3, x);
    float8_128 A_10 = v_f32_mul_b(A, -10.0f);
    float8_128 A_3_x_A_10 = v_f32_add_b(A_3_x, A_10);
    float8_128 A_3_x_A_10_x = v_f32_mul_b(A_3_x_A_10, x);
    float8_128 A_8 = v_f32_mul_b(A, -8.0f);
    float8_128 A_3_x_A_10_x_A_8 = v_f32_add_b(A_3_x_A_10_x, A_8);
    coeffs[0] = A_3_x_A_10_x_A_8;
  }

  x = -t;
  // coeffs[1] = (-3.0f * (A + 2.0f) * x - 2.0f * (A + 3.0f)) * x;
  if (1) {
    float8_128 A_2 = v_f32_add_b(A, 2.0f);
    float8_128 A_2_3 = v_f32_mul_b(A_2, -3.0f);
    float8_128 A_2_3_x = v_f32_mul_b(A_2_3, x);
    float8_128 A_3 = v_f32_add_b(A, 3.0f);
    float8_128 A_3_2 = v_f32_mul_b(A_3, -2.0f);
    float8_128 A_2_3_x_A_3_2 = v_f32_add_b(A_2_3_x, A_3_2);
    float8_128 A_2_3_x_A_3_2_x = v_f32_mul_b(A_2_3_x_A_3_2, x);
    coeffs[1] = A_2_3_x_A_3_2_x;
  }

  x = v_f32_add_b(-t, 1.0f);
  // coeffs[2] = (3.0f * (A + 2.0f) * x - 2.0f * (A + 3.0f)) * x;
  if (1) {
    float8_128 A_2 = v_f32_add_b(A, 2.0f);
    float8_128 A_2_3 = v_f32_mul_b(A_2, 3.0f);
    float8_128 A_2_3_x = v_f32_mul_b(A_2_3, x);
    float8_128 A_3 = v_f32_add_b(A, 3.0f);
    float8_128 A_3_2 = v_f32_mul_b(A_3, -2.0f);
    float8_128 A_2_3_x_A_3_2 = v_f32_add_b(A_2_3_x, A_3_2);
    float8_128 A_2_3_x_A_3_2_x = v_f32_mul_b(A_2_3_x_A_3_2, x);
    coeffs[2] = A_2_3_x_A_3_2_x;
  }

  x = v_f32_add_b(-t, 2.0f);
  // coeffs[3] = (3.0f * A * x - 10.0f * A) * x + 8.0f * A;
  if (1) {
    float8_128 A_3 = v_f32_mul_b(A, 3.0f);
    float8_128 A_3_x = v_f32_mul_b(A_3, x);
    float8_128 A_10 = v_f32_mul_b(A, -10.0f);
    float8_128 A_3_x_A_10 = v_f32_add_b(A_3_x, A_10);
    float8_128 A_3_x_A_10_x = v_f32_mul_b(A_3_x_A_10, x);
    float8_128 A_8 = v_f32_mul_b(A, 8.0f);
    float8_128 A_3_x_A_10_x_A_8 = v_f32_add_b(A_3_x_A_10_x, A_8);
    coeffs[3] = A_3_x_A_10_x_A_8;
  }
}

/* x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3] */
inline float8_128 cubic_interp1d(
    float8_128 x0, float8_128 x1, float8_128 x2, float8_128 x3, float8_128 t) {
	float8_128 __attribute__((address_space(2))) coeffs[4];
	get_cubic_upsample_coefficients(coeffs, t);

  float8_128 x0_coeffs0 = v_f32_mul_b(x0, coeffs[0]);
  float8_128 x1_coeffs1 = v_f32_mul_b(x1, coeffs[1]);
  float8_128 x2_coeffs2 = v_f32_mul_b(x2, coeffs[2]);
  float8_128 x3_coeffs3 = v_f32_mul_b(x3, coeffs[3]);

  float8_128 x0_coeffs0_x1_coeffs1 = v_f32_add_b(x0_coeffs0, x1_coeffs1);
  float8_128 x2_coeffs2_x3_coeffs3 = v_f32_add_b(x2_coeffs2, x3_coeffs3);

	return v_f32_add_b(x0_coeffs0_x1_coeffs1, x2_coeffs2_x3_coeffs3);
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
inline float8_128 reflect_coordinates(
    float8_128 coord, const int twice_low, const int twice_high) {
  if (twice_low == twice_high) {
    return 0.0f;
  }

  float8_128 minn = (float)twice_low * 0.5f;
  float8_128 span = (float)(twice_high - twice_low) * 0.5f;
  coord = v_f32_abs(v_f32_sub_b(coord, minn));

  float8_128 extra = __dlc_fmodf_without_unary(coord, span); // std::fmod(coord, span)

  float8_128 coord_pos = __$F(__$S(coord) & 0x7fffffff);
  float8_128 span_pos = __$F(__$S(span) & 0x7fffffff);
  float8_128 coord_divide_span = v_f32_mul_b(coord_pos, __dlc_frcp_rd_without_unary(span_pos));
  int8_128 coord_span_pos = __dlc_float2int_rz(coord_divide_span);
  coord_span_pos = float2int_clean_invalid(coord_divide_span, coord_span_pos);
  float8_128 coord_coord_span_pos = v_f32_sub_b(coord, v_f32_mul_b(span_pos, v_cvt_itof(coord_span_pos)));
  bool8_128 diff = v_f32_cmp(EQ, span_pos, coord_coord_span_pos);
  coord_span_pos = v_s32_sel(diff, coord_span_pos, v_s32_add(coord_span_pos, 1));
  int8_128 res_flag = (__$S(coord) & 0x80000000) ^ (__$S(span) & 0x80000000);
  int8_128 flips = v_u32_or(coord_span_pos, res_flag);

  int8_128 flips_2 = v_u32_and(flips, 1);
  bool8_128 flips2_eq = v_s32_eq(flips_2, 0);

  float8_128 res = 0.0f;
  res = v_f32_sel(flips2_eq, res, v_f32_add_b(extra, minn));
  res = v_f32_sel(!flips2_eq, res, v_f32_add_b(v_f32_sub_b(span, extra), minn));

  return res;
}

inline float8_128 reflect_coordinates_set_grad(
    float8_128 coords, int twice_low, int twice_high, float8_128 *grad_refl) {
  if (twice_low == twice_high) {
    *grad_refl = 0.0f;
    return 0.0f;
  }
  float8_128 grad_refl_mult = 1.0f;
  float8_128 min = v_f32_mul_b((float)twice_low, 0.5f);
  float8_128 span = v_f32_mul_b((float)(twice_high - twice_low), 0.5f);
  coords = v_f32_sub_b(coords, min);

  bool8_128 coords_le_0 = v_f32_cmp(LS, coords, 0.0f);
  grad_refl_mult = v_f32_sel(coords_le_0, grad_refl_mult, -1.0f);
  coords = v_f32_sel(coords_le_0, coords, -coords);

  float8_128 extra = __dlc_fmodf_without_unary(coords, span);

  float8_128 coords_pos = __$F(__$S(coords) & 0x7fffffff);
  float8_128 span_pos = __$F(__$S(span) & 0x7fffffff);
  float8_128 coords_divide_span = v_f32_mul_b(coords_pos, __dlc_frcp_rd_without_unary(span_pos));
  int8_128 coords_span_pos = __dlc_float2int_rz(coords_divide_span);
  coords_span_pos = float2int_clean_invalid(coords_divide_span, coords_span_pos);
  float8_128 coords_coords_span_pos = v_f32_sub_b(coords, v_f32_mul_b(span_pos, v_cvt_itof(coords_span_pos)));
  bool8_128 diff = v_f32_cmp(EQ, span_pos, coords_coords_span_pos);
  coords_span_pos = v_s32_sel(diff, coords_span_pos, v_s32_add(coords_span_pos, 1));
  int8_128 res_flag = (__$S(coords) & 0x80000000) ^ (__$S(span) & 0x80000000);
  int8_128 flips = v_u32_or(coords_span_pos, res_flag);

  int8_128 flips_2 = v_u32_and(flips, 1);
  bool8_128 flips_mod2_0 = v_s32_eq(flips_2, 0);
  bool8_128 flips_mod2_1 = !flips_mod2_0;

  float8_128 res = 0.0f;
  *grad_refl = v_f32_sel(flips_mod2_0, *grad_refl, grad_refl_mult);
  res = v_f32_sel(flips_mod2_0, res, v_f32_add_b(extra, min));

  *grad_refl = v_f32_sel(flips_mod2_1, *grad_refl, -grad_refl_mult);
  res = v_f32_sel(flips_mod2_1, res, v_f32_add_b(v_f32_sub_b(span, extra), min));

  return res;
}

// Clips coordinates to between 0 and clip_limit - 1
inline float8_128 clip_coordinates(float8_128 in, const int clip_limit) {  
  int8_128 isnan = __dlc_isnanf(in);
  bool8_128 flag = v_s32_cmp(NEQ, isnan, 0);
  float8_128 max_in_0 = v_f32_sel(flag, in, 0);
  max_in_0 = v_f32_sel(!flag, max_in_0, in);

  return v_f32_min((float)(clip_limit - 1.0f), v_f32_max(max_in_0, 0.0f));
}

inline float8_128 clip_coordinates_set_grad(float8_128 coord, int size, float8_128 *grad_clip) {
  float8_128 re = 0.0f;

  bool8_128 coord_lseq_0 = v_f32_cmp(LSEQ, coord, 0.0f);
  *grad_clip = v_f32_sel(coord_lseq_0, *grad_clip, 0.0f);

  float8_128 max = v_f32_sub_b((float)size, 1.0f);

  int8_128 isnan = __dlc_isnanf(coord);
  bool8_128 flag = v_s32_cmp(NEQ, isnan, 0);
  coord = v_f32_sel(flag, coord, 0);

  bool8_128 coord_gteq_max = v_f32_cmp(GTEQ, coord, max);
  *grad_clip = v_f32_sel(coord_gteq_max, *grad_clip, 0.0f);
  re = v_f32_sel(coord_gteq_max, re, max);

  bool8_128 coord_gt_0 = !coord_lseq_0;
  bool8_128 coord_le_max = !coord_gteq_max;
  bool8_128 coord_gt_0_le_max = coord_gt_0 & coord_le_max;
  *grad_clip = v_f32_sel(coord_gt_0_le_max, *grad_clip, 1.0f);
  re = v_f32_sel(coord_gt_0_le_max, re, coord);

  return re;
}

// Mapping the out-of-boundary points back into boundary
// This would only affect padding_mode=border or reflection
inline float8_128 compute_coordinates(
    float8_128 coord, const int size, const int padding_mode, const bool align_corners) {
  if (padding_mode == BORDER) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == REFLECTION) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, (size - 1) * 2);
    } else {
      coord = reflect_coordinates(coord, -1, size * 2 - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }
  return coord;
}

inline float8_128 grid_sampler_unnormalize(float8_128 coord, int size, bool align_corners) {
  if (align_corners) { // ((coord + 1) / 2) * (size - 1)
    coord = v_f32_add_b(coord, 1.0f);
    coord = v_f32_mul_b(coord, 0.5f);
    coord = v_f32_mul_b(coord, (float)size - 1.0f);
    return coord;
  } else {
    coord = v_f32_add_b(coord, 1.0f);
    coord = v_f32_mul_b(coord, (float)size);
    coord = v_f32_sub_b(coord, 1.0f);
    coord = v_f32_mul_b(coord, 0.5f);
    return coord;
  }
}

inline float8_128 grid_sampler_unnormalize_set_grad(
    float8_128 coord, int size, bool align_corners, float8_128 *grad_mult) {
  if (align_corners) { // coord = ((coord + 1.0) / 2.0) * (size - 1.0)
    *grad_mult = v_f32_mul_b(((float)size - 1.0f), 0.5f);
    coord = v_f32_mul_b(v_f32_mul_b(v_f32_add_b(coord, 1.0f), 0.5f), ((float)size - 1.0f));
    return coord;
  } else { // coord = ((coord + 1.0) * size - 1.0) / 2.0
    *grad_mult = v_f32_mul_b((float)size, 0.5f);
    coord = v_f32_mul_b(v_f32_sub_b(v_f32_mul_b(v_f32_add_b(coord, 1.0f), (float)size), 1.0f), 0.5f);
    return coord;
  }
}

// Computes the piwel source index value for a grid coordinate
inline float8_128 grid_sampler_compute_source_index(
    float8_128 coord, int size, int padding_mode, bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}

inline float8_128 grid_sampler_compute_source_index_set_grad(
    float8_128 coord, int size, int padding_mode, bool align_corners, float8_128 *grad_mult) {
  float8_128 grad_clip, grad_refl;
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_mult);

  if (padding_mode == BORDER) {
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_mult = (*grad_mult) * grad_clip;
  } else if (padding_mode == REFLECTION) {
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, (size - 1) * 2, &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, size * 2 - 1, &grad_refl);
    }
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_mult = (*grad_mult) * grad_refl * grad_clip;
  }

  return coord;
}

inline int8_128 within_bounds_2d(
    const int8_128 w, const int8_128 h, const int W, const int H) {
  bool8_128 cmp_w_gteq_0 = v_s32_cmp(GTEQ, w, 0);
  bool8_128 cmp_h_gteq_0 = v_s32_cmp(GTEQ, h, 0);
  bool8_128 cmp_w_le_W = v_s32_cmp(LS, w, W);
  bool8_128 cmp_h_le_H = v_s32_cmp(LS, h, H);

  bool8_128 cmp = cmp_w_gteq_0 & cmp_h_gteq_0 & cmp_w_le_W & cmp_h_le_H;
  int8_128 x = v_s32_sel(cmp, -1, w);
  int8_128 y = v_s32_sel(cmp, -1, h);

  return (x + y * ALIGN128(W));
}

inline bool8_128 within_bounds_2d_vmask(
    const int8_128 w, const int8_128 h, const int W, const int H) {
  bool8_128 cmp_w_gteq_0 = v_s32_cmp(GTEQ, w, 0);
  bool8_128 cmp_h_gteq_0 = v_s32_cmp(GTEQ, h, 0);
  bool8_128 cmp_w_le_W = v_s32_cmp(LS, w, W);
  bool8_128 cmp_h_le_H = v_s32_cmp(LS, h, H);

  bool8_128 cmp = cmp_w_gteq_0 & cmp_h_gteq_0 & cmp_w_le_W & cmp_h_le_H;

  return cmp;
}

inline int8_128 within_bounds_3d(
    const int8_128 w, const int8_128 h, const int8_128 d, const int W, const int H, const int D) {
  bool8_128 cmp_w_gteq = v_s32_cmp(GTEQ, w, 0);
  bool8_128 cmp_h_gteq = v_s32_cmp(GTEQ, h, 0);
  bool8_128 cmp_d_gteq = v_s32_cmp(GTEQ, d, 0);
  bool8_128 cmp_w_ls = v_s32_cmp(LS, w, W);
  bool8_128 cmp_h_ls = v_s32_cmp(LS, h, H);
  bool8_128 cmp_d_ls = v_s32_cmp(LS, d, D);

  bool8_128 cmp = cmp_w_gteq & cmp_h_gteq & cmp_d_gteq & cmp_w_ls & cmp_h_ls & cmp_d_ls;

  int8_128 x = v_s32_sel(cmp, -1, w);
  int8_128 y = v_s32_sel(cmp, -1, h);
  int8_128 z = v_s32_sel(cmp, -1, d);

  return (x + y * ALIGN128(W) + z * H * ALIGN128(W));
}

inline bool8_128 within_bounds_3d_vmask(
    const int8_128 w, const int8_128 h, const int8_128 d, const int W, const int H, const int D) {
  bool8_128 cmp_w_gteq = v_s32_cmp(GTEQ, w, 0);
  bool8_128 cmp_h_gteq = v_s32_cmp(GTEQ, h, 0);
  bool8_128 cmp_d_gteq = v_s32_cmp(GTEQ, d, 0);
  bool8_128 cmp_w_ls = v_s32_cmp(LS, w, W);
  bool8_128 cmp_h_ls = v_s32_cmp(LS, h, H);
  bool8_128 cmp_d_ls = v_s32_cmp(LS, d, D);

  bool8_128 cmp = cmp_w_gteq & cmp_h_gteq & cmp_d_gteq & cmp_w_ls & cmp_h_ls & cmp_d_ls;

  return cmp;
}

inline float8_128 select_data_2d(
    const int8_128 x0, const int8_128 y0, const int W, const int H,
    const SIM_X86::tensor input0_vmem, const int VMEMSize) {
  int8_128 idx = within_bounds_2d(x0, y0, W, H);

  float8_128 ans = 0.0f;

  int vs;
  for (vs = 0; vs < VMEMSize; vs += 1024) {
    float8_128 data = v_f32_ld_tnsr_b(0, tensor_slice(input0_vmem, vs / 32));
    
    #pragma unroll 8
    for (int roll = 0; roll < 8; ++roll) {
      int8_128 nidx = idx - vs - 128 * roll;
      bool8_128 nidx_gteq = v_s32_cmp(GTEQ, nidx, 0);
      bool8_128 nidx_ls = v_s32_cmp(LS, nidx, 128);

      float8_128 data_perm = m_f32_perm(data, nidx, 0, 0);
      data_perm = v_f32_sel(nidx_gteq & nidx_ls, 0, data_perm);

      ans += data_perm;
      data = v_row_rotate(data, 0);
    }
  }

  if (vs < VMEMSize) {
    int ldst_msk = pre_exp2((VMEMSize - vs) / 128);
    float8_128 data = v_f32_ld_tnsr_st_msk(0, tensor_slice(input0_vmem, vs / 32), 1, ldst_msk);

    #pragma unroll 7
    for (int roll = 0; roll < (VMEMSize - vs) / 128; ++roll) {
      int8_128 nidx = idx - vs - 128 * roll;
      bool8_128 nidx_gteq = v_s32_cmp(GTEQ, nidx, 0);
      bool8_128 nidx_ls = v_s32_cmp(LS, nidx, 128);

      float8_128 data_perm = m_f32_perm(data, nidx, 0, 0);
      data_perm = v_f32_sel(nidx_gteq & nidx_ls, 0, data_perm);

      ans += data_perm;
      data = v_row_rotate(data, 0);
    }
  }

  return ans;
}

inline float8_128 select_data_2d_bf16(
    const int8_128 x_0, const int8_128 x_1, const int8_128 y_0, const int8_128 y_1,
    const int W, const int H,
    const SIM_X86::tensor input0_vmem, const int VMEMSize) {
  int8_128 idx_0 = within_bounds_2d(x_0, y_0, W, H);
  int8_128 idx_1 = within_bounds_2d(x_1, y_1, W, H);

  float8_128 ans_0 = 0.0f;
  float8_128 ans_1 = 0.0f;

  int vs;
  for (vs = 0; vs < VMEMSize; vs += 1024) {
    float8_128 data = v_f32_ld_tnsr_b(0, tensor_slice(input0_vmem, vs / 32));
    float8_128 data_0 = bfloat16_to_float(unpack_16b(__$S(data), 0));
    float8_128 data_1 = bfloat16_to_float(unpack_16b(__$S(data), 1));
    
    #pragma unroll 8
    for (int roll = 0; roll < 8; ++roll) {
      int8_128 nidx_0 = idx_0 - vs - 128 * roll;
      int8_128 nidx_1 = idx_1 - vs - 128 * roll;
      bool8_128 nidx_gteq_0 = v_s32_cmp(GTEQ, nidx_0, 0);
      bool8_128 nidx_gteq_1 = v_s32_cmp(GTEQ, nidx_1, 0);
      bool8_128 nidx_ls_0 = v_s32_cmp(LS, nidx_0, 128);
      bool8_128 nidx_ls_1 = v_s32_cmp(LS, nidx_1, 128);

      float8_128 data_perm_0 = m_f32_perm(data_0, nidx_0, 0, 0);
      float8_128 data_perm_1 = m_f32_perm(data_1, nidx_1, 0, 0);
      data_perm_0 = v_f32_sel(nidx_gteq_0 & nidx_ls_0, 0, data_perm_0);
      data_perm_1 = v_f32_sel(nidx_gteq_1 & nidx_ls_1, 0, data_perm_1);

      ans_0 += data_perm_0;
      ans_1 += data_perm_1;
      data_0 = v_row_rotate(data_0, 0);
      data_1 = v_row_rotate(data_1, 0);
    }
  }

  if (vs < VMEMSize) {
    int ldst_msk = pre_exp2((VMEMSize - vs) / 128);
    float8_128 data = v_f32_ld_tnsr_st_msk(0, tensor_slice(input0_vmem, vs / 32), 1, ldst_msk);
    float8_128 data_0 = bfloat16_to_float(unpack_16b(__$S(data), 0));
    float8_128 data_1 = bfloat16_to_float(unpack_16b(__$S(data), 1));

    #pragma unroll 7
    for (int roll = 0; roll < (VMEMSize - vs) / 128; ++roll) {
      int8_128 nidx_0 = idx_0 - vs - 128 * roll;
      int8_128 nidx_1 = idx_1 - vs - 128 * roll;
      bool8_128 nidx_gteq_0 = v_s32_cmp(GTEQ, nidx_0, 0);
      bool8_128 nidx_gteq_1 = v_s32_cmp(GTEQ, nidx_1, 0);
      bool8_128 nidx_ls_0 = v_s32_cmp(LS, nidx_0, 128);
      bool8_128 nidx_ls_1 = v_s32_cmp(LS, nidx_1, 128);

      float8_128 data_perm_0 = m_f32_perm(data_0, nidx_0, 0, 0);
      float8_128 data_perm_1 = m_f32_perm(data_1, nidx_1, 0, 0);
      data_perm_0 = v_f32_sel(nidx_gteq_0 & nidx_ls_0, 0, data_perm_0);
      data_perm_1 = v_f32_sel(nidx_gteq_1 & nidx_ls_1, 0, data_perm_1);

      ans_0 += data_perm_0;
      ans_1 += data_perm_1;
      data_0 = v_row_rotate(data_0, 0);
      data_1 = v_row_rotate(data_1, 0);
    }
  }

  return __$F(float_to_bfloat16(ans_1, ans_0));
}

inline float8_128 select_data_3d(
    const int8_128 iw, const int8_128 ih, const int8_128 id,
    const int W, const int H, const int D,
    const SIM_X86::tensor input0_vmem, const int VMEMSize) {
  int8_128 idx = within_bounds_3d(iw, ih, id, W, H, D);

  float8_128 ans = 0.0f;

  int vs;
  for (vs = 0; vs < VMEMSize; vs += 1024) {
    float8_128 data = v_f32_ld_tnsr_b(0, tensor_slice(input0_vmem, vs / 32));
    
    #pragma unroll 8
    for (int roll = 0; roll < 8; ++roll) {
      int8_128 nidx = idx - vs - 128 * roll;
      bool8_128 nidx_gteq = v_s32_cmp(GTEQ, nidx, 0);
      bool8_128 nidx_ls = v_s32_cmp(LS, nidx, 128);

      float8_128 data_perm = m_f32_perm(data, nidx, 0, 0);
      data_perm = v_f32_sel(nidx_gteq & nidx_ls, 0, data_perm);

      ans += data_perm;
      data = v_row_rotate(data, 0);
    }
  }

  if (vs < VMEMSize) {
    int ldst_msk = pre_exp2((VMEMSize - vs) / 128);
    float8_128 data = v_f32_ld_tnsr_st_msk(0, tensor_slice(input0_vmem, vs / 32), 1, ldst_msk);

    #pragma unroll 7
    for (int roll = 0; roll < (VMEMSize - vs) / 128; ++roll) {
      int8_128 nidx = idx - vs - 128 * roll;
      bool8_128 nidx_gteq = v_s32_cmp(GTEQ, nidx, 0);
      bool8_128 nidx_ls = v_s32_cmp(LS, nidx, 128);

      float8_128 data_perm = m_f32_perm(data, nidx, 0, 0);
      data_perm = v_f32_sel(nidx_gteq & nidx_ls, 0, data_perm);

      ans += data_perm;
      data = v_row_rotate(data, 0);
    }
  }

  return ans;
}

inline void update_and_store_data_with_idx(
    const SIM_X86::tensor input0_vmem, const int VMEMSize,
    const float8_128 data, int len, const int8_128 idx) {
  int vs = 0;

  for (vs = 0; vs < VMEMSize / 1024 * 1024; vs += 1024) {
    float8_128 data_in = v_f32_ld_tnsr_b(0, tensor_slice(input0_vmem, vs / 32));

    #pragma unroll 8
    for (int roll = 0; roll < 8; ++roll) {
      int8_128 nidx = idx - vs - roll * 128;

      for (int i = 0; i < len; ++i) {
        int8_128 nnidx = __$S(m_rotate(__$F(nidx), -i, true));
        float8_128 data_rotate = m_rotate(data, -i, false);
        int idx_single = nnidx[0];

        if (idx_single >= 0 && idx_single < 128) {
          float8_128 data_in_rotate = m_rotate(data_in, -idx_single, true);
          data_in_rotate[0] = data_in_rotate[0] + data_rotate[0];
          data_in = m_rotate(data_in_rotate, idx_single, false);
        }
      }

      data_in = v_row_rotate(data_in, 0);
    }

    v_f32_st_tnsr_b(0, tensor_slice(input0_vmem, vs / 32), data_in);
  }

  if (vs < VMEMSize) {
    int ldst_msk = pre_exp2((VMEMSize - vs) / 128);
    float8_128 data_in = v_f32_ld_tnsr_st_msk(0, tensor_slice(input0_vmem, vs / 32), 1, ldst_msk);

    #pragma unroll 7
    for (int roll = 0; roll < (VMEMSize - vs) / 128; ++roll) {
      int8_128 nidx = idx - vs - roll * 128;

      for (int i = 0; i < len; ++i) {
        int8_128 nnidx = __$S(m_rotate(__$F(nidx), -i, true));
        float8_128 data_rotate = m_rotate(data, -i, false);
        int idx_single = nnidx[0];

        if (idx_single >= 0 && idx_single < 128) {
          float8_128 data_in_rotate = m_rotate(data_in, -idx_single, true);
          data_in_rotate[0] = data_in_rotate[0] + data_rotate[0];
          data_in = m_rotate(data_in_rotate, idx_single, false);
        }
      }

      data_in = v_row_rotate(data_in, 0);
    }
    if ((VMEMSize - vs) / 128 <= 4) {
      for (int roll = 0; roll < (VMEMSize - vs) / 128; ++roll) data_in = v_row_rotate(data_in, 1);
    } else {
      for (int roll = 0; roll < 8 - (VMEMSize - vs) / 128; ++roll) data_in = v_row_rotate(data_in, 0);
    }

    v_f32_st_tnsr_st_msk(0, tensor_slice(input0_vmem, vs / 32), 1, ldst_msk, data_in);
  }
}

inline void update_and_store_data_2d(
    const int8_128 iw, const int8_128 ih, const int W, const int H,
    const SIM_X86::tensor input0_vmem, const int VMEMSize,
    const float8_128 data, int len) {
  int8_128 idx = within_bounds_2d(iw, ih, W, H);

  update_and_store_data_with_idx(input0_vmem, VMEMSize, data, len, idx);
}

inline void update_and_store_data_3d(
    const int8_128 iw, const int8_128 ih, const int8_128 id,
    const int W, const int H, const int D,
    const SIM_X86::tensor input0_vmem, const int VMEMSize,
    const float8_128 data, int len) {
  int8_128 idx = within_bounds_3d(iw, ih, id, W, H, D);

  update_and_store_data_with_idx(input0_vmem, VMEMSize, data, len, idx);
}

inline float8_128 get_value_bounded(
    int8_128 ix, int8_128 iy, const int W, const int H,
    const int padding_mode, const bool align_corners,
	  const SIM_X86::tensor input0_vmem, const int VMEMSize) {

	float8_128 fx = compute_coordinates(v_cvt_itof(ix), W, padding_mode, align_corners);
	float8_128 fy = compute_coordinates(v_cvt_itof(iy), H, padding_mode, align_corners);

  ix = __dlc_float2int_rz(fx);
  ix = float2int_clean_invalid(fx, ix);
  iy = __dlc_float2int_rz(fy);
  iy = float2int_clean_invalid(fy, iy);
  
  return select_data_2d(ix, iy, W, H, input0_vmem, VMEMSize);
}

inline float8_128 get_value_bounded_bf16(
    int8_128 ix_0, int8_128 ix_1, int8_128 iy_0, int8_128 iy_1, const int W, const int H,
    const int padding_mode, const bool align_corners,
	  const SIM_X86::tensor input0_vmem, const int VMEMSize) {

	float8_128 fx_0 = compute_coordinates(v_cvt_itof(ix_0), W, padding_mode, align_corners);
	float8_128 fx_1 = compute_coordinates(v_cvt_itof(ix_1), W, padding_mode, align_corners);
	float8_128 fy_0 = compute_coordinates(v_cvt_itof(iy_0), H, padding_mode, align_corners);
	float8_128 fy_1 = compute_coordinates(v_cvt_itof(iy_1), H, padding_mode, align_corners);

  ix_0 = __dlc_float2int_rz(fx_0);
  ix_0 = float2int_clean_invalid(fx_0, ix_0);
  ix_1 = __dlc_float2int_rz(fx_1);
  ix_1 = float2int_clean_invalid(fx_1, ix_1);
  iy_0 = __dlc_float2int_rz(fy_0);
  iy_0 = float2int_clean_invalid(fy_0, iy_0);
  iy_1 = __dlc_float2int_rz(fy_1);
  iy_1 = float2int_clean_invalid(fy_1, iy_1);
  
  return select_data_2d_bf16(ix_0, ix_1, iy_0, iy_1, W, H, input0_vmem, VMEMSize);
}

inline void store_data_2d(
    const int8_128 iw, const int8_128 ih, const int W, const int H,
    const SIM_X86::tensor input0_vmem, const int VMEMSize,
    const float8_128 data, int len) {
  int8_128 idx = within_bounds_2d(iw, ih, W, H);

  for (int vs = 0; vs < VMEMSize; vs += 128) {
    int ldst_msk = 1;

    float8_128 data_in = v_f32_ld_tnsr_st_msk(0, tensor_slice(input0_vmem, vs / 32), 1, ldst_msk);
    int8_128 nidx = idx - vs;

    for (int i = 0; i < len; ++i) {
      int8_128 nnidx = __$S(m_rotate(__$F(nidx), -i, true));
      float8_128 data_rotate = m_rotate(data, -i, false);
      int idx_single = nnidx[0];

      if (idx_single >= 0 && idx_single < 128) {
        float8_128 data_in_rotate = m_rotate(data_in, -idx_single, true);
        data_in_rotate[0] = data_rotate[0];
        data_in = m_rotate(data_in_rotate, idx_single, false);
      }
    }

    v_f32_st_tnsr_st_msk(0, tensor_slice(input0_vmem, vs / 32), 1, 1, data_in);
  }
}
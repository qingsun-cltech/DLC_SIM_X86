#ifndef _DLC_INTRINSICS_H_X86_
#define _DLC_INTRINSICS_H_X86_

#include "typehint.h"

#define kDataWidth 32
#define kExponentMask 0x7f800000
#define kSignificantMask 0x007fffff
#define kSignMask 0x80000000
#define kMaxofSignedInt 0x7fffffff
#define kIntMaxOfFloat 0x4f000000

template <class BF>
inline float8_128 binop(float8_128 a, float8_128 b, BF bf) {
  for (int i = 0; i < 1024; i++) {
    a.data[i] = bf(a.data[i], b.data[i]);
  }
  return a;
}

template <class UF>
inline float8_128 uniop(float8_128 a, UF fn) {
  for (int i = 0; i < 1024; i++) {
    a.data[i] = fn(a.data[i]);
  }
  return a;
}

/* tools */
inline int get_device_id() {
  return dlc_get_device_id();
}

inline void sync_device() {
  dlc_barrier.Wait();
}

inline float8_128 sub_vector(const float128_128& x, const int& idx) {
  float8_128 y;

  assert(idx < 16);
  std::copy_n(x.data.begin() + idx * 1024, 1024, y.data.begin());

  return y;
}

inline int8_128 sub_vector_s32(const int128_128& x, const int& idx) {
  int8_128 y;

  assert(idx < 16);
  std::copy_n(x.data.begin() + idx * 1024, 1024, y.data.begin());

  return y;
}

inline float8_128 sub_vector_32(const float128_128_2& x, const int& idx) {
  float8_128 y;

  assert(idx < 32);
  std::copy_n(x.data.begin() + idx * 1024, 1024, y.data.begin());

  return y;
}

inline SIM_X86::tensor tensor_slice(SIM_X86::tensor t, int64_t off) {
  return t + off;
  // printf("offset = %lld, szie = %lld\n", off,  t.data_size);
  // assert(off * 32 <= t.data_size);
  // t.data_ptr = t.data_ptr + off * 32;
  // t.data_size = t.data_size - off * 32;
  // return t;
  // return SIM_X86::tensor(t.data_ptr + off * 32, t.data_size - off * 32);
}

// inline SIM_X86::tensor tensor_slice(SIM_X86::tensor t, const int& off) {
//   tensor_slice(t, std::size_t(off));
// }

inline SIM_X86::tensor tensor_slice(void* t, const int64_t& off) {
  return tensor_slice(*(SIM_X86::tensor*)t, off);
}

inline void TensorFixDims(SIM_X86::DLCTensor* x) {
  // x->dim0 = x->shape[0];
  // x->dim1 = x->shape[1] * x->shape[2] * x->shape[3] * x->shape[4];
  // SIM_X86::DLCType t = (SIM_X86::DLCType)x->dtype;
  // if (t == SIM_X86::DLCType::dlc_int8 || t == SIM_X86::DLCType::dlc_uint8 || t == SIM_X86::DLCType::dlc_bool) {
  //   x->dim0_padded = (x->dim0 + 511) & 0xfffffe00;
  // } else if (t == SIM_X86::DLCType::dlc_int16 || t == SIM_X86::DLCType::dlc_bf16 || t == SIM_X86::DLCType::dlc_fp16) {
  //   x->dim0_padded = (x->dim0 + 255) & 0xffffff00;
  // } else {
  //   x->dim0_padded = (x->dim0 + 127) & 0xffffff80;
  // }
}

inline void TensorFixDims(std::vector<SIM_X86::DLCTensor>* x) {
  int len = x->size();
  for (int i = 0; i < len; i++) {
    (*x)[i].dim0 = (*x)[i].shape[0];
    (*x)[i].dim1 = (*x)[i].shape[1] * (*x)[i].shape[2] * (*x)[i].shape[3] * (*x)[i].shape[4];
    SIM_X86::DLCType t = (SIM_X86::DLCType)(*x)[i].dtype;
    if (t == SIM_X86::DLCType::dlc_int8 || t == SIM_X86::DLCType::dlc_uint8 || t == SIM_X86::DLCType::dlc_bool) {
      (*x)[i].dim0_padded = ((*x)[i].dim0 + 511) & 0xfffffe00;
    } else if (t == SIM_X86::DLCType::dlc_int16 || t == SIM_X86::DLCType::dlc_bf16 || t == SIM_X86::DLCType::dlc_fp16) {
      (*x)[i].dim0_padded = ((*x)[i].dim0 + 255) & 0xffffff00;
    } else {
      (*x)[i].dim0_padded = ((*x)[i].dim0 + 127) & 0xffffff80;
    }
  }
}




/* dma */
inline int dlc_dma(SIM_X86::tensor src, int, SIM_X86::tensor dst, int, int len, int src_st,
                   int dst_st, int unit_len, int) {
  for (int i = 0, k = 0; i < len; i += unit_len, k++) {
    if (!(k * src_st + unit_len <= src.data_size)) {
      printf("src_type = %d, dst_type = %d, len = %d, src_st = %d, dst_st = %d, unti_len = %d\n", src.type, dst.type, len, src_st, dst_st, unit_len);
    }
    assert(k * src_st + unit_len <= src.data_size && "ERROR: dlc_dma: src_addr out of range");
    if (!(k * dst_st + unit_len <= dst.data_size)) {
      printf("device_id = %d\n", get_device_id());
      printf("len = %d, src_st = %d, dst_st = %d, unit_len = %d\n", len, src_st, dst_st, unit_len);
      printf("__PTR = %p, ptr = %p, __LEN = %ld, size = %ld, type = %d\n", dst.__PTR, dst.data_ptr, dst.__LEN, dst.data_size, dst.type);
    }
    assert(k * dst_st + unit_len <= dst.data_size && "ERROR: dlc_dma: dst_addr out of range");
    std::copy_n(src.data_ptr + k * src_st, unit_len, dst.data_ptr + k * dst_st);
  }

  return 0;
}

inline int dlc_dma_4B(SIM_X86::tensor, int, SIM_X86::tensor, int, int len, int src_st, int dst_st,
                      int unit_len, int) {
  assert(false && "TODO: dlc_dma_4B ...");
  return 0;
}

inline int dlc_dma_memset(int, SIM_X86::tensor, int, int, int, int, int, int) {
  assert(false && "TODO: dlc_dma_memset ...");
  return 0;
}

inline void dlc_sync_clear(int) {
  // we don't simulate this
}

inline void dlc_sync(int) {
  // we don't simulate this
}

inline void dlc_sync_gt(int, int) {
  // assert(false && "unsupport");
}

inline void dlc_sync_gte(int, int) {
  // assert(false && "unsupport");
}

/* ldst */
inline float8_128 v_f32_ld_tnsr_b(const int& offset, const SIM_X86::tensor& vmem) {
  return dlc_v_f32_load_kernel(tensor_slice(vmem, offset), 1, 0b11111111, true);
}

inline float8_128 v_f32_ld_tnsr_st(const int& offset, const SIM_X86::tensor& vmem, 
                                   const int& stride) {
  return dlc_v_f32_load_kernel(tensor_slice(vmem, offset), stride, 0b11111111, true);
}

inline float8_128 v_f32_ld_tnsr_st_msk(const int& offset, const SIM_X86::tensor& vmem,
                                       const int& stride, const int& ldst_mask) {
  return dlc_v_f32_load_kernel(tensor_slice(vmem, offset), stride, ldst_mask, true);
}

inline void v_f32_st_tnsr_b(const int& offset, const SIM_X86::tensor& vmem, float8_128 x) {
  dlc_v_f32_store_kernel(tensor_slice(vmem, offset), 1, 0b11111111, true, x);
}

inline void v_f32_st_tnsr_st(const int& offset, const SIM_X86::tensor& vmem,
                              const int& stride, float8_128 x) {
  dlc_v_f32_store_kernel(tensor_slice(vmem, offset), stride, 0b11111111, true, x);
}

inline void v_f32_st_tnsr_st_msk(const int& offset, const SIM_X86::tensor& vmem,
                                  const int& stride, const int& ldst_mask, float8_128 x) {
  dlc_v_f32_store_kernel(tensor_slice(vmem, offset), stride, ldst_mask, true, x);
}

// offset, base, stride, mask, value
inline int8_128 v_i32_ld_tnsr(const int& offset, const SIM_X86::tensor& vmem, ...) {
  va_list args;
  va_start(args, vmem);

  const int stride = va_arg(args, int);
  const int ldst_mask = va_arg(args, int);

  int8_128 res = dlc_$S(dlc_v_f32_load_kernel(tensor_slice(vmem, offset), stride, ldst_mask, true));

  va_end(args);

  return res;
}

// offset, base, stride, mask, int8_128
inline void v_st_generic(const int& offset, const SIM_X86::tensor& vmem, ...) {
  va_list args;
  va_start(args, vmem);

  const int stride = va_arg(args, int);
  const int ldst_mask = va_arg(args, int);
  const int8_128 x = va_arg(args, int8_128);

  dlc_v_f32_store_kernel(tensor_slice(vmem, offset), stride, ldst_mask, true, dlc_$F(x));

  va_end(args);
}

inline float8_128 v_ld_vmsk(const int& offset, const SIM_X86::tensor& vmem,
                            const int& stride, const int& ldst_mask, const bool8_128& vmask) {
  return dlc_v_f32_load_kernel(tensor_slice(vmem, offset), stride, ldst_mask, vmask);
}

inline void v_st_vmsk(const int& offset, const SIM_X86::tensor& vmem,
                      const int& stride, const int& ldst_mask,
                      const bool8_128& vmask, float8_128 x) {
  dlc_v_f32_store_kernel(tensor_slice(vmem, offset), stride, ldst_mask, vmask, x);
}




/* fxc */
// fxc_load without pop crf
inline void v_fxc_load_single(const int& offset, const SIM_X86::tensor& cmem, const int& stride,
                              const int& ldst_mask) {
  float8_128 val(0); // defalut 0

  std::bitset<16> bank(0);
  std::bitset<8> mask(ldst_mask);
  for (int i = 0; i < 8; ++i) {
    if (mask.test(i)) {
      assert(offset * 32 + i * stride * 128 + 128 <= cmem.data_size &&
             "ERROR: v_fxc_load_single: src_addr out of range");
      assert(!bank.test((i * stride) % 16) && "ERROR: v_fxc_load_single: cmem bank collision");
      std::copy_n(cmem.data_ptr + offset * 32 + i * stride * 128, 128, val.data.begin() + i * 128);
      bank.set((i * stride) % 16, true);
    }
  }

  dlc_m_push_crf(val);
}

// fxc_load + pop_crf
inline float8_128 v_f32_fxc_load(const int& offset, const SIM_X86::tensor& cmem, const int& stride,
                                 const int& ldst_mask) {
  v_fxc_load_single(offset, cmem, stride, ldst_mask);

  return dlc_m_pop_crf();
}

inline void v_f32_fxc_store(const int& offset, const SIM_X86::tensor& cmem, const int& stride,
                            const int& ldst_mask, float8_128 x) {
  std::bitset<16> bank(0);
  std::bitset<8> mask(ldst_mask);
  for (int i = 0; i < 8; ++i) {
    if (mask.test(i)) {
      assert(offset * 32 + i * stride * 128 + 128 <= cmem.data_size &&
             "ERROR: v_f32_fxc_store: dst_addr out of arange");
      assert(!bank.test((i * stride) % 16) && "ERROR: v_f32_fxc_store: cmem bank collision");
      std::copy_n(x.data.begin() + i * 128, 128, cmem.data_ptr + offset * 32 + i * stride * 128);
      bank.set((i * stride) % 16, true);
    }
  }
}

// read fuxi cord result
inline float8_128 m_pop_crf() { return dlc_m_pop_crf(); }




/* matmul */
// matmul, arguments are [value, mode, PGX]
// mode = [0: mul float rounded, 1: mul gsnf, 2: mul gstf]
inline void m_matmul_single(float8_128 x, const int& mode, const bool& select) {
  dlc_m_matmul(x, SIM_X86::RoundFormat::ROUND, select);

  dlc_update_gmr(mode, select);
}

// matmul hi float16, arguments are [value, mode, PGX]
// mode = [0: mul float rounded, 1: mul gsnf, 2: mul gstf]
inline void m_matmul_f16_hi_single(float8_128 x, const int& mode, const bool& select) {
  dlc_m_matmul(x, SIM_X86::RoundFormat::TRUNCATE, select);

  dlc_update_gmr(mode, select);
}

// matmul lo float16, arguments are [value, mode, PGX]
// mode = [0: mul float rounded, 1: mul gsnf, 2: mul gstf]
inline void m_matmul_f16_lo_single(float8_128 x, const int& mode, const bool& select) {
  dlc_m_matmul(x, SIM_X86::RoundFormat::LOWER_ROUND, select);

  dlc_update_gmr(mode, select);
}

// matmul packed bf16, arguments are [value, mode, PGX]
// mode = [0: packed mul float rounded, 1: packed mul gsnf, 2: packed mul gstf]
inline void m_matmul_packed_single(float8_128 x, const int& mode, const bool& select) {
  for (int CASE = 0; CASE < 2; ++CASE) {
    float8_128 y;

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 128; ++j) {
        dlc_dtype val;

        val.f32 = x[CASE * 512 + i * 128 + j];
        val.u32 = (val.u32 & 0xFFFF) << 16;
        y[i * 2 * 128 + j] = val.f32;
        
        val.f32 = x[CASE * 512 + i * 128 + j];
        val.u32 = ((val.u32 >> 16) & 0xFFFF) << 16;
        y[i * 2 * 128 + 128 + j] = val.f32;
      }
    }

    dlc_m_matmul(y, SIM_X86::RoundFormat::TRUNCATE, select);
  }

  dlc_update_gmr(mode, select);
}

// matmul packed int8, arguments are [value, mode, PGX]
// mode = [0: int8 mul float rounded, 1: int8 mul gsnf, 2: int8 mul gstf]
inline void m_matmul_int8_single(float8_128 x, const int& mode, const bool& select) {
  for (int CASE = 0; CASE < 2; ++CASE) {
    float8_128 y;

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 128; ++j) {
        dlc_dtype val;
        uint32_t uval;

        val.f32 = x[CASE * 512 + i * 128 + j];
        uval = val.u32 & 0xFFFF;
        val.u32 = (((uval >> 8) & 0xFF) << 16) + (uval & 0xFF);
        y[i * 2 * 128 + j] = val.f32;
        
        val.f32 = x[CASE * 512 + i * 128 + j];
        uval = (val.u32 >> 16) & 0xFFFF;
        val.u32 = (((uval >> 8) & 0xFF) << 16) + (uval & 0xFF);
        y[i * 2 * 128 + 128 + j] = val.f32;
      }
    }

    dlc_m_matmul_int(y, select);
  }

  dlc_update_gmr(mode, select);
}

// matmul packed lower16 int8, arguments are [value, mode, PGX]
// mode = [0: int8 lo mul float rounded, 1: int8 lo mul gsnf, 2: int8 lo mul gstf]
inline void m_matmul_int8_lo_single(float8_128 x, const int& mode, const bool& select) {
  float8_128 y;

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 128; ++j) {
      dlc_dtype val;

      val.f32 = x[i * 128 + j];
      uint32_t uval = val.u32 & 0xFFFF;
      val.u32 = (((uval >> 8) & 0xFF) << 16) + (uval & 0xFF);
      y[i * 128 + j] = val.f32;
    }
  }

  dlc_m_matmul_int(y, select);

  dlc_update_gmr(mode, select);
}

// fake mul, includes MTI_MUL_GSTF(GSNF)_ROUNDED + MTR_READ_MATRIX_RESULT
// arguments are[value, is transposed, PGX]
// transpose: false->gsnf, true->gstf
inline float8_128 m_fakemul(float8_128 x, const bool& transpose, const bool& select) {
  m_matmul_single(x, (int)transpose + 1, select);

  return dlc_m_pop_mrf(select);
}

inline float8_128 m_matmul_gsnf(float8_128 x, const bool& select) {
  // mode = [0: mul float rounded, 1: mul gsnf, 2: mul gstf]
  m_matmul_single(x, 1, select);

  return dlc_m_pop_mrf(select);
}

inline float8_128 m_matmul_gstf(float8_128 x, const bool& select) {
  // mode = [0: mul float rounded, 1: mul gsnf, 2: mul gstf]
  m_matmul_single(x, 2, select);

  return dlc_m_pop_mrf(select);
}

inline float8_128 m_matmul_dest_8_128_128(float8_128 l, const float128_128& r,
                                          const bool& select) {
  for (int i = 0; i < 16; ++i) {
    float8_128 x = sub_vector(r, i) ;

    dlc_push_gsnf(x, select);
  }

  m_fakemul(l, bool(SIM_X86::FakeMulType::GSNF), select);

  m_matmul_single(l, 0, select);
  
  return dlc_m_pop_mrf(select);
}

inline float8_128 m_matmul_dest_8_128_128_T(float8_128 l, const float128_128& r,
                                            const bool& select) {
  for (int i = 0; i < 16; ++i) {
    float8_128 x = sub_vector(r, i) ;

    dlc_push_gstf(x, select);
  }

  m_fakemul(l, bool(SIM_X86::FakeMulType::GSTF), select);

  m_matmul_single(l, 0, select);
  
  return dlc_m_pop_mrf(select);
}

inline float128_128 m_matmul_dest_128_128_128(const float128_128& l, const float128_128& r,
                                              const bool& select) {
  for (int i = 0; i < 16; ++i) {
    float8_128 x = sub_vector(r, i) ;

    dlc_push_gsnf(x, select);
  }

  m_fakemul(float8_128(0), bool(SIM_X86::FakeMulType::GSNF), select);

  float128_128 res(0);
  for (int i = 0; i < 16; ++i) {
    m_matmul_single(sub_vector(l, i), int(SIM_X86::FakeMulType::GSNF), select);

    float8_128 y = dlc_m_pop_mrf(select);

    std::copy_n(y.data.begin(), 1024, res.data.begin() + i * 1024);
  }

  return res;
}

inline float128_128 m_matmul_dest_128_128_128_T(const float128_128& l, const float128_128& r,
                                                const bool& select) {
  for (int i = 0; i < 16; ++i) {
    float8_128 x = sub_vector(r, i) ;

    dlc_push_gsnf(x, select);
  }

  m_fakemul(float8_128(0), bool(SIM_X86::FakeMulType::GSTF), select);

  float128_128 res(0);
  for (int i = 0; i < 16; ++i) {
    m_matmul_single(sub_vector(l, i), int(SIM_X86::FakeMulType::GSTF), select);

    float8_128 y = dlc_m_pop_mrf(select);

    std::copy_n(y.data.begin(), 1024, res.data.begin() + i * 1024);
  }

  return res;
}

inline void m_matmul_128_128_128_start(float128_128, bool) {
  assert(false && "m_matmul_128_128_128_start: deprecated and no longer in use");
}

inline void m_matmul_128_128_128_T_start(float128_128, bool) {
  assert(false && "m_matmul_128_128_128_T_start: deprecated and no longer in use");
}

inline float128_128 m_matmul_128_128_128_mid(float128_128, float128_128, bool) {
  assert(false && "m_matmul_128_128_128_mid: deprecated and no longer in use");

  return float128_128(0);
}

inline float128_128 m_matmul_128_128_128_T_mid(float128_128, float128_128, bool) {
  assert(false && "m_matmul_128_128_128_T_mid: deprecated and no longer in use");

  return float128_128(0);
}

inline float128_128 m_matmul_128_128_128_end(float128_128, bool) {
  assert(false && "m_matmul_128_128_128_end: deprecated and no longer in use");

  return float128_128(0);
}

inline float128_128 m_matmul_128_128_128_T_end(float128_128, bool) {
  assert(false && "m_matmul_128_128_128_T_end: deprecated and no longer in use");

  return float128_128(0);
}

inline float128_128_2 m_matmul_dest_128_128_256(const float128_128& l,
                                                const float128_128& r1,
                                                const float128_128& r2) {
  float128_128 o1 = m_matmul_dest_128_128_128(l, r1, 0);
  float128_128 o2 = m_matmul_dest_128_128_128(l, r2, 1);

  float128_128_2 res;
  std::copy_n(o1.data.begin(), 128 * 128, res.data.begin());
  std::copy_n(o2.data.begin(), 128 * 128, res.data.begin() + 128 * 128);

  return res;
}

inline float128_128_2 m_matmul_dest_128_128_256_T(const float128_128& l,
                                                  const float128_128& r1,
                                                  const float128_128& r2) {
  float128_128 o1 = m_matmul_dest_128_128_128_T(l, r1, 0);
  float128_128 o2 = m_matmul_dest_128_128_128_T(l, r2, 1);

  float128_128_2 res;
  std::copy_n(o1.data.begin(), 128 * 128, res.data.begin());
  std::copy_n(o2.data.begin(), 128 * 128, res.data.begin() + 128 * 128);

  return res;
}

// pop mrf, argument decides which PGX
inline float8_128 m_pop_mrf(bool select) { return dlc_m_pop_mrf(select); }




/* pushgain */
inline void push_gsnf(float8_128 x, bool select) {
  std::vector<dlc_dtype> y = getDlcDtype(x);
  for (int i = 0; i < 1024; ++i) {
    y[i] = Float32ToFloat16(y[i], SIM_X86::RoundFormat::ROUND);
  }

  float8_128 val;
  for (int i = 0; i < 1024; ++i) {
    val[i] = y[i].f32;
  }

  dlc_push_gsnf(val, select);
}

inline void push_gstf(float8_128 x, bool select) {
  std::vector<dlc_dtype> y = getDlcDtype(x);
  for (int i = 0; i < 1024; ++i) {
    y[i] = Float32ToFloat16(y[i], SIM_X86::RoundFormat::ROUND);
  }

  float8_128 val;
  for (int i = 0; i < 1024; ++i) {
    val[i] = y[i].f32;
  }

  dlc_push_gstf(val, select);
}

// pushgian high float16, with transpose or not
inline void pushgain_hi(float8_128 x, const bool& transpose, const bool& select) {
  std::vector<dlc_dtype> y = getDlcDtype(x);
  for (int i = 0; i < 1024; ++i) {
    y[i] = Float32ToFloat16(y[i], SIM_X86::RoundFormat::TRUNCATE);
  }

  float8_128 val;
  for (int i = 0; i < 1024; ++i) {
    val[i] = y[i].f32;
  }

  if (transpose) {
    dlc_push_gstf(val, select);
  } else {
    dlc_push_gsnf(val, select);
  }
}

// pushgian low float16, with transpose or not
inline void pushgain_lo(float8_128 x, const bool& transpose, const bool& select) {
  std::vector<dlc_dtype> y = getDlcDtype(x);
  for (int i = 0; i < 1024; ++i) {
    y[i] = Float32ToFloat16(y[i], SIM_X86::RoundFormat::LOWER_ROUND);
  }

  float8_128 val;
  for (int i = 0; i < 1024; ++i) {
    val[i] = y[i].f32;
  }

  if (transpose) {
    dlc_push_gstf(val, select);
  } else {
    dlc_push_gsnf(val, select);
  }
}

// pushgian packed, with transpose or not
inline void packed_push(float8_128 x, const bool& transpose, const bool& select) {
  std::vector<dlc_dtype> y = getDlcDtype(x);

  for (int CASE = 0; CASE < 2; ++CASE) {
    float8_128 val;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 128; ++j) {
        dlc_dtype low = y[CASE * 4 * 128 + i * 128 + j];
        low.u32 = (low.u32 & kLeastTwoByteMask);
        dlc_dtype high = y[CASE * 4 * 128 + i * 128 + j];
        high.u32 = ((low.u32 >> kTwoByteLength) & kLeastTwoByteMask) << kTwoByteLength;
        val[i * 2 * 128 + j] = low.f32;
        val[i * 2 * 128 + 128 + j] = high.f32;
      }
    }
    if (transpose) {
      dlc_push_gstf(val, select);
    } else {
      dlc_push_gsnf(val, select);
    }
  }
}




/* transpose */
inline void m_transpose_start(float8_128 x, const int& width, const bool& select) {
  dlc_memorys._transpose_width[get_device_id()][select] = width;

  dlc_m_push_transpose_buffer(x, select, false);
}

inline void m_transpose_mid(float8_128 x, const bool& select) {
  dlc_m_push_transpose_buffer(x, select, false);
}

inline void m_transpose_end(float8_128 x, const bool& select) {
  dlc_m_push_transpose_buffer(x, select, false);

  dlc_m_transpose_to_trf(select);
}

// height*width(width > 8) m_transpose_packed_start
inline void m_transpose_packed_start(float8_128 x, const int& width, const bool& select) {
  dlc_memorys._transpose_width[get_device_id()][select] = width;

  dlc_m_push_transpose_buffer(x, select, true);
}

// continues loading data from register mti_x, after the builtin m_transpose_packed_start
inline void m_transpose_packed_mid(float8_128 x, const bool& select) {
  dlc_m_push_transpose_buffer(x, select, true);
}

// the last instruction of a transpose instruction set
inline void m_transpose_packed_end(float8_128 x, const bool& select) {
  dlc_m_push_transpose_buffer(x, select, true);

  dlc_m_transpose_to_trf(select);
}

inline float128_128 m_transpose_128_128_nws(const float128_128& x, const bool& select) {
  dlc_memorys._transpose_width[get_device_id()][select] = 128;

  for (int CASE = 0; CASE < 16; ++CASE) {
    float8_128 y = sub_vector(x, CASE);

    dlc_m_push_transpose_buffer(y, select, false);
  }

  dlc_m_transpose_to_trf(select);

  float128_128 res;

  for (int i = 0; i < 16; ++i) {
    float8_128 z = dlc_m_pop_trf(select);

    std::copy_n(z.data.begin(), 1024, res.data.begin() + i * 1024);
  }

  return res;
}

/**
 * @brief: nws[select]: set transpose_width = 8 && push x to transpose_buffer
 *         && execute transpose && load result
 */
inline float8_128 m_transpose_8_128_nws(float8_128 x, const bool& select) {
  dlc_memorys._transpose_width[get_device_id()][select] = 8;

  dlc_m_push_transpose_buffer(x, select, false);

  dlc_m_transpose_to_trf(select);

  return dlc_m_pop_trf(select);
}

/**
 * @brief: nws[select]: set transpose_width && push x to transpose_buffer
 */
inline void m_transpose_push(float8_128 x, const int& width, const bool& select) {
  dlc_memorys._transpose_width[get_device_id()][select] = width;

  dlc_m_push_transpose_buffer(x, select, false);
}

/**
 * @brief: nws[select]: execute transpose && push result to trf
 */
inline void m_transpose_execute(const bool& select) { dlc_m_transpose_to_trf(select); }

/**
 * @brief: nws[select]: pop trf
 */
inline float8_128 m_pop_trf(const bool& select) { return dlc_m_pop_trf(select); }




/* permute & Set Register Instruction*/
/**
 * @brief: get the result of permute x by idx, use "select" select nws, "mode" control idx
 */
// arguments are [value, permute reg, mti_select, mode]
// mode {normal = 0, sublanes = 1, bytes = 2}
inline float8_128 m_f32_perm(float8_128 x, const int8_128& idx, const int& select,
                             const int& mode) {
  dlc_m_set_permute(idx, select, mode);

  dlc_m_permute(x, select);

  return dlc_m_pop_trf(select);
}

// Set every byte of the Permute Control Register of Core n,
// subcore x to the byte[6:0] of Core n, subcore0 register mti_x.
inline void m_set_permute(const int8_128& x, const int& select) { dlc_m_set_permute(x, select, 0); }

// permute the input register, specify nws
inline void m_permute(float8_128 x, const int& select) { dlc_m_permute(x, select); }




/* Reduction */
/**
 * vector sum: normal, packed, segement, packed + segement
 * vector max: normal, packed, segement, packed + segement
 * vector min: normal, packed, segement, packed + segement
 * vector max index: normal, packed, segement, packed + segement
 * vector min index: normal, packed, segement, packed + segement
 * rotate: normal, packed
*/




/* misc */

inline float8_128 v_f32_add_b(float8_128 x, float8_128 y) {
  return x + y;
}

inline float8_128 v_f32_mul_b(float8_128 x, float8_128 y) {
  return x * y;
}

inline float8_128 v_f32_sub_b(float8_128 x, float8_128 y) {
  return x - y;
}

inline float8_128 v_f32_sum_b(float8_128 a) {
  for (int i = 0; i < 8; ++i) {
    float m = a.data[i * 128];
    for (int j = 1; j < 128; ++j) {
      m += a.data[i * 128 + j];
    }
    for (int j = 0; j < 128; ++j) {
      a.data[i * 128 + j] = m;
    }
  }

  return a;
}

inline float128_128 v_concat_16(float8_128 a, float8_128 b, float8_128 c,
                                float8_128 d, float8_128 f, float8_128 g,
                                float8_128 h, float8_128 i, float8_128 j,
                                float8_128 k, float8_128 l, float8_128 m,
                                float8_128 n, float8_128 o, float8_128 p,
                                float8_128 q) {
  float128_128 r;
  std::copy_n(a.data.begin(), 1024, r.data.begin() + 0 * 1024);
  std::copy_n(b.data.begin(), 1024, r.data.begin() + 1 * 1024);
  std::copy_n(c.data.begin(), 1024, r.data.begin() + 2 * 1024);
  std::copy_n(d.data.begin(), 1024, r.data.begin() + 3 * 1024);
  std::copy_n(f.data.begin(), 1024, r.data.begin() + 4 * 1024);
  std::copy_n(g.data.begin(), 1024, r.data.begin() + 5 * 1024);
  std::copy_n(h.data.begin(), 1024, r.data.begin() + 6 * 1024);
  std::copy_n(i.data.begin(), 1024, r.data.begin() + 7 * 1024);
  std::copy_n(j.data.begin(), 1024, r.data.begin() + 8 * 1024);
  std::copy_n(k.data.begin(), 1024, r.data.begin() + 9 * 1024);
  std::copy_n(l.data.begin(), 1024, r.data.begin() + 10 * 1024);
  std::copy_n(m.data.begin(), 1024, r.data.begin() + 11 * 1024);
  std::copy_n(n.data.begin(), 1024, r.data.begin() + 12 * 1024);
  std::copy_n(o.data.begin(), 1024, r.data.begin() + 13 * 1024);
  std::copy_n(p.data.begin(), 1024, r.data.begin() + 14 * 1024);
  std::copy_n(q.data.begin(), 1024, r.data.begin() + 15 * 1024);
  return r;
}

inline int128_128 v_concat_16_s32(const int8_128& a, const int8_128& b, const int8_128& c,
                                  const int8_128& d, const int8_128& f, const int8_128& g,
                                  const int8_128& h, const int8_128& i, const int8_128& j,
                                  const int8_128& k, const int8_128& l, const int8_128& m,
                                  const int8_128& n, const int8_128& o, const int8_128& p,
                                  const int8_128& q) {
  int128_128 r;
  std::copy_n(a.data.begin(), 1024, r.data.begin() + 0 * 1024);
  std::copy_n(b.data.begin(), 1024, r.data.begin() + 1 * 1024);
  std::copy_n(c.data.begin(), 1024, r.data.begin() + 2 * 1024);
  std::copy_n(d.data.begin(), 1024, r.data.begin() + 3 * 1024);
  std::copy_n(f.data.begin(), 1024, r.data.begin() + 4 * 1024);
  std::copy_n(g.data.begin(), 1024, r.data.begin() + 5 * 1024);
  std::copy_n(h.data.begin(), 1024, r.data.begin() + 6 * 1024);
  std::copy_n(i.data.begin(), 1024, r.data.begin() + 7 * 1024);
  std::copy_n(j.data.begin(), 1024, r.data.begin() + 8 * 1024);
  std::copy_n(k.data.begin(), 1024, r.data.begin() + 9 * 1024);
  std::copy_n(l.data.begin(), 1024, r.data.begin() + 10 * 1024);
  std::copy_n(m.data.begin(), 1024, r.data.begin() + 11 * 1024);
  std::copy_n(n.data.begin(), 1024, r.data.begin() + 12 * 1024);
  std::copy_n(o.data.begin(), 1024, r.data.begin() + 13 * 1024);
  std::copy_n(p.data.begin(), 1024, r.data.begin() + 14 * 1024);
  std::copy_n(q.data.begin(), 1024, r.data.begin() + 15 * 1024);
  return r;
}

inline unsigned128 v_f32_cmp_grt_b(char, float8_128, float8_128) {
  // assert(false && "TODO: v_f32_cmp_grt_b ...");
  return unsigned128();
}

inline float8_128 v_f32_mul_vb(float8_128 x, float8_128 y,
                               float8_128 income, const bool8_128& predicate,
                               const char& C = '?') {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 128; ++j) {
      if (predicate[i * 128 + j] == 0) {
        income[i * 128 + j] = x[i * 128 + j] * y[i * 128 + j];
      }
    }
  }
  
  return income;
}

inline float8_128 v_f32_exp(float8_128 a) {
  for (int i = 0; i < 1024; ++i) {
    a.data[i] = expf(a.data[i]);
  }
  return a;
}

inline float8_128 v_f32_rcp_b(float8_128 a) {
  for (int i = 0; i < 1024; ++i) {
    a.data[i] = 1.f / a.data[i];
  }
  return a;
}

inline float8_128 v_f32_max_row(float8_128 a) {
  for (int i = 0; i < 8; i++) {
    float m = a.data[i * 128];
    for (int j = 1; j < 128; j++) {
      m = std::max(m, a.data[i * 128 + j]);
    }
    for (int j = 0; j < 128; j++) {
      a.data[i * 128 + j] = m;
    }
  }
  return a;
}

inline float8_128 v_f32_min_row(float8_128 a) {
  for (int i = 0; i < 8; i++) {
    float m = a.data[i * 128];
    for (int j = 1; j < 128; j++) {
      m = std::min(m, a.data[i * 128 + j]);
    }
    for (int j = 0; j < 128; j++) {
      a.data[i * 128 + j] = m;
    }
  }
  return a;
}

inline float8_128 v_u32_move_b(const int& v) { return float8_128(*(float*)(&v)); }

inline int8_128 v_u32_move_i(const int& v) { return int8_128(v); }

inline float8_128 v_f32_log(float8_128 a) {
  for (int i = 0; i < 1024; i++) {
    a.data[i] = log(a.data[i]);
  }
  return a;
}

inline float8_128 v_f32_exp2(float8_128 a) {
  for (int i = 0; i < 1024; i++) {
    a.data[i] = exp2f(a.data[i]);
  }
  return a;
}

inline float8_128 v_f32_min(float8_128 a, float8_128 b) {
  for (int i = 0; i < 1024; ++i) {
    a.data[i] = std::min(a.data[i], b.data[i]);
  }

  return a;
}

inline float8_128 v_f32_max(float8_128 a, float8_128 b) {
  for (int i = 0; i < 1024; ++i) {
    a.data[i] = std::max(a.data[i], b.data[i]);
  }

  return a;
}

inline float128_128 v_f32_add_row_128_128(float128_128 v) {
  for (int i = 0; i < 128; i++) {
    float sum = 0.0f;
    for (int j = 0; j < 128; j++) {
      sum += v.data[i * 128 + j];
    }
    for (int j = 0; j < 128; j++) {
      v.data[i * 128 + j] = sum;
    }
  }
  return v;
}

inline float128_128 v_f32_min_row_128_128(float128_128 v) {
  for (int i = 0; i < 128; i++) {
    float min = v.data[i * 128];
    for (int j = 1; j < 128; j++) {
      min = std::min(min, v.data[i * 128 + j]);
    }
    for (int j = 0; j < 128; j++) {
      v.data[i * 128 + j] = min;
    }
  }
  return v;
}

inline float128_128 v_f32_max_row_128_128(float128_128 v) {
  for (int i = 0; i < 128; i++) {
    float max = v.data[i * 128];
    for (int j = 1; j < 128; j++) {
      max = std::max(max, v.data[i * 128 + j]);
    }
    for (int j = 0; j < 128; j++) {
      v.data[i * 128 + j] = max;
    }
  }
  return v;
}

inline float8_128 v_f32_abs(float8_128 a) {
  for (int i = 0; i < 1024; i++) {
    a.data[i] = fabs(a.data[i]);
  }
  return a;
}

inline float8_128 v_f32_rsqrt(float8_128 a) {
  for (int i = 0; i < 1024; i++) {
    a.data[i] = 1.0f / sqrt(a.data[i]);
  }
  return a;
}

inline float8_128 v_f32_clamp(float8_128 x, float8_128 y) {
  for (int i = 0; i < 1024; ++i) {
    if (y[i] <= 0)
      x[i] = 0.f;
    else if (x[i] < -y[i])
      x[i] = -y[i];
    else if (x[i] > y[i])
      x[i] = y[i];
  }

  return x;
}

inline float8_128 v_f32_sqrt(float8_128 a) {
  for (int i = 0; i < 1024; i++) {
    a.data[i] = sqrt(a.data[i]);
  }
  return a;
}

inline float8_128 v_cvt_itof(const int8_128& a) {
  float8_128 b;
  for (int i = 0; i < 1024; ++i) {
    b.data[i] = a.data[i];
  }
  return b;
}

inline int8_128 v_cvt_ftoi(float8_128 x, float8_128 y) {
  int8_128 res;

  dlc_dtype val;
  for(int i = 0; i < 1024; ++i) {
    val.f32 = x[i];
    int32_t op_y_value = val.u32;
    val.f32 = y[i];
    int32_t op_x_value = val.u32;

    /* if input number is nan of inf */
    if ((op_x_value & kExponentMask) == kExponentMask) {
      if ((op_x_value & kSignificantMask) == 0) {
        if (op_x_value & kSignMask) {
          res[i] = kSignMask;
          continue;
        } else {
          res[i] = kMaxofSignedInt;
          continue;
        }
      } else {
        res[i] = 0;
        continue;
      }
    }

    /* if input number is subnormal */
    if ((op_x_value & kExponentMask) == 0 &&
        (op_x_value & kSignificantMask) != 0) {
      res[i] = 0;
      continue;
    }

    /* if input number is overflow then clamps to 7fffffff */
    float op_x_float = x[i];
    int32_t op_x_floor = 0;
    if (op_x_float >= IntToFloat(kIntMaxOfFloat)) {
      op_x_floor = kMaxofSignedInt;
    } else {
      op_x_floor = static_cast<int32_t>(op_x_float);
    }

    unsigned int unsigned_y_value = static_cast<unsigned int>(op_y_value);
    unsigned int unsigned_x_fixed = ConvertToFixedPointAndShift(op_x_value);

    if(unsigned_x_fixed > unsigned_y_value){
      if(op_x_value > 0) {
        res[i] = op_x_floor + 1;
      } else {
        res[i] = op_x_floor - 1;
      }
    }
    else
    {
      res[i] = op_x_floor;
    }
  }

  return res;
}

inline int8_128 v_cvt_ftoi(float8_128 x, const int& y) {
  return v_cvt_ftoi(x, dlc_$F(int8_128(y)));
}

inline void Print(const char* str) { printf("%s\n", str); }

inline void PrintScalar(const char* str, int v) {
  printf("%s: %d, 0x%x, %f\n", str, v, v, *(float*)(&v));
}

inline void Print(const float& f, const SIM_X86::PrintType& dtype) {
  dlc_dtype val;
  val.f32 = f;

  switch (dtype) {
  case SIM_X86::PrintType::FLOAT:
    printf("%f", val.f32);
    break;
  case SIM_X86::PrintType::INT:
    printf("%d", val.u32);
    break;
  case SIM_X86::PrintType::HEX:
    printf("%x", val.u32);
    break;
  case SIM_X86::PrintType::BIT:
    for (int i = 31; i >= 0; --i) {
      printf("%c", ((val.u32 >> i) & 1) ? '1' : '0');
    }
    break;
  default:
    printf("ERROE: Print: dtype = %d\n", int(dtype));
    break;
  }

  printf(" ");
}

// dtype: 0->float, 1->int, 2->hex, 3->bit
inline void Print(const char* str, float8_128 v, SIM_X86::PrintType dtype, bool vertical = false) {
  printf("[XYS%d]: %s\n", dlc_get_device_id(), str);

  if (vertical) {
    for (int j = 0; j < 128; ++j) {
      for (int i = 0; i < 8; ++i) {
        Print(v[i * 128 + j], dtype);
      }
      printf("\n");
    }
  } else {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 128; ++j) {
        Print(v[i * 128 + j], dtype);
        if (j % 8 == 7) {
          printf("\n");
        }
      }
    }
  }
}

inline void Print(const char* str, const int8_128& v, SIM_X86::PrintType dtype, bool vertical = false) {
  Print(str, dlc_$F(v), dtype, vertical);
}

// inline void Print(const char* str, const bool8_128& v, SIM_X86::PrintType dtype) {
//   Print(str, dlc_$F(int8_128(v)), dtype);
// }

inline void Print(const char* str, SIM_X86::tensor tensor, int len, SIM_X86::PrintType dtype) {
  printf("[XYS%d]: %s\n", dlc_get_device_id(), str);

  for (int i = 0; i < len; ++i) {
    Print(tensor.data_ptr[i], dtype);
    if (i % 8 == 7) {
      printf("\n");
    }
  }
}

inline void PrintVector(const char* str, float8_128 v) {
  printf("%s:\n", str);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 128; j++) {
      printf("%.5e, ", v.data[i * 128 + j]);
    }
    puts("");
  }
}

inline void PrintVector(const char* str, float128_128 v) {
  printf("%s:\n", str);
  for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 128; j++) {
      printf("%.5e, ", v.data[i * 128 + j]);
    }
    puts("");
  }
}

inline void PrintTensor(const char* str, SIM_X86::tensor t, int, int len) {
  printf("%s:\n", str);
  if (int(sqrt(len)) * int(sqrt(len)) == len) {
    int w = int(sqrt(len));
    for (int i = 0; i < w; i++) {
      for (int j = 0; j < w; j++) {
        printf("%.5e, ", t.data_ptr[i * w + j]);
      }
      puts("");
    }
  } else {
    for (int i = 0; i < len; i++) {
      printf("%.5e, ", t.data_ptr[i]);
    }
  }
}

inline float8_128 v_cvt_itof_i(const int& v) { return float8_128(v); }

inline void dlc_s_delay(int) {
  // we don't simulate this
}

inline float128_128_2 v_concat_32(float8_128 a, float8_128 b, float8_128 c,
                                  float8_128 d, float8_128 e, float8_128 f,
                                  float8_128 g, float8_128 h, float8_128 i,
                                  float8_128 j, float8_128 k, float8_128 l,
                                  float8_128 m, float8_128 n, float8_128 o,
                                  float8_128 p, float8_128 a2, float8_128 b2,
                                  float8_128 c2, float8_128 d2, float8_128 e2,
                                  float8_128 f2, float8_128 g2, float8_128 h2,
                                  float8_128 i2, float8_128 j2, float8_128 k2,
                                  float8_128 l2, float8_128 m2, float8_128 n2,
                                  float8_128 o2, float8_128 p2) {
  float128_128_2 r;
  std::copy_n(a.data.begin(), 1024, r.data.begin() + 0 * 1024);
  std::copy_n(b.data.begin(), 1024, r.data.begin() + 1 * 1024);
  std::copy_n(c.data.begin(), 1024, r.data.begin() + 2 * 1024);
  std::copy_n(d.data.begin(), 1024, r.data.begin() + 3 * 1024);
  std::copy_n(e.data.begin(), 1024, r.data.begin() + 4 * 1024);
  std::copy_n(f.data.begin(), 1024, r.data.begin() + 5 * 1024);
  std::copy_n(g.data.begin(), 1024, r.data.begin() + 6 * 1024);
  std::copy_n(h.data.begin(), 1024, r.data.begin() + 7 * 1024);
  std::copy_n(i.data.begin(), 1024, r.data.begin() + 8 * 1024);
  std::copy_n(j.data.begin(), 1024, r.data.begin() + 9 * 1024);
  std::copy_n(k.data.begin(), 1024, r.data.begin() + 10 * 1024);
  std::copy_n(l.data.begin(), 1024, r.data.begin() + 11 * 1024);
  std::copy_n(m.data.begin(), 1024, r.data.begin() + 12 * 1024);
  std::copy_n(n.data.begin(), 1024, r.data.begin() + 13 * 1024);
  std::copy_n(o.data.begin(), 1024, r.data.begin() + 14 * 1024);
  std::copy_n(p.data.begin(), 1024, r.data.begin() + 15 * 1024);
  std::copy_n(a2.data.begin(), 1024, r.data.begin() + 16 * 1024);
  std::copy_n(b2.data.begin(), 1024, r.data.begin() + 17 * 1024);
  std::copy_n(c2.data.begin(), 1024, r.data.begin() + 18 * 1024);
  std::copy_n(d2.data.begin(), 1024, r.data.begin() + 19 * 1024);
  std::copy_n(e2.data.begin(), 1024, r.data.begin() + 20 * 1024);
  std::copy_n(f2.data.begin(), 1024, r.data.begin() + 21 * 1024);
  std::copy_n(g2.data.begin(), 1024, r.data.begin() + 22 * 1024);
  std::copy_n(h2.data.begin(), 1024, r.data.begin() + 23 * 1024);
  std::copy_n(i2.data.begin(), 1024, r.data.begin() + 24 * 1024);
  std::copy_n(j2.data.begin(), 1024, r.data.begin() + 25 * 1024);
  std::copy_n(k2.data.begin(), 1024, r.data.begin() + 26 * 1024);
  std::copy_n(l2.data.begin(), 1024, r.data.begin() + 27 * 1024);
  std::copy_n(m2.data.begin(), 1024, r.data.begin() + 28 * 1024);
  std::copy_n(n2.data.begin(), 1024, r.data.begin() + 29 * 1024);
  std::copy_n(o2.data.begin(), 1024, r.data.begin() + 30 * 1024);
  std::copy_n(p2.data.begin(), 1024, r.data.begin() + 31 * 1024);
  return r;
}

inline bool8_128 v_s32_eq(const int8_128& x, const int8_128& y) {
  bool8_128 z;
  for (int i = 0; i < 1024; ++i) {
    z.data[i] = (x.data[i] == y.data[i]);
  }
  return z;
}

inline bool8_128 v_f32_eq(float8_128 a, float8_128& b) {
  bool8_128 m;
  for (int i = 0; i < 1024; ++i) {
    m.data[i] = (a.data[i] == b.data[i]);
  }
  return m;
}

inline int8_128 v_s32_sel(const bool8_128& m, int8_128 a, const int8_128& b) {
  for (int i = 0; i < 1024; ++i) {
    a.data[i] = (m.data[i] ? b.data[i] : a.data[i]);
  }
  return a;
}

inline float8_128 v_f32_sel(const bool8_128& m, float8_128 a, float8_128 b) {
  for (int i = 0; i < 1024; ++i) {
    a.data[i] = (m.data[i] ? b.data[i] : a.data[i]);
  }
  return a;
}

inline int8_128 get_core_id() {
  int8_128 v;

  for (int i = 0; i < 1024; ++i) {
    v.data[i] = i;
  }

  return v;
}

inline float8_128 v_row_rotate(float8_128 x, int d) {
  float8_128 y;
  d = 1 - d * 2;

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 128; j++) {
      y[i * 128 + j] = x[(i + d + 8) % 8 * 128 + j];
    }
  }

  return y;
}

// logical shift right
inline int8_128 v_u32_shr(int8_128 a, const int8_128& b) {
  dlc_dtype val;

  for (int i = 0; i < 1024; ++i) {
    val.u32 = a[i];

    if (b[i] < 0 || b[i] > 31) val.u32 = 0;
    else val.u32 >>= b[i];

    a[i] = int(val.u32);
  }

  return a;
}

// logical shift left
inline int8_128 v_u32_shl(int8_128 a, int8_128 b) {
  dlc_dtype val;

  for (int i = 0; i < 1024; ++i) {
    val.u32 = a[i];

    if (b[i] < 0 || b[i] > 31) val.u32 = 0;
    else val.u32 <<= b[i];

    a[i] = int(val.u32);
  }

  return a;
}

// round arithmetic shift right
inline int8_128 v_s32_shrar(int8_128 x, const int8_128& y) {
  for (int i = 0; i < 1024; ++i) {
    if (y[i] < 0 || y[i] > 31) {
      x[i] = 0;
    } else {
      int32_t inc = (1 << y[i]);
      int32_t need_carry = (1 << (y[i] - 1));
      int32_t msk = (1 << y[i]) - 1;
      if (x[i] & need_carry) {
        if ((x[i] & msk) == need_carry) {
          if (x[i] & inc) {
            x[i] = (x[i] + inc) >> y[i];
          } else {
            x[i] = x[i] >> y[i];
          }
        } else {
          x[i] = (x[i] + inc) >> y[i];
        }
      } else {
        x[i] = x[i] >> y[i];
      }
    }
  }

  return x;
}

inline int8_128 v_u32_and(const int8_128& a, const int8_128& b) {
  return a & b;
}

inline int8_128 v_s32_add(const int8_128& a, const int8_128& b) {
  return a + b;
}

inline bool8_128 v_s32_cmp(int op, const int8_128& a, const int8_128& b) {
  enum {
    EQ = 0,
    NEQ = 1,
    LS = 2,
    GT = 3,
    LSEQ = 4,
    GTEQ = 5
  };
  bool8_128 res;
  for (int i = 0; i < 1024; ++i) {
    switch (op) {
      case EQ:
        res.data[i] = (a.data[i] == b.data[i]);
        break;
      case NEQ:
        res.data[i] = (a.data[i] != b.data[i]);
        break;
      case LS:
        res.data[i] = (a.data[i] < b.data[i]);
        break;
      case GT:
        res.data[i] = (a.data[i] > b.data[i]);
        break;
      case LSEQ:
        res.data[i] = (a.data[i] <= b.data[i]);
        break;
      case GTEQ:
        res.data[i] = (a.data[i] >= b.data[i]);
        break;
      default:
        break;
    }
  }
  return res;
}

inline bool8_128 v_f32_cmp(int op, float8_128 a, float8_128 b) {
  enum {
    EQ = 0,
    NEQ = 1,
    LS = 2,
    GT = 3,
    LSEQ = 4,
    GTEQ = 5
  };
  bool8_128 res;
  for (int i = 0; i < 1024; ++i) {
    switch (op) {
      case EQ:
        res.data[i] = (a.data[i] == b.data[i]);
        break;
      case NEQ:
        res.data[i] = (a.data[i] != b.data[i]);
        break;
      case LS:
        res.data[i] = (a.data[i] < b.data[i]);
        break;
      case GT:
        res.data[i] = (a.data[i] > b.data[i]);
        break;
      case LSEQ:
        res.data[i] = (a.data[i] <= b.data[i]);
        break;
      case GTEQ:
        res.data[i] = (a.data[i] >= b.data[i]);
        break;
      default:
        break;
    }
  }
  return res;
}

inline int8_128 v_u32_xor(int8_128 a, const int8_128& b) {
  return a ^ b;
  // for (int i = 0; i < 1024; ++i) {
  //   a.data[i] ^= b.data[i];
  // }
  // return a;
}

// m_rotate with out pop, you have to use m_pop_trf to get result
inline void m_rotate_single(float8_128 x, const int& shift, const bool& select) {
  float8_128 y;

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 128; ++j) {
      // y.data[i * 128 + (j + shift) % 128] = x.data[i * 128 + j];
      y[i * 128 + j] = x[i * 128 + (j - shift + 128) % 128];
    }
  }

  dlc_m_push_trf(select, y);
}

// m_rotate and pop
inline float8_128 m_rotate(float8_128 x, const int& shift, const bool& select) {
  m_rotate_single(x, shift, select);

  return dlc_m_pop_trf(select);
}

inline bool8_128 set_vmask(const int8_128& a) { return v_s32_eq(a, 1); }

inline int8_128 v_s32_add_non_clamp(int8_128 x, const int8_128& y) {
  for (int i = 0; i < 1024; ++i) {
    x[i] += y[i];
  }

  return x;
}

inline int8_128 v_genrng(int8_128) {
  assert(false && "TODO: v_genrng ...");
  return int8_128();
}

inline int8_128 v_genrng_01(int8_128) {
  assert(false && "TODO: v_genrng_01 ...");
  return int8_128();
}

inline float8_128 v_u32_move_f(const float& v) { return float8_128(v); }

inline long long s_read_localclock() {
  assert(false && "TODO: s_read_localclock ...");
  return 0ll;
}

inline long long s_read_localclock_end(long long) {
  assert(false && "TODO: s_read_localclock_end ...");
  return 0ll;
}

inline int8_128 pack_16b(const short8_128& x, const short8_128& y) {
  int8_128 result;

  for (int i = 0; i < 1024; ++i) {
    result.data[i] = (int(x.data[i]) << 16) | y.data[i];
  }

  return result;
}

inline short8_128 unpack_16b(const int8_128& x, const int& index) {
  short8_128 result;

  for (int i = 0; i < 1024; ++i) {
    result.data[i] = short((x.data[i] >> (index * 16)) & 0xFFFF);
  }

  return result;
}

// result = (x << 8) | y
inline short8_128 pack_8b(const char8_128& x, const char8_128& y) {
  short8_128 result;

  for (int i = 0; i < 1024; ++i) {
    result.data[i] = (short(x.data[i]) << 8) | y.data[i];
  }

  return result;
}

inline char8_128 unpack_8b(const short8_128& x, const int& index) {
  char8_128 result;

  for (int i = 0; i < 1024; ++i) {
    result.data[i] = char((x.data[i] >> (index * 8)) & 0xFF);
  }

  return result;
}

// result = (a << 24) | (b << 16) | (c << 8) | d
inline int8_128 pack_8_quad(const char8_128& x0, const char8_128& x1, const char8_128& x2,
                            const char8_128& x3) {
  int8_128 result;

  for (int i = 0; i < 1024; ++i) {
    result.data[i] = (int(x0.data[i]) << 24) | (int(x1.data[i]) << 16) | (int(x2.data[i]) << 8) |
                     int(x3.data[i]);
  }

  return result;
}

// index: 0->d, 1->c, 2->b, 3->a
inline char8_128 unpack_8_quad(const int8_128& x, const int& index) {
  char8_128 result;

  for (int i = 0; i < 1024; ++i) {
    result.data[i] = char((x.data[i] >> (index * 8)) & 0xFF);
  }

  return result;
}

inline int8_128 int_to_int16(int8_128 x, const int8_128& y) {
  for (int i = 0; i < 1024; ++i) {
    x.data[i] = ((x.data[i] && 0xFFFF) << 16) | (y.data[i] & 0xFFFF);
  }

  return x;
}

inline int8_128 int16_to_int(const char8_128& x) {
  int8_128 result;

  for (int i = 0; i < 1024; ++i) {
    result.data[i] = int(x.data[i]);
  }

  return result;
}

/**
 * TODO: change rules ?
 */
inline int8_128 float_to_bfloat16(float8_128 x, float8_128 y) {
  int8_128 val = 0;

  for (int i = 0; i < 1024; ++i) {
    dlc_dtype dd;
    dd.f32 = x[i];
    uint32_t a = Float32ToFloat16(dd, SIM_X86::RoundFormat::ROUND).u32;
    dd.f32 = y[i];
    uint32_t b = Float32ToFloat16(dd, SIM_X86::RoundFormat::ROUND).u32;
    val[i] = (a & 0xFFFF0000) | ((b >> 16) & 0x0000FFFF);
  }

  return val;
}

inline float8_128 bfloat16_to_float(const short8_128& x) {
  float8_128 result;

  for (int i = 0; i < 1024; ++i) {
    int y = int(x.data[i]) << 16;
    result.data[i] = reinterpret_cast<float&>(y);
  }

  return result;
}

inline short8_128 int16_to_int8(short8_128 x, const short8_128& y) {
  for (int i = 0; i < 1024; ++i) {
    x.data[i] = ((x.data[i] & 0xFF) << 8) | (y.data[i] & 0xFF);
  }

  return x;
}

inline short8_128 int8_to_int16(const char8_128& x) {
  short8_128 result;

  for (int i = 0; i < 1024; ++i) {
    result.data[i] = short(x.data[i]);
  }

  return result;
}

inline float8_128 v_f32_tanh(float8_128 x) {
  for (int i = 0; i < 1024; ++i) {
    x.data[i] = tanh(x.data[i]);
  }

  return x;
}

inline float vstore_wait(float8_128) {
  // assert(false && "TODO: vstore_wait ...");
  return 0.f;
}

// Get param index, only available in main function for now
// It should be called at the start of main function
// TensorInfo is not supported now
inline int get_param_idx(...) {
  // assert(false && "TODO: get_param_idx ...");
  return 0;
}

// return bool8_128 produced by dlc setvmsk
inline bool8_128 v_set_vmask(const int& x) {
  bool8_128 y(false);

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < std::min(128, x); ++j) {
      y[i * 128 + j] = true;
    }
  }
  
  return y;
}

// cast a float to bfloat16
inline __bf16 float2bfloat16(float x) {
  int y = reinterpret_cast<int&>(x);

  return __bf16(y >> 16);
}

// cast a bfloat16 to float
inline float bfloat162float(const __bf16& x) {
  int y = int(x);

  return reinterpret_cast<float&>(y);
}

// cast a bfloat16 to int
// int bfloat162int(const __bf16) {
//     return int()
//     // assert(false && "TODO: bfloat162int ...");
//     return 0;
// }

// cast a int to bfloat16
inline __bf16 int2bfloat16(int) {
  // assert(false && "TODO: int2bfloat16 ...");
  return __bf16(0.f);
}

// input a int8_128, return the count of leading zeros
inline int8_128 v_u32_clz(int8_128 x) {
  for (int i = 0; i < 1024; ++i) {
    int cnt = 0;
    for (int j = 31; j >= 0; --j) {
      if (!(x[i] & (1 << j))) {
        ++cnt;
      } else {
        break;
      }
    }
    x[i] = cnt;
  }

  return x;
}

// get Exponent of float
inline int8_128 v_s32_exte(int8_128 x) {
  for (int i = 0; i < 1024; ++i) {
    x[i] = x[i] & 0x7f800000;
  }

  return x;
}

// get Significand / Mantissa of float
inline int8_128 v_s32_exts(int8_128 x) {
  for (int i = 0; i < 1024; ++i) {
    x[i] = x[i] & 0x7fffff;
  }

  return x;
}

// halts the program and stops the PC
inline void dlc_s_halt() {
  // assert(false && "TODO: dlc_s_halt ...");
}

// clear result fifo
inline void m_clear_fifo(int misc_op) {
  switch (misc_op) {
    case 0:
      while (dlc_memorys.urf[get_device_id()].size()) {
        dlc_m_pop_urf();
        // dlc_memorys.urf[get_device_id()].pop_front();
      }
      break;
    case 1:
      while (dlc_memorys.mrf[get_device_id()][0].size()) {
        dlc_m_pop_mrf(0);
        // dlc_memorys.mrf[get_device_id()][0].pop_front();
      }
      break;
    case 2:
      while (dlc_memorys.mrf[get_device_id()][1].size()) {
        dlc_m_pop_mrf(1);
        // dlc_memorys.mrf[get_device_id()][1].pop_front();
      }
      break;
    case 3:
      while (dlc_memorys.trf[get_device_id()][0].size()) {
        dlc_m_pop_trf(0);
        // dlc_memorys.trf[get_device_id()][0].pop_front();
      }
      break;
    case 4:
      while (dlc_memorys.trf[get_device_id()][1].size()) {
        dlc_m_pop_trf(1);
        // dlc_memorys.trf[get_device_id()][1].pop_front();
      }
      break;
    default:
      break;
  }
}

// generates reduce_sum with mti_select, without pop trf
inline void m_sum_single(float8_128 x, bool select) {
  for (int i = 0; i < 8; ++i) {
    int sum = x[i * 128];
    for (int j = 1; j < 128; ++j) {
      sum += x[i * 128 + j];
    }
    for (int j = 0; j < 128; ++i) {
      x[i * 128 + j] = sum;
    }
  }

  dlc_memorys.trf[get_device_id()][select].push_back(x);
}

// input a float32 vector, returns if it is inf or nan
inline bool8_128 v_f32_infnan(float8_128 x) {
  bool8_128 result;
  dlc_dtype dd;

  for (int i = 0; i < 1024; ++i) {
    float val = x.data[i];
    dd.f32 = val;
    int y = int(dd.u32);
  
    result[i] = ((y & 0x7f800000) == 0x7f800000);
  }

  return result;
}

// return the carry bit of the unsigned integer addition
inline bool8_128 v_u32_carry(const int8_128& x, const int8_128& y) {
  bool8_128 result;

  for (int i = 0; i < 1024; ++i) {
    uint32_t temp_x = static_cast<uint32_t>(x[i]);
    uint32_t temp_y = static_cast<uint32_t>(y[i]);
    if (uint64_t{temp_x} + uint64_t{temp_y} > 0xFFFFFFFF) {
      result[i] = true;
    } else {
      result[i] = false;
    }
  }

  return result;
}

// this would directly go to mul instruction in our isa
// the symbolic "*" would be lowered to a series of instructions to support smul
inline int s_u32_mul(const int& x, const int& y) { return x * y; }

inline std::pair<int, uint32_t> FixedPointToSignificant(int32_t data) {
  unsigned int sign, unsigned_number;
  sign = kSignMask & data;
  unsigned_number = abs(data);
  int32_t shift_number = 0;
  for (int i = kDataWidth - 1; i >= 0; i--) {
    if (!(unsigned_number & ((1 << i)))) {
      shift_number++;
    } else {
      break;
    }
  }
  // case that the integer part is more than one, shift right.
  if (shift_number <= 8) {
    unsigned_number = unsigned_number >> (8 - shift_number);
    unsigned_number = unsigned_number & kSignificantMask;
  }
  // case that the integer part is smaller than one.
  else {
    unsigned_number = unsigned_number << (shift_number - 8);
    unsigned_number = unsigned_number & kSignificantMask;
  }
  unsigned_number = unsigned_number | sign;
  return std::make_pair((shift_number - 8), unsigned_number);
}

// Take 1st param as exponent and 2nd as the significand, return the composed float vector.
inline float8_128 v_f32_compose(const int8_128& x, const int8_128& y) {
  float8_128 result;

  for (int i = 0; i < 1024; ++i) {
    if (y.data[i] == 0) {
      result[i] = 0;
      continue;
    }
    std::pair<int, uint32_t> holder = FixedPointToSignificant(y.data[i]);
    int x_new = x.data[i] - holder.first;
    if (x_new <= -127) {
      if (y.data[i] < 0) {
        result[i] = 0x80000000;
      } else {
        result[i] = 0x0;
      }
    } else if (x_new == 127) {
      if (y.data[i] < 0) {
        result[i] = 0xffc00000;
      } else {
        result[i] = 0x7fc00000;
      }
    } else if (x_new >= 128) {
      if (y.data[i] < 0) {
        result[i] = 0xff800000;
      } else {
        result[i] = 0x7f800000;
      }
    } else {
      uint32_t exponent = static_cast<uint32_t>(x_new + 127);
      exponent = exponent << 23;
      result[i] = exponent | holder.second;
    }
  }

  return result;
}

#endif
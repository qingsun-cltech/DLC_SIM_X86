#ifndef _X86_H_
#define _X86_H_

#include "align.h"
#include "bf16.h"
// #include "chunk.h"
#include "dma.h"
#include "ldst.h"
#include "math.h"
#include "matmul_t.h"
// #include "mm_t.h"
#include "nn.h"
#include "libdevice.h"
// #include "permute.h"
#include "pingpong.h"

#include "dlc-intrinsics.h"
#include "typehint.h"

int sharedVariable = 0;
std::mutex sharedMutex;
SIM_X86::DLC_MEMORYS dlc_memorys = SIM_X86::DLC_MEMORYS();
SIM_X86::Barrier dlc_barrier(2);

std::condition_variable barrier_cond_;
std::mutex barrier_mutex;
int barrier_thread_count = 2;
int barrier_count = 0;
int barrier_waiting = 0;

template<typename Func, typename... Args>
void invokeFunction(Func&& func, Args&&... args) {
  func(std::forward<Args>(args)...);
}

inline void Tensor2Vector32(const syn::nn::Tensor& input, float* hbm) {
  for (unsigned d4 = 0; d4 < input.size(4); ++d4) {
    for (unsigned d3 = 0; d3 < input.size(3); ++d3) {
      for (unsigned d2 = 0; d2 < input.size(2); ++d2) {
        for (unsigned d1 = 0; d1 < input.size(1); ++d1) {
          for (unsigned d0 = 0; d0 < input.size(0); ++d0) {
            int d0_128 = (input.size(0) + 127) / 128 * 128;
            int offset = d4 * input.size(3) * input.size(2) * input.size(1) * d0_128 +
                         d3 * input.size(2) * input.size(1) * d0_128 +
                         d2 * input.size(1) * d0_128 +
                         d1 * d0_128 +
                         d0;
            if (input.dtype() == dlc_fp32) {
              hbm[offset] = input.get_double({d0, d1, d2, d3, d4});
            } else {
              dlc_dtype val;
              val.u32 = input.get_long({d0, d1, d2, d3, d4});
              hbm[offset] = val.f32;
            }
          }
        }
      }
    }
  }
}

inline void Tensor2Vector16(const syn::nn::Tensor& input, float* hbm) {
  for (unsigned d4 = 0; d4 < input.size(4); ++d4) {
    for (unsigned d3 = 0; d3 < input.size(3); ++d3) {
      for (unsigned d2 = 0; d2 < input.size(2); ++d2) {
        for (unsigned d1 = 0; d1 < input.size(1); ++d1) {
          unsigned group_size = (input.size(0) + 255) / 256;
          for (unsigned i = 0; i < group_size; ++i) {
            for (unsigned j = 0; j < 128; ++j) {
              unsigned d0 = i * 256 + j;
              unsigned dx = d0 + 128;

              if (d0 >= input.size(0)) { break; }
              uint32_t x = 0, y = 0;
              if (input.dtype() == dlc_bf16) {
                float x_f = input.get_double({d0, d1, d2, d3, d4});
                x = *reinterpret_cast<uint32_t*>(&x_f);

                if (dx < input.size(0)) {
                  float y_f = input.get_double({dx, d1, d2, d3, d4});
                  y = *reinterpret_cast<uint32_t*>(&y_f);
                }
              } else {
                x = input.get_long({d0, d1, d2, d3, d4});
                y = (dx < input.size(0) ? input.get_long({dx, d1, d2, d3, d4}) : 0);
              }
              uint32_t val_f = (y & 0xFFFF0000) | ((x >> 16) & 0x0000FFFF);
              float val = *reinterpret_cast<float*>(&val_f);

              int d0_128 = (input.size(0) + 255) / 256 * 256 / 2;
              int offset = d4 * input.size(3) * input.size(2) * input.size(1) * d0_128 +
                           d3 * input.size(2) * input.size(1) * d0_128 +
                           d2 * input.size(1) * d0_128 +
                           d1 * d0_128 +
                           i * 128 + j;
              hbm[offset] = val;
            }
          }
        }
      }
    }
  }
}

inline void Vector2Tensor32(float* hbm, syn::nn::Tensor& input) {
  for (unsigned d4 = 0; d4 < input.size(4); ++d4) {
    for (unsigned d3 = 0; d3 < input.size(3); ++d3) {
      for (unsigned d2 = 0; d2 < input.size(2); ++d2) {
        for (unsigned d1 = 0; d1 < input.size(1); ++d1) {
          for (unsigned d0 = 0; d0 < input.size(0); ++d0) {
            int d0_128 = (input.size(0) + 127) / 128 * 128;
            int offset = d4 * input.size(3) * input.size(2) * input.size(1) * d0_128 +
                         d3 * input.size(2) * input.size(1) * d0_128 +
                         d2 * input.size(1) * d0_128 +
                         d1 * d0_128 +
                         d0;
            if (input.dtype() == dlc_fp32) {
              input.set_double({d0, d1, d2, d3, d4}, hbm[offset]);
            } else {
              dlc_dtype val;
              val.f32 = hbm[offset];
              input.set_long({d0, d1, d2, d3, d4}, val.u32);
            }
          }
        }
      }
    }
  }
}

inline void Vector2Tensor16(float* hbm, syn::nn::Tensor& input) {
  for (unsigned d4 = 0; d4 < input.size(4); ++d4) {
    for (unsigned d3 = 0; d3 < input.size(3); ++d3) {
      for (unsigned d2 = 0; d2 < input.size(2); ++d2) {
        for (unsigned d1 = 0; d1 < input.size(1); ++d1) {
          unsigned group_size = (input.size(0) + 255) / 256;
          for (unsigned i = 0; i < group_size; ++i) {
            for (unsigned j = 0; j < 128; ++j) {
              if (j >= (input.size(0) - i * 256)) break;
              int d0_128 = (input.size(0) + 255) / 256 * 256 / 2;
              int offset = d4 * input.size(3) * input.size(2) * input.size(1) * d0_128 +
                           d3 * input.size(2) * input.size(1) * d0_128 +
                           d2 * input.size(1) * d0_128 +
                           d1 * d0_128 +
                           i * 128 + j;

              float val = hbm[offset];
              uint32_t intVal = *reinterpret_cast<uint32_t*>(&val);
              uint32_t x_i = (intVal & 0xFFFF) << 16;
              uint32_t y_i = (intVal & 0xFFFF0000);
              float x = *reinterpret_cast<float*>(&x_i);
              float y = *reinterpret_cast<float*>(&y_i);
              
              unsigned d0 = i * 256 + j;
              unsigned dx = d0 + 128;

              if (input.dtype() == dlc_bf16 || input.dtype() == dlc_fp32) {
                input.set_double({d0, d1, d2, d3, d4}, x);
                if (dx < input.size(0)) {
                  input.set_double({dx, d1, d2, d3, d4}, y);
                }
              } else {
                input.set_long({d0, d1, d2, d3, d4}, x);
                if (dx < input.size(0)) {
                  input.set_long({dx, d1, d2, d3, d4}, y);
                }
              }
            }
          }
        }
      }
    }
  }
}

inline void Tensor2Vector(const syn::nn::Tensor& tensor, float* hbm) {
  if (tensor.dtype() == dlc_int16 || tensor.dtype() == dlc_bf16) {
    Tensor2Vector16(tensor, hbm);
  } else {
    Tensor2Vector32(tensor, hbm);
  }
}

inline void Vector2Tensor(float* hbm, syn::nn::Tensor& tensor) {
  if (tensor.dtype() == dlc_int16 || tensor.dtype() == dlc_bf16) {
    Vector2Tensor16(hbm, tensor);
  } else {
    Vector2Tensor32(hbm, tensor);
  }
}

inline void InitDLCTensor(SIM_X86::DLCTensor& dlc_tensor, float* hbm,
                   const syn::nn::Tensor& input0_hbm) {
  int length = input0_hbm.dlc_dim0_padded() * input0_hbm.dlc_dim1();
  SIM_X86::tensor hbm_tensor(hbm, std::size_t(length));
  dlc_tensor.address = &hbm_tensor;

  for (int i = 0; i < 5; ++i) dlc_tensor.shape[i] = input0_hbm.size(i);
  for (int i = 0; i < 5; ++i) dlc_tensor.stride[i] = input0_hbm.stride(i);

  dlc_tensor.dim0 = input0_hbm.size(0);
  dlc_tensor.dim1 = input0_hbm.dlc_dim1();
  dlc_tensor.dim0_padded = input0_hbm.dlc_dim0_padded();
}

// #define asm
// #define volatile(...)

// #define _IN_CPP_LOGIC_CHECKER_
// // #include "../dlc_kernels/typehint.h"
// #undef _IN_CPP_LOGIC_CHECKER_

#endif
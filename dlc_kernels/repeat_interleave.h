#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#include "align.h"
#include "convert_element_type.h"
#include "permute.h"
// #include "typehint.h"

/**
 * TODO: special judge for small cases(vmem is big enough)
*/

/**
 * Auther: qingsun
 * 20241002
 * 参考：https://alidocs.dingtalk.com/i/nodes/yQod3RxJKMDYgdMpIgMpPzjBWkb4Mw9r?utm_scene=person_space&sideCollapsed=true&iframeQuery=utm_source%253Dportal%2526utm_medium%253Dportal_new_tab_open&corpId=ding10a90f5b23be3daf24f2f5cc6abecb85
 * 优化思路：（以 dlc SIM_X86::tensor 为例，a,b,c 只考虑 repeats = int）
    a. 对于 5 维 SIM_X86::tensor，假设 dim=2，repeats = 2, 那么，可以把 dim0*dim1 当作 dim0，因为它们在地址上是连续的，同时可以把 dim2,dim3,dim4 合并成一个 dim，最后变成一个 2 维矩阵举个栗子：SIM_X86::tensor(128, 5, 4, 3, 2) => SIM_X86::tensor(640, 24)，在最后的结果上是一致的为什么要看成 2 维矩阵：HBM2HBM 每次的 size 就不再局限于 dim0，而是直接和新的二维矩阵息息相关，假设新 SIM_X86::tensor(x, y)，那么 HBM2HBM 可以考虑 3 种方向：
        i. repeats                   * y           * HBM2HBM(x)
        ii. ALIGN128(x) / 128 * y           * HBM2HBM(repeats * 128)
        iii. ALIGN128(x) / 128 * repeats * HBM2HBM(y * 128)我们从三种情况里选择 dma 启动次数最少的情况就行，因为从理论上来说，传输的数据长度一致，dma 启动次数越少，cycle 越少
    b. 对于 dim = 0 的情况，还是先把 5 维 SIM_X86::tensor 看成 2 维 SIM_X86::tensor，然后转置二维矩阵，再 repeats，最后转置回去即可
    c. 对于 dim = null 的情况，这个情况下最难优化
        i. 最暴力的方法，把每个 dim0 都转置，然后让转置的所有矩阵在地址上连续，然后 repeats，最后转置 [1, dim0*dim1*dim2*dim3*dim4*repeats]这个矩阵，就获得了一个进行了 repeats 的 1 维矩阵。写起来比较方便，但是效率很低。
        ii. 对于 repeats = 128 * n：可以使用 mti_permute，因为 pcr 设置起来很方便，由于 repeats 是 128 的倍数，对于 dlc 来说就是最完美的状态
        iii. 对于 repeats = 2 ^ n（n > 0）：也可以使用 mti_permute，但是 pcr 设置会有点麻烦，因为 dim0 * repeats 不一定是 128 的倍数，所以需要拼接 dim0
        iv. 除了上述两种特殊情况，说实在都挺麻烦的，一方面由于最后的结果是一个 1 维 SIM_X86::tensor，需要拼接数据保证地址连续，另一方面，整数除法的精度不高，pcr 设置会出错
        v. 或者直接考虑用 scalar 做，感觉会更快
    d. 对于 repeats = SIM_X86::tensor 的情况
        i. 如果 SIM_X86::tensor 是 broadcast 的，那么直接当 repeats = int 来看就行
        ii. 对于 dim >= 0，此时可以把 SIM_X86::tensor 看成 3 维矩阵，例如 dim=2，SIM_X86::tensor(128, 5, 4, 3, 2) => SIM_X86::tensor(640, 4, 6)，此时由于对于 4*640 来说，里面每个 640 的 repeats 都不一样，所以就没有 repeats=int 那么方便，此时也有三种方向：
            1. 假设 SIM_X86::tensor(x, y, z)
            2. 假设 m = SUM(repeats_tensor)，repeats 的总和
            3. 假设 n = LEN(repeats_tensor)，repeats 的个数
            4. z * m * HBM2HBM(x)
            5. z * ALIGN128(x) / 128 * (HBM2HBM(r0 * 128) + HBM2HBM(r0 * 128) + ... + HBM2HBM(rn * 128))
            6. ALIGN128(x / 128) * m * HBM2HBM(z * 128)
            7. 从 3 种情况里选择最小的情况即可
        iii. 对于 dim = NULL：我的评价是超级麻烦，最简单不过暴力 permute，毕竟要不断拼接数据保持地址连续，或者直接考虑用 scalar 做
    e. 还有 2xys 的优化：
        i. repeats = int
            1. 横向 dma，如果 repeats 或者 y 有一个是偶数，就按其中的偶数对半，否则对于最后一块，需要考虑再分配给两个 xys，在特殊数据下会有较大的提升
            2. 列向 dma，如果 ALIGN128(x) / 128 或者 repeats 或者 y 是偶数，同样直接对半
*/

const int D_SMEM = 0;
const int D_CMEM = 3;

inline void HBM2SMem(SIM_X86::tensor hbm_address, int* smem_address, int length) {
  int handle = dlc_dma(hbm_address, D_HBM, (int*)((unsigned)(smem_address) / 128), D_SMEM, length, 128, 128, 128, 7);
  dlc_sync(handle);
}
inline void SMem2HBM(int* smem_address, SIM_X86::tensor hbm_address, int length){
    int handle = dlc_dma((int*)((unsigned)(smem_address) / 128), D_SMEM, hbm_address, D_HBM, length, 128, 128, 128, 7);
    dlc_sync(handle);    
}

inline void HBM2HBMstride(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm, SIM_X86::tensor input0_vmem, int VMEMSIZE,
                          int dma_length, int src_stride, int dst_stride) {
  int diff_addr   = ((int)(input0_hbm) - (int)(output_hbm)) * 32 * 4;
  int diff_stride = (src_stride - dst_stride);

  if (diff_addr % 2048 == 0 && diff_stride % 512 == 0) {
    int handle = dlc_dma(input0_hbm, D_HBM, output_hbm, D_HBM, dma_length, src_stride, dst_stride, 128, 7);
    dlc_sync(handle);
  } else {
    for (int len = 0; len < dma_length; len += VMEMSIZE) {
      int dma_sub_length = min(dma_length - len, VMEMSIZE);

      int handle1 = dlc_dma(tensor_slice(input0_hbm, len / 128 * src_stride / 32), D_HBM,
                            input0_vmem, D_VMEM, dma_sub_length, src_stride, 128, 128, 7);
      dlc_sync(handle1);

      int handle2 = dlc_dma(input0_vmem, D_VMEM,
                            tensor_slice(output_hbm, len / 128 * dst_stride / 32), D_HBM, 
                            dma_sub_length, 128, dst_stride, 128, 7);
      dlc_sync(handle2);
    }
  }
}

inline int CountBit(unsigned n) {
  n = n - ((n >> 1) & 0x55555555);                // 0x55555555 is 01010101010101010101010101010101 in binary
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333); // 0x33333333 is 00110011001100110011001100110011 in binary
  n = (n + (n >> 4)) & 0x0F0F0F0F;                // 0x0F0F0F0F is 00001111000011110000111100001111 in binary
  n = n + (n >> 8);
  n = n + (n >> 16);

  return n & 0x3F;                                // 0x0000003F is 00000000000000000000000000111111 in binary
}

inline int SumVreg_i(int8_128 vr) {
  int8_128 sum = vr;

  for (int i = 0; i < 7; ++i) {
    vr = __$S(v_row_rotate(__$F(vr), 0));
    sum += vr;
  }

  vr = sum;
  for (int i = 0; i < 127; ++i) {
    vr = __$S(m_rotate(__$F(vr), -1, 0));
    sum += vr;
  }

  return sum[0];
}

inline int SumRepeatsTensor(SIM_X86::tensor repeats, int len) {
  int8_128 re = 0;

  int vs = 0;
  for (; vs < (len / 1024 * 1024); vs += 1024) {
    re += __$S(v_f32_ld_tnsr_b(vs / 32, repeats));
  }
  if (len & 0x7FF) {
    float8_128 x = v_f32_ld_tnsr_st_msk(vs / 32, repeats, 1, pre_exp2((ALIGN128(len) & 0x7FF) / 128));
    int8_128 id = get_core_id();
    bool8_128 flag = v_s32_cmp(LS, id, len & 0x7FF);
    re += v_s32_sel(flag, 0, __$S(x));
  }
  
  return SumVreg_i(re);
}

/* repeats_tensor to repeats[] */
inline void RepeatInterleaveRepeatsTensorToArray(int8_128 vreg, int* repeats, int len) {
  int cnt = 0;

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 128; ++j) {
      repeats[cnt++] = vreg[0];
      if (len == cnt) return;
      vreg = __$S(m_rotate(__$F(vreg), -1, false));
    }
    vreg = __$S(v_row_rotate(__$F(vreg), 0));
  }
}

/**
 * 有三种CASE
 * 
 * case 0: repeats * dim1 次 hbm，每次 len = dim0
 * case 1: dim1 * (dim0 / 128) 次 hbm，每次 len = repeats * 128（列向dma，每次stride_src = 0, stride_dst = dim0, 复制repeats次，len = repeats * 128）
 * case 2: repeats * (dim0 / 128) 次 hbm，每次 len = dim1 * 128（列向dma，每次stride_src = dim0, stride_dst = dim0 * dim1, 复制dim1次，len = dim1 * repeats）
 * 
 * dim0    = 256
 * dim1    = 5
 * repeats = 3
 *                             0, 1, ..., 255, 256
 *                             0, 1, ..., 255, 256
 *                             0, 1, ..., 255, 256
 *                             4, 5, ..., 255, 256
 *                             4, 5, ..., 255, 256
 * 0, 1, ..., 255, 256         4, 5, ..., 255, 256
 * 4, 5, ..., 255, 256         8, 9, ..., 255, 256
 * 8, 9, ..., 255, 256    =>   8, 9, ..., 255, 256   列向dma，考虑两种分组，一种是按 repeats，一种是按 dim1
 * 3, 4, ..., 255, 256         8, 9, ..., 255, 256
 * 7, 8, ..., 255, 256         3, 4, ..., 255, 256
 *                             3, 4, ..., 255, 256
 *                             3, 4, ..., 255, 256
 *                             7, 8, ..., 255, 256
 *                             7, 8, ..., 255, 256
 *                             7, 8, ..., 255, 256
*/
inline int RepeatInterleaveDim0IntCalCase(int dim0, int dim1, int repeats) {
  int t0 = repeats * dim1;
  int t1 = dim1    * ALIGN128(dim0) / 128;
  int t2 = repeats * ALIGN128(dim0) / 128;

  int MIN = t0 < t1 ? t0 : t1;
  MIN = MIN < t2 ? MIN : t2;

  if (MIN == t0) return 0;
  if (MIN == t1) return 1;
  return 2;
}

inline int RepeatInterleaveTensorCalCase(int dim0, int dim2, int repeats_sum, int repeats_len) {
  int dim0_128 = ALIGN128(dim0);
  int dim0_div_128 = dim0_128 / 128;

  int m = repeats_sum;
  int n = repeats_len;

  int case0 = dim2 * m;
  int case1 = dim0_div_128 * dim2 * n;
  int case2 = dim0_div_128 * m;

  int MIN = case0 < case1 ? case0 : case1;
  MIN = MIN < case2 ? MIN : case2;

  if (MIN == case0) return 0;
  if (MIN == case1) return 1;
  return 2;
}

inline void RepeatInterleaveIntKernel(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm,
                                      SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                      int repeats, int *input0_shape) {
  int dim0_128 = ALIGN128(input0_shape[0]);
  int dim1 = input0_shape[1];

  int dim0_128_repeats = dim0_128 * repeats;

  int CASE = RepeatInterleaveDim0IntCalCase(dim0_128, dim1, repeats);

  if (CASE == 0) {
    if ((dim1 & 1) == 0) {
      int start_i = 0;
      int end_i   = 0;
      
      if (get_device_id()) {
        start_i = dim1 - dim1 / 2;
        end_i   = dim1;
      } else {
        start_i = 0;
        end_i   = dim1 / 2;
      }

      for (int i = start_i; i < end_i; ++i) {
        for (int j = 0; j < repeats; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, i * dim0_128 / 32),
                        tensor_slice(output_hbm, (i * dim0_128_repeats + j * dim0_128) / 32),
                        input0_vmem, VMEMSIZE, dim0_128, 128, 128);
        }
      }
    } else if ((repeats & 1) == 0) {
      int start_j = 0;
      int end_j   = 0;
      
      if (get_device_id()) {
        start_j = repeats - repeats / 2;
        end_j   = repeats;
      } else {
        start_j = 0;
        end_j   = repeats / 2;
      }

      for (int i = 0; i < dim1; ++i) {
        for (int j = start_j; j < end_j; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, i * dim0_128 / 32),
                        tensor_slice(output_hbm, (i * dim0_128_repeats + j * dim0_128) / 32),
                        input0_vmem, VMEMSIZE, dim0_128, 128, 128);
        }
      }
    } else {
      int start_i = 0;
      int end_i   = 0;
      int start_j = 0;
      int end_j   = 0;

      if (get_device_id()) {
        start_i = dim1 / 2;
        end_i   = dim1 - 1;
        start_j = repeats / 2;
        end_j   = repeats - 1;
      } else {
        start_i = 0;
        end_i   = dim1 / 2;
        start_j = 0;
        end_j   = repeats / 2;
      }
      
      for (int i = start_i; i < end_i; ++i) {
        for (int j = 0; j < repeats; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, i * dim0_128 / 32),
                        tensor_slice(output_hbm, (i * dim0_128_repeats + j * dim0_128) / 32),
                        input0_vmem, VMEMSIZE, dim0_128, 128, 128);
        }
      }
      
      for (int i = dim1 - 1; i < dim1; ++i) {
        for (int j = start_j; j < end_j; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, i * dim0_128 / 32),
                        tensor_slice(output_hbm, (i * dim0_128_repeats + j * dim0_128) / 32),
                        input0_vmem, VMEMSIZE, dim0_128, 128, 128);
        }
      }

      int half_len = 0;
      int half_offset = 0;
      
      if (get_device_id()) {
        half_len = dim0_128 - dim0_128 / 128 / 2 * 128;
        half_offset = dim0_128 / 128 / 2 * 128;
      } else {
        half_len = dim0_128 / 128 / 2 * 128;
        half_offset = 0;
      }

      HBM2HBMstride(tensor_slice(input0_hbm, ((dim1 - 1) * dim0_128 + half_offset) / 32),
                    tensor_slice(output_hbm, ((repeats * dim1 - 1) * dim0_128 + half_offset) / 32),
                    input0_vmem, VMEMSIZE, half_len, 128, 128);
    }
  } else if (CASE == 1) {
    if (((dim0_128 / 128) & 1) == 0) {
      int start_i = 0;
      int end_i = 0;

      if (get_device_id()) {
        start_i = dim0_128 / 128 / 2 * 128;
        end_i = dim0_128;
      } else {
        start_i = 0;
        end_i = dim0_128 / 128 / 2 * 128;
      }

      for (int i = start_i; i < end_i; i += 128) {
        for (int j = 0; j < dim1; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, (j * dim0_128 + i) / 32),
                        tensor_slice(output_hbm, (j * dim0_128_repeats + i) / 32),
                        input0_vmem, VMEMSIZE, repeats * 128, 0, dim0_128);
        }
      }
    } else if ((dim1 & 1) == 0) {
      int start_j = 0;
      int end_j = 0;

      if (get_device_id()) {
        start_j = dim1 / 2;
        end_j = dim1;
      } else {
        start_j = 0;
        end_j = dim1 / 2;
      }

      for (int i = 0; i < dim0_128; i += 128) {
        for (int j = start_j; j < end_j; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, (j * dim0_128 + i) / 32),
                        tensor_slice(output_hbm, (j * dim0_128_repeats + i) / 32),
                        input0_vmem, VMEMSIZE, repeats * 128, 0, dim0_128);
        }
      }
    } else {
      int start_i = 0;
      int end_i = 0;
      int start_j = 0;
      int end_j = 0;

      if (get_device_id()) {
        start_i = dim0_128 / 128 / 2 * 128;
        end_i   = dim0_128 - 128;
        start_j = dim1 / 2;
        end_j   = dim1 - 1;
      } else {
        start_i = 0;
        end_i   = dim0_128 / 128 / 2 * 128;
        start_j = 0;
        end_j   = dim1 / 2;
      }

      for (int i = 0; i < dim0_128; i += 128) {
        for (int j = start_j; j < end_j; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, (j * dim0_128 + i) / 32),
                        tensor_slice(output_hbm, (j * dim0_128_repeats + i) / 32),
                        input0_vmem, VMEMSIZE, repeats * 128, 0, dim0_128);
        }
      }

      for (int i = start_i; i < end_i; i += 128) {
        for (int j = dim1 - 1; j < dim1; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, (j * dim0_128 + i) / 32),
                        tensor_slice(output_hbm, (j * dim0_128_repeats + i) / 32),
                        input0_vmem, VMEMSIZE, repeats * 128, 0, dim0_128);
        }
      }

      int half_len = 0;
      int half_offset_i = 0;
      int half_offset_j = 0;

      if (get_device_id()) {
        half_len = repeats - repeats / 2;
        // ((repeats + 1) & 1) & 1) is used for special judge of dim1 & 1 == 0
        half_offset_i = dim0_128 * dim1 - 128;
        half_offset_j = (dim1 - 1) * dim0_128_repeats + dim0_128 * (half_len + ((repeats + 1) & 1)) - 128;
      } else {
        half_len = repeats / 2;
        half_offset_i = dim0_128 * dim1 - 128;
        half_offset_j = (dim1 - 1) * dim0_128_repeats + dim0_128 - 128;
      }

      HBM2HBMstride(tensor_slice(input0_hbm, half_offset_i / 32),
                    tensor_slice(output_hbm, half_offset_j / 32),
                    input0_vmem, VMEMSIZE, half_len * 128, 0, dim0_128);

    }
  } else {
    if (((dim0_128 / 128) & 1) == 0) {
      int start_i = 0;
      int end_i = 0;

      if (get_device_id()) {
        start_i = dim0_128 / 128 / 2 * 128;
        end_i = dim0_128;
      } else {
        start_i = 0;
        end_i = dim0_128 / 128 / 2 * 128;
      }

      for (int i = start_i; i < end_i; i += 128) {
        for (int j = 0; j < repeats; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, i / 32),
                        tensor_slice(output_hbm, (j * dim0_128 + i) / 32),
                        input0_vmem, VMEMSIZE, dim1 * 128, dim0_128, dim0_128_repeats);
        }
      }
    } else if ((repeats & 1) == 0) {
      int start_j = 0;
      int end_j = 0;

      if (get_device_id()) {
        start_j = repeats / 2;
        end_j = repeats;
      } else {
        start_j = 0;
        end_j = repeats / 2;
      }
      
      for (int i = 0; i < dim0_128; i += 128) {
        for (int j = start_j; j < end_j; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, i / 32),
                        tensor_slice(output_hbm, (j * dim0_128 + i) / 32),
                        input0_vmem, VMEMSIZE, dim1 * 128, dim0_128, dim0_128_repeats);
        }
      }
    } else {
      int start_i = 0;
      int end_i = 0;
      int start_j = 0;
      int end_j = 0;

      if (get_device_id()) {
        start_i = dim0_128 / 128 / 2 * 128;
        end_i = dim0_128 - 128;
        start_j = repeats / 2;
        end_j = repeats - 1;
      } else {
        start_i = 0;
        end_i = dim0_128 / 128 / 2 * 128;
        start_j = 0;
        end_j = repeats / 2;
      }

      for (int i = 0; i < dim0_128; i += 128) {
        for (int j = start_j; j < end_j; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, i / 32),
                        tensor_slice(output_hbm, (j * dim0_128 + i) / 32),
                        input0_vmem, VMEMSIZE, dim1 * 128, dim0_128, dim0_128_repeats);
        }
      }

      for (int i = start_i; i < end_i; i += 128) {
        for (int j = repeats - 1; j < repeats; ++j) {
          HBM2HBMstride(tensor_slice(input0_hbm, i / 32),
                        tensor_slice(output_hbm, (j * dim0_128 + i) / 32),
                        input0_vmem, VMEMSIZE, dim1 * 128, dim0_128, dim0_128_repeats);
        }
      }

      int half_len = 0;
      int half_offset_i = 0;
      int half_offset_j = 0;

      if (get_device_id()) {
        half_len = dim1 - dim1 / 2;
        // ((dim1 + 1) & 1) is used for special judge of dim1 & 1 == 0
        half_offset_i = (half_len + ((dim1 + 1) & 1)) * dim0_128 - 128;
        half_offset_j = (half_len + ((dim1 + 1) & 1)) * dim0_128_repeats - 128;
      } else {
        half_len = dim1 / 2;
        half_offset_i = dim0_128 - 128;
        half_offset_j = dim0_128_repeats - 128;
      }

      HBM2HBMstride(tensor_slice(input0_hbm, half_offset_i / 32),
                    tensor_slice(output_hbm, half_offset_j / 32),
                    input0_vmem, VMEMSIZE, half_len * 128, dim0_128, dim0_128_repeats);
    }
  }

  sync_device();
}

inline void RepeatInterleaveTensorKernel(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm,
                                        SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                        SIM_X86::tensor repeats_hbm, int *input0_shape,
                                        int repeats_length, int *repeats_sum_) {
  SIM_X86::tensor repeats_vmem = input0_vmem;
  input0_vmem = tensor_slice(input0_vmem, ALIGN128(repeats_length) / 32);
  VMEMSIZE = VMEMSIZE - ALIGN128(repeats_length);

  HBM2VMem(repeats_hbm, repeats_vmem, ALIGN128(repeats_length));
  int repeats_sum = SumRepeatsTensor(repeats_vmem, repeats_length);
  *repeats_sum_ = repeats_sum;

  int dim0 = ALIGN128(input0_shape[0]);
  int dim2 = input0_shape[2];
  int CASE = RepeatInterleaveTensorCalCase(dim0, dim2, repeats_sum, repeats_length);

  int repeats_array[1024] = {0};

  if (CASE == 0) {
    /**
     * TODO: if the repeats_tensor has many repeats that satisfy "(repeats & 1) == 1",
     *       xys1 will do much dma than xys1, there might be some ideas to solve this problem
    */
    int repeats_prefix_sum = 0;
    for (int repeats_index = 0; repeats_index < repeats_length; repeats_index += 1024) {
      int k_len = min(1024, repeats_length - repeats_index);
      int8_128 vreg = __$S(v_f32_ld_tnsr_st_msk(repeats_index / 32, repeats_vmem, 1, pre_exp2(ALIGN128(k_len) / 128)));
      RepeatInterleaveRepeatsTensorToArray(vreg, repeats_array, k_len);

      for (int k = 0; k < k_len; ++k) {
        int repeats = repeats_array[k];

        for (int d2 = 0; d2 < dim2; ++d2) {
          int r = 1 * get_device_id();

          for (; r < repeats; r += 2) {
            HBM2HBMstride(tensor_slice(input0_hbm, (repeats_index + k + d2 * repeats_length) * dim0 / 32),
                          tensor_slice(output_hbm, (repeats_prefix_sum + r + d2 * repeats_sum) * dim0 / 32),
                          input0_vmem, VMEMSIZE, dim0, 128, 128);
          }
        }

        repeats_prefix_sum += repeats;
      }
    }
  } else if (CASE == 1) {
    /**
     * TODO: if ((dim0 / 128) & 1) == 1, xys1 will do more than xys0,
     *       so we can divide the dma of the last column equally into two xys
    */
    int repeats_prefix_sum = 0;
    for (int repeats_index = 0; repeats_index < repeats_length; repeats_index += 1024) {
      int repeats_sub_len = min(1024, repeats_length - repeats_index);
      int8_128 vreg = __$S(v_f32_ld_tnsr_st_msk(repeats_index / 32, repeats_vmem, 1, pre_exp2(ALIGN128(repeats_sub_len) / 128)));
      RepeatInterleaveRepeatsTensorToArray(vreg, repeats_array, repeats_sub_len);

      for (int repeats_sub_index = 0; repeats_sub_index < repeats_sub_len; ++repeats_sub_index) {
        int repeats = repeats_array[repeats_sub_index];

        for (int d2 = 0; d2 < dim2; ++d2) {
          int vs = 128 * get_device_id();

          for (; vs < dim0; vs += 256) {
            HBM2HBMstride(tensor_slice(input0_hbm, ((repeats_index + repeats_sub_index + d2 * repeats_length) * dim0 + vs) / 32),
                          tensor_slice(output_hbm, ((repeats_prefix_sum + d2 * repeats_sum) * dim0 + vs) / 32),
                          input0_vmem, VMEMSIZE, 128 * repeats, 0, dim0);
          }
        }

        repeats_prefix_sum += repeats;
      }
    }
  } else {
    /**
     * TODO: if ((dim0 / 128) & 1) == 1, xys1 will do more than xys0,
     *       so we can divide the dma of the last column equally into two xys
    */
    int repeats_prefix_sum = 0;
    for (int repeats_index = 0; repeats_index < repeats_length; repeats_index += 1024) {
      int repeats_sub_len = min(1024, repeats_length - repeats_index);
      int8_128 vreg = __$S(v_f32_ld_tnsr_st_msk(repeats_index / 32, repeats_vmem, 1, pre_exp2(ALIGN128(repeats_sub_len) / 128)));
      RepeatInterleaveRepeatsTensorToArray(vreg, repeats_array, repeats_sub_len);

      for (int repeats_sub_index = 0; repeats_sub_index < repeats_sub_len; ++repeats_sub_index) {
        int repeats = repeats_array[repeats_sub_index];

        int vs = 128 * get_device_id();
        for (; vs < dim0; vs += 256) {

          for (int r = 0; r < repeats; ++r) {
            HBM2HBMstride(tensor_slice(input0_hbm, ((repeats_index + repeats_sub_index) * dim0 + vs) / 32),
                          tensor_slice(output_hbm, ((repeats_prefix_sum + r) * dim0 + vs) / 32),
                          input0_vmem, VMEMSIZE, 128 * dim2, repeats_length * dim0, repeats_sum * dim0);
          }
        }

        repeats_prefix_sum += repeats;
      }
    }
  }

  sync_device();
}

// in this case, turn to 2D Tensor also make correct answer
inline void RepeatInterleaveInt(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm,
                                SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                int dim, int repeats, int *input0_shape) {
  int dim1 = 0;
  int dim0_128 = 0;

  if (dim == 4) {
    dim1 = input0_shape[4];
    dim0_128  = ALIGN128(input0_shape[0]) * input0_shape[1] * input0_shape[2] * input0_shape[3];
  } else if (dim == 3) {
    dim1 = input0_shape[4] * input0_shape[3];
    dim0_128  = ALIGN128(input0_shape[0]) * input0_shape[1] * input0_shape[2];
  } else if (dim == 2) {
    dim1 = input0_shape[4] * input0_shape[3] * input0_shape[2];
    dim0_128  = ALIGN128(input0_shape[0]) * input0_shape[1];
  } else if (dim == 1) {
    dim1 = input0_shape[4] * input0_shape[3] * input0_shape[2] * input0_shape[1];
    dim0_128  = ALIGN128(input0_shape[0]);
  }

  int permute_shape[5] = {dim0_128, dim1, 1, 1, 1};
  RepeatInterleaveIntKernel(input0_hbm, output_hbm,
                            input0_vmem, VMEMSIZE,
                            repeats, permute_shape);
}

inline void RepeatInterleaveTensor(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm,
                                   SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                   int dim, SIM_X86::tensor repeats, int *input0_shape) {
  int dim2 = 1;
  int dim1 = 0;
  int dim0_128 = 0;

  if (dim == 4) {
    dim2 = 1;
    dim1 = input0_shape[4];
    dim0_128  = ALIGN128(input0_shape[0]) * input0_shape[1] * input0_shape[2] * input0_shape[3];
  } else if (dim == 3) {
    dim2 = input0_shape[4];
    dim1 = input0_shape[3];
    dim0_128  = ALIGN128(input0_shape[0]) * input0_shape[1] * input0_shape[2];
  } else if (dim == 2) {
    dim2 = input0_shape[4] * input0_shape[3];
    dim1 = input0_shape[2];
    dim0_128  = ALIGN128(input0_shape[0]) * input0_shape[1];
  } else if (dim == 1) {
    dim2 = input0_shape[4] * input0_shape[3] * input0_shape[2];
    dim1 = input0_shape[1];
    dim0_128  = ALIGN128(input0_shape[0]);
  }
  
  int permute_shape[5] = {dim0_128, dim1, dim2, 1, 1};
  int repeats_sum = 0;
  RepeatInterleaveTensorKernel(input0_hbm, output_hbm,
                               input0_vmem, VMEMSIZE,
                               repeats, permute_shape, input0_shape[dim], &repeats_sum);
}

inline void RepeatInterleaveDim0Int(SIM_X86::tensor input0_hbm,
                                    SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm,
                                    SIM_X86::tensor output_hbm,
                                    SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                    int repeats, int dim0, int dim1) {
  int dim[5] = {dim0, dim1, 1, 1, 1};
  int perm[5] = {1, 0, 2, 3, 4};
  _permute_hbm(input0_hbm, input1_hbm, input0_vmem, VMEMSIZE, dim, perm);

  int dim0_permute = dim1;
  int dim1_permute = dim0;

  int permute_shape[5] = {ALIGN128(dim0_permute), dim1_permute, 1, 1, 1};
  RepeatInterleaveIntKernel(input1_hbm, input2_hbm,
                            input0_vmem, VMEMSIZE,
                            repeats, permute_shape);

  dim[0] = dim0_permute;
  dim[1] = dim1_permute * repeats;
  _permute_hbm(input2_hbm, output_hbm, input0_vmem, VMEMSIZE, dim, perm);
}

inline void RepeatInterleaveDim0Tensor(SIM_X86::tensor input0_hbm,
                                        SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm,
                                        SIM_X86::tensor output_hbm,
                                        SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                        SIM_X86::tensor repeats_hbm, int dim0, int dim1) {
  int dims[5] = {dim0, dim1, 1, 1, 1};
  int perm[5] = {1, 0, 2, 3, 4};
  _permute_hbm(input0_hbm, input1_hbm, input0_vmem, VMEMSIZE, dims, perm);

  int dims_permute[5] = {ALIGN128(dim1), dim0, 1, 1, 1};
  int repeats_sum = 0;
  RepeatInterleaveTensorKernel(input1_hbm, input2_hbm,
                                input0_vmem, VMEMSIZE,
                                repeats_hbm, dims_permute, dim0,
                                &repeats_sum);

  dims_permute[0] = dim1;
  dims_permute[1] = repeats_sum;
  _permute_hbm(input2_hbm, output_hbm, input0_vmem, VMEMSIZE, dims_permute, perm);
}

inline void RepeatInterleaveDim0TensorBf16(SIM_X86::tensor input0_hbm,
                                            SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm,
                                            SIM_X86::tensor output_hbm,
                                            SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                            SIM_X86::tensor repeats_hbm, int dim0, int dim1) {
  int dims[5] = {dim0, dim1, 1, 1, 1};
  int perm[5] = {1, 0, 2, 3, 4};
  _permute_hbm_bf16(input0_hbm, input1_hbm, input0_vmem, VMEMSIZE, dims, perm);

  int dims_permute[5] = {ALIGN256(dim1) / 2, dim0, 1, 1, 1};
  int repeats_sum = 0;
  RepeatInterleaveTensorKernel(input1_hbm, input2_hbm,
                                input0_vmem, VMEMSIZE,
                                repeats_hbm, dims_permute, dim0,
                                &repeats_sum);

  dims_permute[0] = dim1;
  dims_permute[1] = repeats_sum;
  _permute_hbm_bf16(input2_hbm, output_hbm, input0_vmem, VMEMSIZE, dims_permute, perm);
}

/**
 * TODO: special judge for (repeats == int) && (repeats % 128 == 0)
*/
// inline void RepeatInterleaveArrayInt128(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm,
//                                         SIM_X86::tensor input0_vmem, int VMEMSIZE,
//                                         int repeats, int dim0, int dim1) {
//   int input_vmem_size = (soft_sdiv(VMEMSIZE, (repeats + 1))) / 128 * 128;

//   SIM_X86::tensor output_vmem = tensor_slice(input0_vmem, input_vmem_size / 32);

//   int dim0_128 = ALIGN128(dim0);
//   int dma_length = dim0_128 * dim1;

//   int max_n_load_dim0 = soft_sdiv(VMEMSIZE, dim0_128);

//   // if (max_n_)

//   int count = 0;
//   for (int vs = 0; vs < dma_length; vs +=  VMEMSIZE) {
//     int dma_sub_length = min(VMEMSIZE, dma_length - vs);

//     HBM2VMem(input0_hbm, input0_vmem, dma_sub_length);

//   }



//   for (int d1 = 0; d1 < dim1; ++d1) {
//     for (int d0 = 0; d0 < dim0 / 128 * 128; d0 += 128) {
      
//     }
//     if (dim0 & 0x7F) {

//     }
//   }
// }

inline void RepeatInterleaveArrayIntDefault(SIM_X86::tensor input0_hbm,
                                            SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm,
                                            SIM_X86::tensor output_hbm,
                                            SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                            int repeats, int dim0, int dim1) {
  int perm[5] = {1, 0, 2, 3, 4};
  int dims[5] = {dim0, 1, 1, 1, 1};
  for (int d1 = 0; d1 < dim1; ++d1) {
    _permute_hbm(tensor_slice(input0_hbm, d1 * ALIGN128(dim0) / 32),
                 tensor_slice(input1_hbm, d1 * dim0 * 128 / 32),
                 input0_vmem, VMEMSIZE, dims, perm);
  }

  int dim0_permute = 1;
  int dim1_permute = dim0 * dim1;

  int permute_shape[5] = {ALIGN128(dim0_permute), dim1_permute, 1, 1, 1};
  RepeatInterleaveIntKernel(input1_hbm, input2_hbm,
                            input0_vmem, VMEMSIZE,
                            repeats, permute_shape);

  dims[0] = dim0_permute;
  dims[1] = dim1_permute * repeats;
  _permute_hbm(input2_hbm, output_hbm, input0_vmem, VMEMSIZE, dims, perm);
}

inline void RepeatInterleaveArrayIntDefaultBf16(SIM_X86::tensor input0_hbm,
                                                SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm,
                                                SIM_X86::tensor output_hbm,
                                                SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                                int repeats, int dim0, int dim1) {
  int dim0_128 = ALIGN256(dim0) / 2;
  int perm[5] = {1, 0, 2, 3, 4};
  int dims[5] = {dim0, 1, 1, 1, 1};

  for (int d1 = 0; d1 < dim1; ++d1) {
    _permute_hbm_bf16(tensor_slice(input0_hbm, d1 * dim0_128 / 32),
                      tensor_slice(input1_hbm, d1 * dim0 * 128 / 32),
                      input0_vmem, VMEMSIZE, dims, perm);
  }

  int dim0_permute = 1;
  int dim1_permute = dim0 * dim1;

  int permute_shape[5] = {ALIGN256(dim0_permute) / 2, dim1_permute, 1, 1, 1};
  RepeatInterleaveIntKernel(input1_hbm, input2_hbm,
                            input0_vmem, VMEMSIZE,
                            repeats, permute_shape);

  dims[0] = dim0_permute;
  dims[1] = dim1_permute * repeats;
  _permute_hbm_bf16(input2_hbm, output_hbm, input0_vmem, VMEMSIZE, dims, perm);
}

inline void RepeatInterleaveArrayTensorDefault(SIM_X86::tensor input0_hbm,
                                                SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm,
                                                SIM_X86::tensor output_hbm,
                                                SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                                SIM_X86::tensor repeats_hbm, int dim0, int dim1) {
  int perm[5] = {1, 0, 2, 3, 4};
  int dims[5] = {dim0, 1, 1, 1, 1};
  for (int d1 = 0; d1 < dim1; ++d1) {
    _permute_hbm(tensor_slice(input0_hbm, d1 * ALIGN128(dim0) / 32),
                 tensor_slice(input1_hbm, d1 * dim0 * 128 / 32),
                 input0_vmem, VMEMSIZE, dims, perm);
  }

  int dim0_permute = 1;
  int dim1_permute = dim0 * dim1;

  int permute_shape[5] = {dim0_permute, dim1_permute, 1, 1, 1};
  int repeats_sum = 0;
  RepeatInterleaveTensorKernel(input1_hbm, input2_hbm,
                                input0_vmem, VMEMSIZE,
                                repeats_hbm, permute_shape, dim0 * dim1,
                                &repeats_sum);

  dims[0] = dim0_permute;
  dims[1] = repeats_sum;
  _permute_hbm(input2_hbm, output_hbm, input0_vmem, VMEMSIZE, dims, perm);
}

inline void RepeatInterleaveArrayTensorDefaultBf16(SIM_X86::tensor input0_hbm,
                                                    SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm,
                                                    SIM_X86::tensor output_hbm,
                                                    SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                                    SIM_X86::tensor repeats_hbm, int dim0, int dim1) {
  int dim0_128 = ALIGN256(dim0) / 2;
  int perm[5] = {1, 0, 2, 3, 4};
  int dims[5] = {dim0, 1, 1, 1, 1};

  for (int d1 = 0; d1 < dim1; ++d1) {
    _permute_hbm_bf16(tensor_slice(input0_hbm, d1 * dim0_128 / 32),
                      tensor_slice(input1_hbm, d1 * dim0 * 128 / 32),
                      input0_vmem, VMEMSIZE, dims, perm);
  }

  int dim0_permute = 1;
  int dim1_permute = dim0 * dim1;

  int permute_shape[5] = {ALIGN128(dim0_permute), dim1_permute, 1, 1, 1};
  int repeats_sum = 0;
  RepeatInterleaveTensorKernel(input1_hbm, input2_hbm,
                                input0_vmem, VMEMSIZE,
                                repeats_hbm, permute_shape, dim0 * dim1,
                                &repeats_sum);              

  dims[0] = dim0_permute;
  dims[1] = repeats_sum;
  _permute_hbm_bf16(input2_hbm, output_hbm, input0_vmem, VMEMSIZE, dims, perm);
}

/**
 * Experimental code
 * test speed between scalar and vector in case "dim = null"
*/
inline void RepeatInterleaveArrayIntDefaultScalar(SIM_X86::tensor input0_hbm, SIM_X86::tensor output_hbm,
                                                  SIM_X86::tensor input_smem, int SMEMSIZE,
                                                  int repeats, int dim0, int dim1) {
  SIM_X86::tensor output_smem = (input_smem + SMEMSIZE / 128 / 2 * 128 * 4);
  for (int i = 0; i < dim1; ++i) {
    HBM2SMem(tensor_slice(input0_hbm, i * ALIGN128(dim0) / 32), input_smem, ALIGN128(dim0));
    for (int j = 0; j < dim0; ++j) {
      for (int r = 0; r < repeats; ++r) {
        ((int*)output_smem)[i * dim0 * repeats + j * repeats + r] = ((int*)input_smem)[i * dim0 + j];
      }
    }
    SMem2HBM(output_smem, tensor_slice(output_hbm, i * ALIGN128(dim0 * repeats) / 32), ALIGN128(dim0 * repeats));
  }
}


// dim[0] not padded
inline void RepeatInterleaveDim0IntBf16(SIM_X86::tensor input0_hbm,
                                        SIM_X86::tensor input1_hbm, SIM_X86::tensor input2_hbm,
                                        SIM_X86::tensor output_hbm,
                                        SIM_X86::tensor input0_vmem, int VMEMSIZE,
                                        int repeats, int dim0, int dim1) {
  int dim[5] = {dim0, dim1, 1, 1, 1};
  int perm[5] = {1, 0, 2, 3, 4};

  _permute_hbm_bf16(input0_hbm, input1_hbm, input0_vmem, VMEMSIZE, dim, perm);

  int dim0_permute = dim1;
  int dim1_permute = dim0;

  int permute_shape[5] = {ALIGN256(dim1) / 2, dim1_permute, 1, 1, 1};
  RepeatInterleaveIntKernel(input1_hbm, input2_hbm,
                            input0_vmem, VMEMSIZE,
                            repeats, permute_shape);

  dim[0] = dim0_permute;
  dim[1] = dim1_permute * repeats;
  _permute_hbm_bf16(input2_hbm, output_hbm, input0_vmem, VMEMSIZE, dim, perm);
}
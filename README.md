## dlc_sim_x86

  使用x86模拟函数

## 说明

  dlc-intrinsics.h typehint.h x86.h
  这三个库是自定义的，所以基本所有的修改都是围绕这三个文件
  sync_h.c sync_c.c 主要为了方便同步 dlc_kernels 和 算子
  主要是靠 string.replace 实现的，所以需要根据库的实际需要及时修改

## 使用方法

1. 运行 sync_h.py 同步 dlc_kernels 库里的 .h 库
2. 运行 sync_c.py 同步你想测试的 kernel，把 kernel 的文件名放进去就行
3. 修改对应的 SynTest.cpp，参考下面的代码

```C++
/* 在 runTest 上面实现这个 */
#ifdef DLC_X86
#include "../../x86/matmul_t_pingpong.c"

void ToX86(syn::nn::Tensor& input0_hbm,
           syn::nn::Tensor& input1_hbm,
           syn::nn::Tensor& output_hbm,
           int ah, int aw, int bw,
           bool lhs_Transpose, bool rhs_Transpose, bool res_Transpose,
           std::string kernel_name) {
  DLCType dtype = (kernel_name.substr(kernel_name.size() - 4) == "bf16" ? dlc_bf16 : dlc_fp32);

  SIM_X86::DLCTensor input0_hbm_(dlc_memorys.hbm_alloc(input0_hbm), input0_hbm);
  SIM_X86::DLCTensor input1_hbm_(dlc_memorys.hbm_alloc(input1_hbm), input1_hbm);
  SIM_X86::DLCTensor output_hbm_(dlc_memorys.hbm_alloc(output_hbm), output_hbm, false);
  SIM_X86::DLCScalar ChunkBlock[3] = {ah, aw, bw};
  SIM_X86::DLCScalar Is_Trans[3] = {lhs_Transpose, rhs_Transpose, res_Transpose};

  auto start = std::chrono::high_resolution_clock::now();
  std::thread thread1(main_x86, &dlc_memorys.info_xys0, 
                      &input0_hbm_, &input1_hbm_, &output_hbm_, ChunkBlock, Is_Trans);
  std::thread thread2(main_x86, &dlc_memorys.info_xys1,
                      &input0_hbm_, &input1_hbm_, &output_hbm_, ChunkBlock, Is_Trans);
  thread1.join();
  thread2.join();
  auto end = std::chrono::high_resolution_clock::now();

  Vector2Tensor(output_hbm_.address->data_ptr, output_hbm);

  printf("%s%s: X86 Execution time: %fms\n",
          kernel_name.c_str(), (dtype == dlc_bf16 ? "_bf16" : ""),
          std::chrono::duration<double, std::milli>(end - start).count());
  printf("MIN_VMEM_SIZE = %d\n", MIN_VMEM_SIZE);
}
#endif

/* 在 runTest 里实现这个，把入参改成自己需要的 */
#ifdef DLC_X86
  ToX86(input0_hbm, input1_hbm, output_hbm,
    ah, aw, bw, lhs_Transpose, rhs_Transpose, res_Transpose, kernel_name);

  double diff_max = 0;
  for (int i = 0; i < output_hbm.numel(); ++i) {
    diff_max = std::max(diff_max, std::fabs(output_hbm.get_double_flat(i) - output_ref.get_double_flat(i)));
    // if (std::fabs(output_hbm.get_double_flat(i) - output_ref.get_double_flat(i)) > 0.f)
    //   printf("idx = %d, hbm = %f, ref = %f, diff = %f\n", i,
    //           output_hbm.get_double_flat(i),
    //           output_ref.get_double_flat(i),
    //           std::fabs(output_hbm.get_double_flat(i) - output_ref.get_double_flat(i)));
  }
  printf("diff_max = %f\n", diff_max);
#endif
```
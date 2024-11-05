#include "../x86.h"

void test_x86_vmem_ldst(SIM_X86::tensor vmem) {
  float8_128 x = v_cvt_itof(get_core_id());
  float8_128 y = x + x;

  printf("1q\n");
  v_f32_st_tnsr_b(128 / 32, vmem, x);
  v_f32_st_tnsr_b((1024 + 256) / 32, vmem, y);

  float8_128 x2 = v_f32_ld_tnsr_b(128 / 32, vmem);
  float8_128 y2 = v_f32_ld_tnsr_b((1024 + 256) / 32, vmem);

  x = x2 + y2;
  y = x + x;

  printf("2q\n");
  // v_f32_st_tnsr_st(128 / 32, vmem, 256 / 128, x);
  v_f32_st_tnsr_st((1024 + 256) / 32, vmem, 384 / 128, y);

  // x2 = v_f32_ld_tnsr_st(128 / 32, vmem, 256 / 128);
  y2 = v_f32_ld_tnsr_st((1024 + 256) / 32, vmem, 384 / 128);

  x = x2 + y2;
  y = x + x;

  printf("3q\n");
  // v_f32_st_tnsr_st_msk(128 / 32, vmem, 256 / 128, 0b10100101, x);
  v_f32_st_tnsr_st_msk((1024 + 256) / 32, vmem, 384 / 128, 0b11110000, y);

  // x2 = v_f32_ld_tnsr_st_msk(128 / 32, vmem, 256 / 128, 0b10100101);
  y2 = v_f32_ld_tnsr_st_msk((1024 + 256) / 32, vmem, 384 / 128, 0b11110000);

  x = x2 + y2;
  y = x + x;

  Print("x = ", x, PrintType::FLOAT);
}

void test_x86_v_s32_cmp() {
  int8_128 x(0x1);
  int8_128 y(0xF0F0F0F0);

  int8_128 res = v_u32_and(x, y);

  Print("test_x86_v_s32_cmp", __$F(res), PrintType::HEX);
}

void test_x86_v_u32_and() {
  int8_128 x(0xAAAAAAAA);
  int8_128 y(0xF0F0F0F0);

  int8_128 res = v_u32_and(x, y);

  Print("test_x86_v_u32_and", __$F(res), PrintType::HEX);
}

void test_x86_v_u32_shr() {
  int8_128 G(0xFFFFFFFF);

  int8_128 f2 = v_u32_shr(G, 1);

  Print("test_x86_v_u32_shr", __$F(f2), PrintType::BIT);
}

void test_x86_v_f32_add_b() {
  int8_128 G = get_core_id();
  float8_128 f = v_cvt_itof(G);

  float8_128 f2 = v_f32_add_b(f, f);

  Print("test_x86_v_f32_add_b", f2, PrintType::FLOAT);
}

void test_x86_v_f32_mul_b() {
  int8_128 G = get_core_id();
  float8_128 f = v_cvt_itof(G);

  float8_128 f2 = v_f32_mul_b(f, f);

  Print("test_x86_v_f32_mul_b", f2, PrintType::FLOAT);
}

void test_x86_frcp() {
  int8_128 G = get_core_id();
  float8_128 f = v_cvt_itof(G);

  float8_128 f2 = __dlc_frcp_rd_without_unary(f);

  Print("test_x86_frcp", f2, PrintType::FLOAT);
}
  
void test_x86_row_rotate() {
  int8_128 G = get_core_id();

  int8_128 x = __$S(v_row_rotate(__$F(G), 0));
  int8_128 y = __$S(v_row_rotate(__$F(G), 1));

  Print("test_x86_row_rotate", __$F(x), PrintType::INT);
  Print("test_x86_row_rotate", __$F(y), PrintType::INT);
}

void test_x86_mti_rotate() {
  int8_128 G = get_core_id();

  int8_128 x = __$S(m_rotate(__$F(G), -1, 0)); // -1
  int8_128 y = __$S(m_rotate(__$F(x), 2, 1));    // +2
  m_rotate_single(__$F(y), -3, 0);
  int8_128 a = __$S(m_pop_trf(0));
  m_rotate_single(__$F(a), 4, 1);
  int8_128 b = __$S(m_pop_trf(1));

  Print("test_x86_mti_rotate", __$F(b), PrintType::INT);
}

void test_v_u32_shr() {
  auto core_id = get_core_id();
  auto mask = v_s32_cmp(LS, core_id, 64);
  // int8_128 G = v_s32_sel(mask, 0x7FFFFFFF, 0xFFFFFFFF);
  // Print("G", G, PrintType::BIT, true);
  
  // auto res = v_u32_shr(G, core_id);
  // Print("v_u32_shr", res, PrintType::BIT, true);

  // auto clz = v_u32_clz(res);
  // Print("v_u32_clz", clz, PrintType::INT, true);

  // auto clz2 = 63 - clz;
  // Print("int - int8_128", clz2, PrintType::INT, true);

  // auto f = v_u32_move_f(-1);
  // Print("v_u32_move_f", f, PrintType::FLOAT, true);

  auto move_b = v_u32_move_b(2);
  Print("move_b", move_b, PrintType::HEX, true);



  // auto res = v_u32_shl(G, core_id);
  // Print("v_u32_shl", res, PrintType::BIT, false);
}

void test_v_s32_sel() {
  // auto core_id = get_core_id();
  // auto mask = v_s32_cmp(LS, core_id, 64);

}

void test_matmul() {
  float8_128 core_id = v_cvt_itof(get_core_id());

  bool select = 0;

  for (int i = 0; i < 18; ++i) {
    push_gsnf(core_id * (i + 1.0), select);
  }

  float8_128 ans = m_fakemul(core_id, 0, select);
  // Print("ans = ", ans, PrintType)

  for (int i = 0; i < 128; ++i) {
    for (int j = 0; j < 128; ++j) {
      printf("%f ", dlc_memorys._gmr[get_device_id()][select][i * 128 + j]);
    }
    puts("");
  }

  m_matmul_single(core_id, 0, select);
  ans = m_pop_mrf(select);
  
  Print("ans = ", ans, PrintType::HEX);
  printf("mrf.size = %d\n", dlc_memorys.mrf[get_device_id()][select].size());
}

void main_x86(SIM_X86::DLCMem *INFO) {
  test_matmul();

  // test_x86_vmem_ldst(*(SIM_X86::tensor*)INFO->vmem_addr);

  // test_v_u32_shr();

  // test_x86_mti_rotate();

  // test_x86_row_rotate();

  // test_x86_v_f32_add_b();

  // test_x86_v_f32_mul_b();

  // test_x86_frcp();

  // test_x86_v_u32_shr();

  // test_x86_v_u32_and();

  // sync_device();
}
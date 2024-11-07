#ifndef _TYPEHINT_H_X86_
#define _TYPEHINT_H_X86_

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

inline void Tensor2Vector32(const syn::nn::Tensor& input, float* hbm);
inline void Tensor2Vector16(const syn::nn::Tensor& input, float* hbm);
inline void Vector2Tensor32(float* hbm, syn::nn::Tensor& input);
inline void Vector2Tensor16(float* hbm, syn::nn::Tensor& input);
inline void Tensor2Vector(const syn::nn::Tensor& input, float* hbm);
inline void Vector2Tensor(float* hbm, syn::nn::Tensor& input);

#define ACTUAL_VMEM_SIZE 4096 * 1024

int MIN_VMEM_SIZE = 4096 * 1024;

/** TODO:
 * int8_128 * int8_128: 边界问题
 * bool8_128: 使用 std::bitset<1024> 会有问题，std::array<bool, 1024>就没事
 * 
*/

/* do not delete */
enum {
  SMEM = 0,
  HBM = 1,
  VMEM = 2,
  CMEM = 3,
};

enum {
  EQ = 0,
  NEQ = 1,
  LS = 2,
  GT = 3,
  LSEQ = 4,
  GTEQ = 5,
};

enum { UNDONE = 0, DONE = 1 };
/* end */


struct float8_128 {
  // std::array<float, 1024> data;

  // float8_128() {}
  // float8_128(int v) { data.fill(v); }
  // float8_128(float v) { data.fill(v); }
  // float8_128(double v) { data.fill(float(v)); }
  
  std::vector<float> data;

  float8_128() : data(1024, 0.f) {}
  float8_128(int v) : data(1024, v) {}
  float8_128(unsigned int v) : data(1024, v) {}
  float8_128(float v) : data(1024, v) {}
  float8_128(double v) : data(1024, v) {}
  float8_128(const float8_128& other) : data(other.data) {}

  float8_128& operator=(float v) {
    // std::fill(data.begin(), data.end(), v);
    for (int i = 0; i < 1024; ++i) this->data[i] = v;
    return *this;
  }

  float8_128& operator=(const float8_128& other) {
    if (this != &other) {
      this->data = other.data;
    }
    return *this;
  }

  float& operator[](std::size_t index) {
    if (index >= data.size()) {
      throw std::out_of_range("float& operator[]: Index out of range");
    }
    return data[index];
  }

  const float& operator[](std::size_t index) const {
    if (index >= data.size()) {
      throw std::out_of_range("const float& operator[]: Index out of range");
    }
    return data[index];
  }

  float8_128 operator+(float8_128 x) {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = data[i] + x[i];
    return res;
  }

  friend float8_128 operator+(const float& y, float8_128 x) {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = y + x[i];
    return res;
  }

  float8_128& operator+=(float8_128 x) {
    for (int i = 0; i < 1024; ++i) this->data[i] += x[i];
    return *this;
  }

  float8_128 operator-(const float8_128& x) const {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = data[i] - x[i];
    return res;
  }

  friend float8_128 operator-(const float& y, float8_128 x) {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = y - x[i];
    return res;
  }

  float8_128 operator-() const {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = -this->data[i];
    return res;
  }

  friend float8_128 operator*(const float& y, float8_128 x) {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = y * x[i];
    return res;
  }

  float8_128 operator*(float8_128 x) {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = data[i] * x[i];
    return res;
  }

  float8_128& operator*=(float8_128 x) {
    for (int i = 0; i < 1024; ++i) this->data[i] *= x[i];
    return *this;
  }

  friend float8_128 operator/(const float& y, float8_128 x) {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = y / x[i];
    return res;
  }

  float8_128 operator/(float8_128 x) {
    float8_128 res = 0;
    for (int i = 0; i < 1024; ++i) res[i] = data[i] / x[i];
    return res;
  }
};

struct int8_128 {
  // std::array<int, 1024> data;

  // int8_128() {}
  // int8_128(int v) { data.fill(v); }

  std::vector<int> data;

  int8_128() : data(1024, 0) {}
  int8_128(int v) : data(1024, v) {}
  int8_128(const int8_128& other) : data(other.data) {}

  int8_128& operator=(int v) {
    std::fill(data.begin(), data.end(), v);
    return *this;
  }

  int8_128& operator=(const int8_128& other) {
    if (this != &other) {
      data = other.data;
    }
    return *this;
  }

  int& operator[](std::size_t index) {
    if (index >= data.size()) {
      throw std::out_of_range("Index out of range");
    }
    return data[index];
  }

  const int& operator[](std::size_t index) const {
    if (index >= data.size()) {
      throw std::out_of_range("Index out of range");
    }
    return data[index];
  }

  #define INT_MAX 0x7FFFFFFF
  #define INT_MIN (-INT_MAX - 1)
  int8_128 operator+(const int8_128& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      int64_t res = static_cast<int64_t>(this->data[i]) + static_cast<int64_t>(other[i]);

      if (res > INT_MAX) result[i] = INT_MAX;
      else if (res < INT_MIN) result[i] = INT_MIN;
      else result[i] = this->data[i] + other[i];
    }
    return result;
  }

  int8_128 operator+(const int& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      int64_t res = static_cast<int64_t>(this->data[i]) + static_cast<int64_t>(other);

      if (res > INT_MAX) result[i] = INT_MAX;
      else if (res < INT_MIN) result[i] = INT_MIN;
      else result[i] = this->data[i] + other;
    }
    return result;
  }

  friend int8_128 operator+(int lhs, const int8_128& rhs) {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      int64_t res = static_cast<int64_t>(lhs) + static_cast<int64_t>(rhs[i]);

      if (res > INT_MAX) result[i] = INT_MAX;
      else if (res < INT_MIN) result[i] = INT_MIN;
      else result[i] = lhs + rhs[i];
    }
    return result;
  }

  int8_128& operator+=(const int& other) {
    for (int i = 0; i < 1024; ++i) { 
      int64_t res = static_cast<int64_t>(this->data[i]) + static_cast<int64_t>(other);

      if (res > INT_MAX) this->data[i] = INT_MAX;
      else if (res < INT_MIN) this->data[i] = INT_MIN;
      else this->data[i] = this->data[i] + other;
    }
    return *this;
  }

  int8_128& operator+=(const int8_128& other) {
    for (int i = 0; i < 1024; ++i) {
      int64_t res = static_cast<int64_t>(this->data[i]) + static_cast<int64_t>(other[i]);

      if (res > INT_MAX) this->data[i] = INT_MAX;
      else if (res < INT_MIN) this->data[i] = INT_MIN;
      else this->data[i] = this->data[i] + other[i];
    }
    return *this;
  }

  int8_128 operator-(const int8_128& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      int64_t res = static_cast<int64_t>(this->data[i]) - static_cast<int64_t>(other[i]);

      if (res > INT_MAX) result[i] = INT_MAX;
      else if (res < INT_MIN) result[i] = INT_MIN;
      else result[i] = this->data[i] - other[i];
    }
    return result;
  }

  int8_128 operator-(const int& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      int64_t res = static_cast<int64_t>(this->data[i]) - static_cast<int64_t>(other);

      if (res > INT_MAX) result[i] = INT_MAX;
      else if (res < INT_MIN) result[i] = INT_MIN;
      else result[i] = this->data[i] - other;
    }
    return result;
  }

  friend int8_128 operator-(int lhs, const int8_128& rhs) {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      int64_t res = static_cast<int64_t>(lhs) - static_cast<int64_t>(rhs[i]);

      if (res > INT_MAX) result[i] = INT_MAX;
      else if (res < INT_MIN) result[i] = INT_MIN;
      else result[i] = lhs - rhs[i];
    }
    return result;
  }

  int8_128 operator-() const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result = -this->data[i];
    }
    return result;
  }

  int8_128 operator*(int other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] * other;
    }
    return result;
  }

  friend int8_128 operator*(int lhs, const int8_128& rhs) {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      int64_t res = static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs[i]);

      if (res > INT_MAX) result[i] = INT_MAX;
      else if (res < INT_MIN) result[i] = INT_MIN;
      else result[i] = lhs * rhs[i];
    }
    return result;
  }

  int8_128 operator/(const int& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] / other;
    }
    return result;
  }

  int8_128 operator%(const int& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] % other;
    }
    return result;
  }

  int8_128 operator&(const int8_128& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] & other[i];
    }
    return result;
  }

  int8_128 operator&(const int& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] & other;
    }
    return result;
  }

  int8_128& operator&=(const int& other) {
    for (int i = 0; i < 1024; ++i) {
      this->data[i] &= other;
    }
    return *this;
  }

  int8_128 operator|(const int8_128& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] | other[i];
    }
    return result;
  }

  int8_128 operator|(const int& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] | other;
    }
    return result;
  }

  int8_128& operator|=(const int8_128& other) {
    for (int i = 0; i < 1024; ++i) {
      this->data[i] |= other[i];
    }
    return *this;
  }

  int8_128 operator^(const int8_128& other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] ^ other[i];
    }
    return result;
  }

  int8_128 operator~() const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = ~this->data[i];
    }
    return result;
  }

  int8_128 operator<<(int other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] << other;
    }
    return result;
  }

  int8_128 operator>>(int other) const {
    int8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] >> other;
    }
    return result;
  }
};

struct bool8_128 {
  // std::vector<int> data;
  
  // bool8_128() : data(1024) {}
  // bool8_128(bool v) : data(1024, v) {}
  // bool8_128(int v) : data(1024, v) {}

  // int& operator[](std::size_t index) {
  //   assert(index < data.size() && "bool& operator[]: Index out of range");

  //   return data[index];
  // } 

  // const int& operator[](std::size_t index) const {
  //   assert(index < data.size() && "const bool& operator[]: Index out of range");

  //   return data[index] & 1;
  // }

  // bool8_128 operator&(const bool8_128& rhs) const {
  //   bool8_128 result;
  //   for (int i = 0; i < 1024; ++i) {
  //     result[i] = this->data[i] & rhs[i];
  //     result[i] &= 1;
  //   }
  //   return result;
  // }

  // bool8_128 operator|(const bool8_128& rhs) const {
  //   bool8_128 result;
  //   for (int i = 0; i < 1024; ++i) {
  //     result[i] = this->data[i] | rhs[i];
  //     result[i] &= 1;
  //   }
  //   return result;
  // }

  // bool8_128 operator!() const {
  //   bool8_128 result;
  //   for (int i = 0; i < 1024; ++i) {
  //     result[i] = !this->data[i];
  //     result[i] &= 1;
  //   }
  //   return result;
  // }

  // bool8_128 operator^(const bool8_128& rhs) const {
  //   bool8_128 result;
  //   for (int i = 0; i < 1024; ++i) {
  //     result[i] = this->data[i] ^ rhs[i];
  //     result[i] &= 1;
  //   }
  //   return result;
  // }

  std::array<bool, 1024> data;
  
  bool8_128() {}
  bool8_128(bool v) { data.fill(v); }

  bool& operator[](std::size_t index) {
    assert(index < data.size() && "bool& operator[]: Index out of range");

    return data[index];
  } 

  const bool& operator[](std::size_t index) const {
    assert(index < data.size() && "const bool& operator[]: Index out of range");

    return data[index];
  }

  bool8_128 operator&(const bool8_128& rhs) const {
    bool8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] & rhs[i];
    }
    return result;
  }

  bool8_128 operator|(const bool8_128& rhs) const {
    bool8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] | rhs[i];
    }
    return result;
  }

  bool8_128 operator!() const {
    bool8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = !this->data[i];
    }
    return result;
  }

  bool8_128 operator^(const bool8_128& rhs) const {
    bool8_128 result;
    for (int i = 0; i < 1024; ++i) {
      result[i] = this->data[i] ^ rhs[i];
    }
    return result;
  }

  // std::bitset<1024> data;

  // bool8_128() : data(false) {}
  // bool8_128(bool v) : data(v) {}
  // bool8_128(const std::bitset<1024>& v) : data(v) {}
  
  // // 非const版本的[]运算符
  // std::bitset<1024>::reference operator[](size_t pos) {
  //   return data[pos];
  // }

  // // const版本的[]运算符，用于只读访问
  // bool operator[](size_t pos) const {
  //   return data[pos];
  // }

  // bool8_128 operator&(const bool8_128& rhs) const {
  //   return (this->data & rhs.data);
  // }

  // bool8_128 operator|(const bool8_128& rhs) const {
  //   return (this->data | rhs.data);
  // }

  // bool8_128 operator!() const {
  //   return (~this->data);
  // }

  // bool8_128 operator^(const bool8_128& rhs) const {
  //   return (this->data ^ rhs.data);
  // }
};

struct float128_128 {
  std::vector<float> data;
  int idx;

  float128_128() : data(128 * 128), idx(15) {}
  float128_128(float v) : data(128 * 128, v), idx(15) {}

  float& operator[](std::size_t index) {
    if (index >= data.size()) {
      throw std::out_of_range("float128_128 float& operator[]: Index out of range");
    }
    return data[index];
  }

  const float& operator[](std::size_t index) const {
    if (index >= data.size()) {
      throw std::out_of_range("float128_128 const float& operator[]: Index out of range");
    }
    return data[index];
  }
};

struct int128_128 {
  std::vector<int> data;

  int128_128() : data(128 * 128) {}
  int128_128(int v) : data(128 * 128, v) {}
};

struct float128_128_2 {
  std::vector<float> data;

  float128_128_2() : data(128 * 128 * 2) {}
  float128_128_2(float v) : data(128 * 128 * 2, v) {}
};

struct unsigned128 {
  std::array<unsigned, 128> data;

  unsigned& operator[](std::size_t index) {
    if (index >= data.size()) {
      throw std::out_of_range("unsigned128 unsigned& operator[]: Index out of range");
    }
    return data[index];
  }

  const unsigned& operator[](std::size_t index) const {
    if (index >= data.size()) {
      throw std::out_of_range("unsigned128 const unsigned& operator[]: Index out of range");
    }
    return data[index];
  }
};

struct short8_128 {
  std::array<short, 1024> data;
};

struct char8_128 {
  std::array<char, 1024> data;
};

typedef short __bf16;

#define bool1024 bool8_128

typedef union {
  float f32;
  uint32_t u32;
} dlc_dtype;

inline dlc_dtype getDlcDtype(uint32_t x) {
  dlc_dtype y;
  y.u32 = x;
  return y;
}

inline dlc_dtype getDlcDtype(float x) {
  dlc_dtype y;
  y.f32 = x;
  return y;
}

inline std::vector<dlc_dtype> getDlcDtype(float8_128 x) {
  std::vector<dlc_dtype> y(1024);
  for (int i = 0; i < 1024; ++i) y[i].f32 = x[i];
  return y;
}

inline std::vector<dlc_dtype> getDlcDtype(const int8_128& x) {
  std::vector<dlc_dtype> y(1024);
  for (int i = 0; i < 1024; ++i) y[i].u32 = x[i];
  return y;
}

namespace SIM_X86 {
  // designed size
  const int64_t designed_smem_size = 256 * 1024;             // 256 K
  const int64_t designed_vmem_size = 4 * 1024 * 1024;        // 4 M
  const int64_t designed_cmem_size = 8 * 1024 * 1024;        // 8 M
  const int64_t designed_hbm_size  = 1 * 1024 * 1024 * 1024; // 1 G

  enum DLCType {
    dlc_int8 = 1,
    dlc_uint8 = 2,
    dlc_bool = 3,
    dlc_int16 = 4,
    dlc_fp16 = 5,
    dlc_bf16 = 6
  };

  enum RoundFormat {
    ROUND = 1, //takes the higher 16bits, and round tie to even.
    TRUNCATE = 2, // trancate.
    LOWER_ROUND = 3 // takes the lower 16bits and round tie to even.
  };

  enum class FakeMulType {
    GSNF = false,
    GSTF = true
  };

  enum class MatMulType {
    NORMAL = 0,
    GSNF = 1,
    GSTF = 2
  };

  enum class PrintType {
    FLOAT = 0,
    INT = 1,
    HEX = 2,
    BIT = 3
  };

  enum class CMP {
    EQ = 0,
    NEQ = 1,
    LS = 2,
    GT = 3,
    LSEQ = 4,
    GTEQ = 5
  };

  class Barrier {  // 线程同步
  public:
    Barrier() : thread_count(2) {}
    explicit Barrier(int count) : thread_count(count), counter(0), waiting(0) {}

    void Wait() {
      std::unique_lock<std::mutex> lock(mutex_);
      ++counter;
      ++waiting;

      if (waiting < thread_count) {
        // cond_.wait(lock, [this] { return waiting >= thread_count; });
        cond_.wait(lock);
      } else {
        waiting = 0;         // 重置计数以便重新使用屏障
        cond_.notify_all();  // 唤醒所有等待的线程
      }
    }

    // private:
    const int thread_count;         // 线程的数量
    int counter;                    // 当前到达屏障的线程数
    int waiting;                    // 当前等待的线程数
    std::mutex mutex_;              // 保护共享资源的互斥锁
    std::condition_variable cond_;  // 用于同步的条件变量
  };

  class tensor {
  public:
    // const float* __PTR; // 整个memory的起点
    // const int64_t __LEN; // 整个memory的大小
    // const int type; // 0: smem, 1: vmem, 2: cmem, 3: hbm
    float* __PTR; // 整个memory的起点
    int64_t __LEN; // 整个memory的大小
    int type; // 0: smem, 1: vmem, 2: cmem, 3: hbm

    float* data_ptr; // 当前tensor的地址
    int64_t data_size; // 当前tensor能用的大小

    tensor() : __PTR(nullptr), __LEN(0), data_ptr(nullptr),
              data_size(0), type(-1) {}

    tensor(float* ptr, int64_t sz) : __PTR(ptr), __LEN(sz), data_ptr(ptr),
                                    data_size(sz), type(-1) {}

    tensor(float* ptr, int64_t sz, int type) : __PTR(ptr), __LEN(sz), data_ptr(ptr),
                                              data_size(sz), type(type) {}
    tensor(float* __PTR, int64_t __LEN, 
          float* ptr, int64_t sz, int type) : __PTR(__PTR), __LEN(__LEN), data_ptr(ptr),
                                              data_size(sz), type(type) {}

    tensor(const tensor& t) : __PTR(t.__PTR), __LEN(t.__LEN), data_ptr(t.data_ptr),
                              data_size(t.data_size), type(t.type) {}

    // 浅拷贝
    tensor& operator=(const tensor& t) {
      assert(this->type == -1 || this->type == t.type);
      this->__PTR = t.__PTR;
      this->__LEN = t.__LEN;
      this->type = t.type;

      this->data_ptr = t.data_ptr;
      this->data_size = t.data_size;

      return *this;
    }

    explicit operator int() const {
      return static_cast<int>(data_size);
    }

    tensor operator+(const int64_t& off) const {
      int nlen = this->data_size - off * 32 / 128 * 128;
      if (!(nlen >= 0 && nlen <= this->__LEN)) {
        printf("off = %d, data_size = %d, type = %d, data_ptr = %x, __PTR = %x, __LEN = %d\n",
               off, this->data_size, this->type, this->data_ptr, this->__PTR, this->__LEN);
      }
      assert(nlen >= 0 && nlen <= this->__LEN);

      tensor re = *this;
      re.data_ptr += off * 32 / 128 * 128;
      re.data_size -= off * 32 / 128 * 128;
      if (this->type == 1)
        MIN_VMEM_SIZE = (MIN_VMEM_SIZE > re.data_size ? re.data_size : MIN_VMEM_SIZE);
      return re;
    }

    tensor operator-(const int64_t& off) const {
      int nlen = this->data_size - off * 32 / 128 * 128;
      assert(nlen >= 0 && nlen <= this->__LEN);

      tensor re = *this;
      re.data_ptr += off * 32 / 128 * 128;
      re.data_size -= off * 32 / 128 * 128;
      if (this->type == 1)
        MIN_VMEM_SIZE = (MIN_VMEM_SIZE > re.data_size ? re.data_size : MIN_VMEM_SIZE);
      return re;
    }

    tensor& operator+=(const int64_t& off) {
      int nlen = this->data_size - off * 32 / 128 * 128;
      assert(nlen >= 0 && nlen <= this->__LEN);
    
      this->data_ptr += off * 32 / 128 * 128;
      this->data_size -= off * 32 / 128 * 128;
      if (this->type == 1)
        MIN_VMEM_SIZE = (MIN_VMEM_SIZE > this->data_size ? this->data_size : MIN_VMEM_SIZE);
      return *this;
    }

    float& operator[](const int64_t& index) {    
      int nlen = this->data_size - index;
      // 这里判定 nlen > 0，稍微不太一样，因为这里要的是index，index∈[0, data_len - 1], 但是如果是offset则是[1, data_len]
      assert(nlen > 0 && nlen <= this->__LEN);

      return data_ptr[index];
    }
  };

  struct DLCTensor {
    #define DLC_MAX_DIM 5

    // Tensor
    tensor* address;
    unsigned dtype;                // Tensor Data Type
    unsigned shape[DLC_MAX_DIM];   // Tensor Sizes (elements)
    unsigned stride[DLC_MAX_DIM];  // Tensor Strides (elements)
    unsigned storage_offset;       // Tensor Storage Offset (elements)
    unsigned storage_offset_high;  // Tensor Storage Offset high 32 bits if overflow
    // Storage
    unsigned dim0;           // Lowest Dimension
    unsigned dim1;           // Other Dimensions
    unsigned dim0_padded;    // Padded Lowest Dimension
    unsigned layout;         // DLCLayout
    unsigned is_conj;        // Is Conjugate
    unsigned memory_format;  // Memory Format
    unsigned reserved;       // Reserved

    DLCTensor() {}

    DLCTensor(tensor t, const syn::nn::Tensor& input, bool p = true) {
      this->address = new tensor(t);

      for (int i = 0; i < 5; ++i) this->shape[i] = input.size(i);
      for (int i = 0; i < 5; ++i) this->stride[i] = input.stride(i);

      this->dim0 = input.size(0);
      this->dim1 = input.dlc_dim1();
      this->dim0_padded = input.dlc_dim0_padded();

      if (p) {
        Tensor2Vector(input, this->address->data_ptr);
      }
    }

    ~DLCTensor() {
      delete this->address;  // 注意析构时释放内存
    }
  };

  struct DLCScalar{
    uint32_t value;
    uint32_t value_high;
    uint32_t image;  // for complex data type
    uint32_t image_high;
    uint32_t dtype;  // Scalar Data Type, dlc scalar has 1-16 bytes

    DLCScalar() : value(0) {}
    DLCScalar(bool value) { this->value = value; }
    DLCScalar(uint32_t value) { this->value = value; }
    DLCScalar(int value) { this->value = value; }
    DLCScalar(float value) { this->value = *reinterpret_cast<uint32_t*>(&value); }
  };

  struct DLCMem {
    int smem_size;
    int vmem_size;
    int cmem_size;

    // 理论上，之后所有的tensor都必须从这里出去，不然几个const就失效了
    // 也就是它们溯源的头必须是这几个tensor
    tensor* smem_addr;
    tensor* vmem_addr;
    tensor* cmem_addr;

    DLCMem() : smem_size(0), vmem_size(0), cmem_size(0), 
              smem_addr(nullptr), vmem_addr(nullptr), cmem_addr(nullptr)  {}

    DLCMem(float* smem, float* vmem, float* cmem, int _smem_size, int _vmem_size, int _cmem_size) {
      // DLCsim 似乎保存的是Byte的长度，所以 *4
      smem_size = _smem_size * 4;
      vmem_size = _vmem_size * 4;
      cmem_size = _cmem_size * 4;
    
      smem_addr = new tensor(smem, _smem_size, 0);
      vmem_addr = new tensor(vmem, designed_vmem_size, vmem, _vmem_size, 1);
      cmem_addr = new tensor(cmem, _cmem_size, 2);
    }

    DLCMem& operator=(const DLCMem& t) {
      this->smem_size = t.smem_size;
      this->vmem_size = t.vmem_size;
      this->cmem_size = t.cmem_size;

      smem_addr = new tensor(*t.smem_addr);
      vmem_addr = new tensor(*t.vmem_addr);
      cmem_addr = new tensor(*t.cmem_addr);

      return *this;
    }

    ~DLCMem() {
      delete smem_addr;
      delete vmem_addr;
      delete cmem_addr;
    }
  };

  struct TensorInfo {
    int SpaceSize[5];

    TensorInfo() {}
  };

  struct DLC_MEMORYS {
    // memorys
    float* smem_xys0;
    float* smem_xys1;
    float* vmem_xys0;
    float* vmem_xys1;
    float* cmem;
    float* hbm;
    tensor* hbm_tensor;

    // fifo & gmr
    // urf: 32, mrf: 16, trf:32, fxc: 32
    float128_128 _gmr[2][2];           // _gmr[2][2]
    std::deque<float8_128> gstf[2][2]; // gsnf
    std::deque<float8_128> gsnf[2][2]; // gstf
    std::deque<float8_128> crf[2];     // fxc
    std::deque<float8_128> mrf[2][2];  // mti
    std::deque<float8_128> trf[2][2];  // transpose max=32
    std::deque<float8_128> urf[2];     // unary

    std::vector<bool8_128> vmask[2][8];

    int8_128 _pcr[2][2]; // Permute Control Register

    int _transpose_width[2][2]; // save trnapose width from transpose_start & transpose_start_end
    std::deque<float8_128> _transpose_buffer[2][2];

    DLCMem info_xys0;
    DLCMem info_xys1;

    DLC_MEMORYS() {
      smem_xys0 = new float[designed_smem_size];  // 256K
      smem_xys1 = new float[designed_smem_size];  // 256K
      vmem_xys0 = new float[designed_vmem_size];  // 4M
      vmem_xys1 = new float[designed_vmem_size];  // 4M
      cmem      = new float[designed_cmem_size];  // 8M
      hbm       = new float[designed_hbm_size];   // 4G
    
      hbm_tensor = new tensor(hbm, designed_hbm_size, 3);

      info_xys0 = DLCMem(smem_xys0, vmem_xys0, cmem, designed_smem_size, ACTUAL_VMEM_SIZE, designed_cmem_size);
      info_xys1 = DLCMem(smem_xys1, vmem_xys1, cmem, designed_smem_size, ACTUAL_VMEM_SIZE, designed_cmem_size);
    
      uint32_t* ptr;
      ptr = reinterpret_cast<uint32_t*>(smem_xys0);
      for (int64_t i = 0; i < designed_smem_size; ++i) { ptr[i] = 0xdeadbeef; }
      ptr = reinterpret_cast<uint32_t*>(smem_xys1);
      for (int64_t i = 0; i < designed_smem_size; ++i) { ptr[i] = 0xdeadbeef; }
      ptr = reinterpret_cast<uint32_t*>(vmem_xys0);
      for (int64_t i = 0; i < designed_vmem_size; ++i) { ptr[i] = 0xdeadbeef; }
      ptr = reinterpret_cast<uint32_t*>(vmem_xys1);
      for (int64_t i = 0; i < designed_vmem_size; ++i) { ptr[i] = 0xdeadbeef; }
      ptr = reinterpret_cast<uint32_t*>(cmem);
      for (int64_t i = 0; i < designed_cmem_size; ++i) { ptr[i] = 0xdeadbeef; }
      ptr = reinterpret_cast<uint32_t*>(hbm);
      for (int64_t i = 0; i < designed_hbm_size ; ++i) { ptr[i] = 0xdeadbeef; }
    }

    tensor hbm_alloc(int64_t length) {
      tensor t = *hbm_tensor;
      t.data_size = length;

      (*hbm_tensor) += length / 32;

      return t;
    }

    tensor hbm_alloc(const syn::nn::Tensor& input) {
      int64_t len = int64_t(input.dlc_dim0_padded()) * int64_t(input.dlc_dim1());

      return hbm_alloc(len);
    }

    ~DLC_MEMORYS() {
      delete[] smem_xys0;
      delete[] smem_xys1;
      delete[] vmem_xys0;
      delete[] vmem_xys1;
      delete[] cmem;
      delete[] hbm;
      delete hbm_tensor;
    }
  };
}  // namespace SIM_X86

// extern int sharedVariable;
// extern std::mutex sharedMutex;
// extern SIM_X86::DLC_MEMORYS dlc_memorys;
// extern SIM_X86::Barrier dlc_barrier;

// extern std::condition_variable barrier_cond_;
// extern std::mutex barrier_mutex;
// extern int barrier_thread_count;
// extern int barrier_count;
// extern int barrier_waiting;

int sharedVariable = 0;
std::mutex sharedMutex;
SIM_X86::DLC_MEMORYS dlc_memorys = SIM_X86::DLC_MEMORYS();
SIM_X86::Barrier dlc_barrier(2);

std::condition_variable barrier_cond_;
std::mutex barrier_mutex;
int barrier_thread_count = 2;
int barrier_count = 0;
int barrier_waiting = 0;

template <typename T, typename Func>
T process1024(T x, Func func) {
  for (int i = 0; i < 1024; ++i) {
    x[i] = func(x[i]);
  }

  return x;
}

template <typename T, typename Func>
T process1024(T x, const T& y, Func func) {
  for (int i = 0; i < 1024; ++i) {
    x[i] = func(x[i], y[i]);
  }

  return x;
}

inline int dlc_get_device_id() {
  std::thread::id this_id = std::this_thread::get_id();
  std::size_t hashed_id = std::hash<std::thread::id>{}(this_id);
  int int_id = static_cast<int>(hashed_id);

  std::lock_guard<std::mutex> lock(sharedMutex);
  if (sharedVariable == 0) {
    sharedVariable = int_id;
  }

  return (sharedVariable == int_id);
}


/* result buffers */
#define kFxcBufferSize 32
#define kTransposeBufferSize 32
#define kMatrixBufferSize 16
#define kUnaryBufferSize 32

inline void dlc_m_push_mrf(const bool& select, float8_128 x) {
  int device_id = dlc_get_device_id();
  if (dlc_memorys.mrf[device_id][select].size() < kMatrixBufferSize) {
    dlc_memorys.mrf[device_id][select].push_back(x);
  } else {
    printf("ERROR: dlc_m_push_mrf while mrf.size == 16, device_id = %d, select = %d\n", device_id, select);
    assert(false);
  }
}

inline float8_128 dlc_m_pop_mrf(const bool& select) {
  int device_id = dlc_get_device_id();
  if (dlc_memorys.mrf[device_id][select].size()) {
    float8_128 x = dlc_memorys.mrf[device_id][select].front();
    dlc_memorys.mrf[device_id][select].pop_front();
    #ifdef DEBUG_MRF_XYS0
    if (dlc_get_device_id() == 0) {
      printf("[XYS%d]: pop mrf in mtr_select = %d, fifo_size = %d\n", device_id, select, dlc_memorys.mrf[device_id][select].size() + 1);
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 128; ++j) {
          printf("%f ", x[i * 128 + j]);
        }
        puts("");
      }
    }
    #endif
    return x;
  } else {
    printf("ERROR: dlc_m_pop_mrf, mrf.size = 0, device_id = %d, select = %d\n", device_id, select);
    assert(false);
  }
}

inline void dlc_m_push_crf(float8_128 x) {
  int device_id = dlc_get_device_id();

  if (dlc_memorys.crf[device_id].size() < kFxcBufferSize) {
    dlc_memorys.crf[device_id].push_back(x);
  } else {
    printf("ERROR: dlc_m_push_crf while crf is full, device_id = %d\n", device_id);
    assert(false);
  }
}

inline float8_128 dlc_m_pop_crf() {
  int device_id = dlc_get_device_id();

  if (dlc_memorys.crf[device_id].size()) {
    float8_128 x = dlc_memorys.crf[device_id].front();
    dlc_memorys.crf[device_id].pop_front();
    return x;
  } else {
    printf("ERROR: dlc_m_pop_crf while crf is empty, device_id = %d\n", device_id);
    assert(false);
  }
}

inline void dlc_m_push_trf(const bool& select, float8_128 x) {
  int device_id = dlc_get_device_id();

  if (dlc_memorys.trf[device_id][select].size() < kTransposeBufferSize) {
    dlc_memorys.trf[device_id][select].push_back(x);
  } else {
    printf("ERROR: dlc_m_push_trf while trf.size == 16, device_id = %d, select = %d\n", device_id, select);
    assert(false);
  }
}

inline float8_128 dlc_m_pop_trf(const bool& select) {
  int device_id = dlc_get_device_id();

  if (dlc_memorys.trf[device_id][select].size()) {
    float8_128 x = dlc_memorys.trf[device_id][select].front();
    dlc_memorys.trf[device_id][select].pop_front();
    return x;
  } else {
    printf("ERROR: dlc_m_pop_trf while trf.size == 0, device_id = %d, select = %d\n", device_id, select);
    assert(false);
  }
}

inline void dlc_m_push_urf(float8_128 x) {
  int device_id = dlc_get_device_id();

  if (dlc_memorys.urf[device_id].size() < kUnaryBufferSize) {
    dlc_memorys.urf[device_id].push_back(x);
  } else {
    printf("ERROR: dlc_m_push_urf while urf is full, device_id = %d\n", device_id);
    assert(false);
  }
}

inline float8_128 dlc_m_pop_urf() {
  int device_id = dlc_get_device_id();

  if (dlc_memorys.urf[device_id].size()) {
    float8_128 x = dlc_memorys.urf[device_id].front();
    dlc_memorys.urf[device_id].pop_front();
    return x;
  } else {
    printf("ERROR: dlc_m_pop_urf while urf is empty, device_id = %d\n", device_id);
    assert(false);
  }
}
/* end */


/* gsnf, gstf, gmr */
#define kGsnfGstfSize 16

inline void dlc_push_gsnf(float8_128 x, bool select) {
  if (dlc_memorys.gsnf[dlc_get_device_id()][select].size() >= kGsnfGstfSize) {
    dlc_memorys.gsnf[dlc_get_device_id()][select].pop_back();
  }
  dlc_memorys.gsnf[dlc_get_device_id()][select].push_front(x);
}

inline void dlc_push_gstf(float8_128 x, bool select) {
  if (dlc_memorys.gstf[dlc_get_device_id()][select].size() >= kGsnfGstfSize) {
    dlc_memorys.gstf[dlc_get_device_id()][select].pop_back();
  }
  dlc_memorys.gstf[dlc_get_device_id()][select].push_front(x);
}

inline void dlc_update_gmr(const int& mode, const bool& select) {
  if (mode == 0) {
    // nothing
  } else if (mode == 1) {
    // load with gsnf
    for (int CASE = 0; CASE < int(dlc_memorys.gsnf[dlc_get_device_id()][select].size()); ++CASE) {
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 128; ++j) {
          dlc_memorys._gmr[dlc_get_device_id()][select][CASE  * 1024 + i * 128 + j] =
            dlc_memorys.gsnf[dlc_get_device_id()][select].at(CASE)[i * 128 + j];
        }
      }
    }
    for (int CASE = dlc_memorys.gsnf[dlc_get_device_id()][select].size(); CASE < 8; ++CASE) {
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 128; ++j) {
          dlc_memorys._gmr[dlc_get_device_id()][select][CASE  * 1024 + i * 128 + j] = 0;
        }
      }
    }
    while(dlc_memorys.gsnf[dlc_get_device_id()][select].size()) {
      dlc_memorys.gsnf[dlc_get_device_id()][select].pop_front();
    }
  } else if (mode == 2) {
    // load with gstf
    for (int CASE = 0; CASE < int(dlc_memorys.gstf[dlc_get_device_id()][select].size()); ++CASE) {
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 128; ++j) {
          dlc_memorys._gmr[dlc_get_device_id()][select][j * 128 + CASE * 8 + i] =
            dlc_memorys.gstf[dlc_get_device_id()][select].at(CASE)[i * 128 + j];
        }
      }
    }
    for (int CASE = dlc_memorys.gstf[dlc_get_device_id()][select].size(); CASE < 8; ++CASE) {
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 128; ++j) {
          dlc_memorys._gmr[dlc_get_device_id()][select][j * 128 + CASE * 8 + i] = 0;
        }
      }
    }
    while(dlc_memorys.gstf[dlc_get_device_id()][select].size()) {
      dlc_memorys.gstf[dlc_get_device_id()][select].pop_front();
    }
  } else {
    assert(false && "ERROR: dlc_update_gmr mode error");
  }
}
/* end */


/* int2float & float2int */
#define kExponentMask 0x7f800000
#define kBFloatSignificant 0x007f0000
#define kBFloatSignificantInc 0x00010000
#define kBFloatSignificantEven 0x00008000
#define kSignificantMask 0x007fffff
#define kExponentInc 0x00800000
#define kSignMask 0x80000000
#define kMostTwoByteMask 0xffff0000
#define kBFloatSignificantForQNaN 0x00410000
#define kBFloatSignificantEvenRest 0x00007fff
#define kTwoByteLength 16
#define kLeastTwoByteMask 0xffff
#define kBytePerWord 4
#define kLeastByteMask 0xff

#define kNumberOfBitsOfF32Significant 23
#define kNumberOfBitsOfF32Exponent 8
#define kExponentOffset 127
#define kBigExponentPos 31
#define kSmallExponentNega -33
#define kF32FractionExponent -1

inline void RoundTieToEven(dlc_dtype& data) {
  // see if the data is zero or small enough
  if ((data.u32 & kExponentMask) == 0) {
    data.u32 &= kSignMask;
  }
  // see if the data is inf or NaN
  else if ((data.u32 & kExponentMask) == kExponentMask) {
    // inf
    if ((data.u32 & kSignificantMask) == 0) {
      data.u32 &= kMostTwoByteMask;
    }
    // NaN
    else {
      data.u32 &= (~kSignificantMask);
      data.u32 |= kBFloatSignificantForQNaN;
    }
  }
  // normal
  else {
    // see if the significant need to be rounded
    if (data.u32 & kBFloatSignificantEven) {
      // see if the significant should be rounded tie to even or rounded up
      if ((data.u32 & kBFloatSignificantEvenRest) == 0) {
        // see if the significant need be rounded up to even
        if (data.u32 & kBFloatSignificantInc) {
          // see if the significant will overflow
          if ((data.u32 & kBFloatSignificant) == kBFloatSignificant) {
            data.u32 += kExponentInc;
            data.u32 &= (~kBFloatSignificant);
            data.u32 &= kMostTwoByteMask;
          } else {
            data.u32 += kBFloatSignificantInc;
            data.u32 &= kMostTwoByteMask;
          }
        } else {
          data.u32 &= kMostTwoByteMask;
        }
      } else {
        // see if the significant will be overflow
        if ((data.u32 & kBFloatSignificant) == kBFloatSignificant) {
          data.u32 += kExponentInc;
          data.u32 &= (~kBFloatSignificant);
          data.u32 &= kMostTwoByteMask;
        } else {
          data.u32 += kBFloatSignificantInc;
          data.u32 &= kMostTwoByteMask;
        }
      }
    }
    else {
      data.u32 &= kMostTwoByteMask;
    }
  }
}

inline dlc_dtype Float32ToFloat16(dlc_dtype data, SIM_X86::RoundFormat format) {
  dlc_dtype result;
  switch(format) {
    case SIM_X86::RoundFormat::ROUND:
      RoundTieToEven(data);
      break;
    case SIM_X86::RoundFormat::TRUNCATE:
      result.u32 = data.u32 & kMostTwoByteMask; 
      return result;
    case SIM_X86::RoundFormat::LOWER_ROUND:
      dlc_dtype immediate;
      immediate.u32 = data.u32 & kMostTwoByteMask;
      data.f32 = data.f32 - immediate.f32;
      RoundTieToEven(data);
      break;
    default:
      break;
  }
  result = data;
  return data;
}

float IntToFloat(int32_t data) {
  return *reinterpret_cast<float*>(&data);
}

uint32_t ConvertToFixedPointAndShift(int data) {
  unsigned int mantissa;
  int32_t exponent;
  mantissa = data & kSignificantMask;
  data = data >> kNumberOfBitsOfF32Significant;
  exponent = data & kLeastByteMask;
  mantissa = (mantissa << kNumberOfBitsOfF32Exponent) | kSignMask;
  exponent = exponent - kExponentOffset;

  if ((exponent >= kBigExponentPos) || (exponent <= kSmallExponentNega)) {
    return 0;
  }

  if(exponent == kF32FractionExponent) 
    return mantissa << (exponent + 1);
  if(exponent >= 0)
    return mantissa << (exponent + 1);
  else 
    return mantissa >> (-exponent - 1);
}
/* end */


/* permute */
// mode = 0: set permute
// mode = 1: set permute sublanes
// mode = 2: set permute bytes
inline void dlc_m_set_permute(const int8_128& x, const bool& select, const int& mode) {
  int8_128* pcr = &dlc_memorys._pcr[dlc_get_device_id()][select];
  if (mode == 0) {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 128; ++j) {
        int y = 0x7F & x[j];
        (*pcr)[i * 128 + j] = (y << 24) + (y << 16) + (y << 8) + y;
      }
    }
    // printf("pcr: %d\n", select);
    // for (int j = 0; j < 128; ++j) {
    //   for (int i = 0; i < 8; ++i) {
    //     printf("%x ", (*pcr)[i * 128 + j]);
    //   }
    //   printf("\n");
    // }
  } else if (mode == 1) {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 128; ++j) {
        int y = 0x7F & x[i * 128 + j];
        (*pcr)[i * 128 + j] = (y << 24) + (y << 16) + (y << 8) + y;
      }
    }
  } else if (mode == 2) {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 128; ++j) {
          (*pcr)[i * 128 + j] = x[i * 128 + j];
      }
    }
  }
}

inline void dlc_m_permute(float8_128 x, const int& select) {
  std::vector<dlc_dtype> y = getDlcDtype(x);
  float8_128 res;

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 128; ++j) {
      dlc_dtype val;
      int pcr = dlc_memorys._pcr[dlc_get_device_id()][select][i * 128 + j];
      val.u32 = (y[i * 128 + ((pcr >> 24) & 0x7F)].u32 & 0xFF000000) +
                (y[i * 128 + ((pcr >> 16) & 0x7F)].u32 & 0xFF0000) +
                (y[i * 128 + ((pcr >> 8) & 0x7F)].u32 & 0xFF00) +
                (y[i * 128 + (pcr & 0x7F)].u32 & 0xFF);
      res[i * 128 + j] = val.f32;
    }
  }

  dlc_m_push_trf(select, res);
}
/* end */


/* transpose */
inline void dlc_m_push_transpose_buffer(float8_128 x, const bool& select, const bool& packed) {
  std::deque<float8_128>* buffer = &dlc_memorys._transpose_buffer[dlc_get_device_id()][select];

  if (packed) {
    float8_128 low;

    for (int CASE = 0; CASE < 2; ++CASE) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 128; ++j) {
          dlc_dtype val;

          val.f32 = x[CASE * 4 * 128 + i * 128 + j];
          // val.u32 = ((val.u32 & 0xFFFF) << 16);
          val.u32 = (val.u32 & 0xFFFF);
          low[i * 2 * 128 + j] = val.f32;

          val.f32 = x[CASE * 4 * 128 + i * 128 + j];
          // val.u32 = (((val.u32 >> 16) & 0xFFFF) << 16);
          val.u32 = ((val.u32 >> 16) & 0xFFFF);
          low[i * 2 * 128 + 128 + j] = val.f32;
        }
      }

      // for (int i = 0; i < 1024; ++i) {
      //   printf("case = %d, i = %d\n buffer = %f\n", CASE, i, low[i]);
      // }

      if (buffer->size() < kTransposeBufferSize) {
        buffer->push_back(low);
      } else {
        assert(false && "dlc_m_push_transpose_buffer: transpose_buffer overfload");
      }
    }
  } else {
    if (buffer->size() < kTransposeBufferSize) {
      buffer->push_back(x);
    } else {
      assert(false && "dlc_m_push_transpose_buffer: transpose_buffer overfload");
    }
  }
}

inline void dlc_m_transpose_to_trf(const int& select) {
  int width = dlc_memorys._transpose_width[dlc_get_device_id()][select];

  std::deque<float8_128>* buffer = &dlc_memorys._transpose_buffer[dlc_get_device_id()][select];

  // printf("width = %d\n", width);
  // printf("buffer->size() = %d\n", buffer->size());

  for (int CASE = 0; CASE < width / 8; ++CASE) {
    float8_128 y;

    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < int(buffer->size()) * 8; ++j) {
        y[i * 128 + j] = (*buffer)[j / 8][(j & 0x7) * 128 + i + CASE * 8];
        // if (dlc_get_device_id())
        //   printf("buffer = %f\n", (*buffer)[j / 8][(j & 0x7) * 128 + i + CASE * 8]);
      }
    }

    dlc_m_push_trf(select, y);
    // dlc_m_push_trf(select, 0.f);
  }
  // printf("trf->size() = %d\n", dlc_memorys.trf[dlc_get_device_id()][select].size());

  // while(dlc_memorys._transpose_buffer[dlc_get_device_id()][select].size()) {
  //   dlc_memorys._transpose_buffer[dlc_get_device_id()][select].pop_front();
  // }
  while(buffer->size()) {
    buffer->pop_front();
  }
}
/* end */


/* matmul */
inline void dlc_m_matmul(float8_128 x, const SIM_X86::RoundFormat& mode, const bool& select) {
  float8_128 val;

  std::vector<dlc_dtype> y = getDlcDtype(x);

  for (int i = 0; i < 1024; ++i) {
    y[i] = Float32ToFloat16(y[i], mode);
  }

  float128_128* gmr = &dlc_memorys._gmr[dlc_get_device_id()][select];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 128; ++j) {
      float sum = 0;
      for (int k = 0; k < 128; ++k) {
        sum += y[i * 128 + k].f32 * (*gmr)[k * 128 + j];
      }
      val[i * 128 + j] = sum;
    }
  }

  dlc_m_push_mrf(select, val);
}

inline void dlc_m_matmul_int(float8_128 x, const bool& select) {
  float8_128 val;

  dlc_dtype sub_x, sub_y, res;

  float128_128* gmr = &dlc_memorys._gmr[dlc_get_device_id()][select];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 128; ++j) {
      int sum_high = 0, sum_low = 0;
      for (int k = 0; k < 128; ++k) {
        sub_x.f32 = x[i * 128 + k];
        sub_y.f32 = (*gmr)[k * 128 + j];

        int x_high = static_cast<int32_t>(static_cast<int8_t>((sub_x.u32 >> 16) & 0xFF));
        int x_low  = static_cast<int32_t>(static_cast<int8_t>((sub_x.u32 * 0xFF)));
        int y_high = static_cast<int32_t>(static_cast<int8_t>((sub_y.u32 >> 24) & 0xFF));
        int y_low  = static_cast<int32_t>(static_cast<int8_t>((sub_y.u32 >> 16) & 0xFF));

        #define kInt12Max 2047
        #define kInt12Min -2048

        int high_res = x_high * y_high;
        int low_res  = x_low  * y_low;
        if (high_res > kInt12Max) high_res = kInt12Max;
        if (high_res < kInt12Min) high_res = kInt12Min;
        if (low_res > kInt12Max) low_res = kInt12Max;
        if (low_res < kInt12Min) low_res = kInt12Min;

        sum_high += high_res;
        sum_low += low_res;
        if (sum_high > kInt12Max) sum_high = kInt12Max;
        if (sum_high < kInt12Min) sum_high = kInt12Min;
        if (sum_low > kInt12Max) sum_low = kInt12Max;
        if (sum_low < kInt12Min) sum_low = kInt12Min;
      }
      res.u32 = uint32_t((sum_high << kTwoByteLength) | (sum_low & kLeastTwoByteMask));
      val[i * 128 + j] = res.f32;
    }
  }

  dlc_m_push_mrf(select, val);
}
/* end */


/* vmem load store, with stride, ldst_mask and vmask*/
inline float8_128 dlc_v_f32_load_kernel(const SIM_X86::tensor& vmem, const int& stride,
                                        const int& ldst_mask, const bool8_128& vmask) {
  float8_128 x(0); // default 0

  std::bitset<8> bank(0);
  std::bitset<8> mask(ldst_mask);
  for (int i = 0; i < 8; ++i) {
    // if load this subcore
    if (mask.test(i)) {
      assert(i * stride * 128 + 128 <= vmem.data_size &&
             "ERROR: dlc_v_f32_load_kernel: src_addr out of range");
      assert(!bank.test((i * stride) % 8) && "ERROR: dlc_v_f32_load_kernel: vmem bank collision");
      for (int j = 0; j < 128; ++j) {
        // vmask
        if (vmask[i * 128 + j]) {
          x[i * 128 + j] = vmem.data_ptr[i * stride * 128 + j];
        }
      }
      // set bank flag
      bank.set((i * stride) % 8, true);
    }
  }

  #ifdef DEBUG_V_LOAD_XYS0
  if (dlc_get_device_id() == 0) {
    printf("[XYS%d]: V_LOAD: stride = %d, ldst_mask = %d\n", dlc_get_device_id(), stride, ldst_mask);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 128; ++j) {
        printf("%f ", x[i * 128 + j]);
      }
      puts("");
    }
  }
  #endif

  return x;
}

inline void dlc_v_f32_store_kernel(const SIM_X86::tensor& vmem, const int& stride,
                                    const int& ldst_mask, const bool8_128& vmask,
                                    float8_128 x) {
  std::bitset<8> bank(0);
  std::bitset<8> mask(ldst_mask);

  for (int i = 0; i < 8; ++i) {
    // if store this subcore
    if (mask.test(i)) {
      assert(!bank.test((i * stride) % 8) && "ERROR: dlc_v_f32_store_kernel: vmem bank collision");
      if (i * stride * 128 + 128 > vmem.data_size) {
        printf("stride = %d, ldst_mask = %d, data_size = %d, __LEN = %d\n", 
                stride, ldst_mask, vmem.data_size, vmem.__LEN);
      }
      assert(i * stride * 128 + 128 <= vmem.data_size &&
             "ERROR: dlc_v_f32_store_kernel: dst_addr out of range");
      for (int j = 0; j < 128; ++j) {
        // vmask
        if (vmask[i * 128 + j]) {
          vmem.data_ptr[i * stride * 128 + j] = x[i * 128 + j];
        }
      }
      // set bank flag
      bank.set((i * stride) % 8, true);
    }
  }

  #ifdef DEBUG_V_STORE_XYS0
  if (dlc_get_device_id() == 0) {
    printf("[XYS%d]: V_STORE: stride = %d, ldst_mask = %d\n", dlc_get_device_id(), stride, ldst_mask);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 128; ++j) {
        printf("%f ", x[i * 128 + j]);
      }
      puts("");
    }
  }
  #endif
}
/* end */


/* data type convert */
inline float8_128 dlc_$F(const int8_128& x) {
  float8_128 y;
  dlc_dtype val;

  for (int i = 0; i < 1024; ++i) {
    val.u32 = x[i];
    y[i] = val.f32;
  }

  return y;
}

inline int8_128 dlc_$S(float8_128 x) {
  int8_128 y;
  dlc_dtype val;

  for (int i = 0; i < 1024; ++i) {
    val.f32 = x[i];
    y[i] = int(val.u32);
  }

  return y;
}
/* end */


/* syn to x86 */
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

inline void Tensor2Vector(const syn::nn::Tensor& input, float* hbm) {
  if (input.dtype() == dlc_int16 || input.dtype() == dlc_bf16) {
    Tensor2Vector16(input, hbm);
  } else {
    Tensor2Vector32(input, hbm);
  }
}

inline void Vector2Tensor(float* hbm, syn::nn::Tensor& input) {
  if (input.dtype() == dlc_int16 || input.dtype() == dlc_bf16) {
    Vector2Tensor16(hbm, input);
  } else {
    Vector2Tensor32(hbm, input);
  }
}
/* end */

#endif
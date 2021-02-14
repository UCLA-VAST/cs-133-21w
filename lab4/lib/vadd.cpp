#include <chrono>
#include <iostream>
#include <random>

using std::clog;
using std::endl;

void VaddKernel(const float* a, const float* b, float* c, uint64_t n);

int main(int argc, const char* argv[]) {
  const uint64_t n = 1 << 20;
  static float a[n];
  static float b[n];
  static float c[n];
  static float c_base[n];

  std::default_random_engine generator(
      std::chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> distribution(0.f, 1.f);

  for (uint64_t i = 0; i < n; ++i) {
    a[i] = distribution(generator);
    b[i] = distribution(generator);
    c_base[i] = a[i] + b[i];
  }

  VaddKernel(a, b, c, n);

  for (uint64_t i = 0; i < n; ++i) {
    if (c[i] != c_base[i]) {
      clog << "FAIL" << endl;
      return 1;
    }
  }
  clog << "PASS" << endl;
  return 0;
}

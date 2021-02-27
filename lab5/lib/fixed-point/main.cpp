#include <chrono>
#include <iostream>
#include <string>

#include "cnn.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;
using std::string;

int main(int argc, char** argv) {
  // Allocate memory on heap to avoid stack overflow.
  static uint8_t input[kNum][kInImSize][kInImSize];
  static int8_t  weight[kNum][kNum][kKernel][kKernel];
  static uint8_t bias[kNum];
  static uint8_t output[kNum][kOutImSize][kOutImSize];

  if (argc > 2) {
    clog << "Usage: " << argv[0] << " [data dir]\n";
    return EXIT_FAILURE;
  }

  const string data_dir = argc == 2 ? string(argv[1]) + "/" : "";
  LoadData(data_dir, input, weight, bias);
  clog << "Invoke CNN computation kernel\n";

  CnnKernel(input, weight, bias, output);

  int error = Verify(data_dir, output);
  if (error != 0) {
    clog << "Found " << error << " error" << (error > 1 ? "s\n" : "\n");
    clog << "FAIL" << endl;
    return EXIT_FAILURE;
  } else {
    clog << "PASS" << endl;
    return EXIT_SUCCESS;
  }
}

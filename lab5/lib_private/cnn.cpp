#include <cmath>

#include <chrono>
#include <iostream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cnn.h"

using std::clog;
using std::endl;
using std::isfinite;
using std::max;
using std::string;

void LoadData(const string& data_dir, char input[kNum][kInImSize][kInImSize],
              char weight[kNum][kNum][kKernel][kKernel], char bias[kNum]) {
  const char kInputFile[] = "lib/testdata/input.bin";
  const char kWeightFile[] = "lib/testdata/weight.bin";
  const char kBiasFile[] = "lib/testdata/bias.bin";

  int input_fd = open((data_dir + kInputFile).c_str(), O_RDONLY);
  int weight_fd = open((data_dir + kWeightFile).c_str(), O_RDONLY);
  int bias_fd = open((data_dir + kBiasFile).c_str(), O_RDONLY);

  if (input_fd == -1) {
    clog << "Cannot find " << kInputFile << endl;
    exit(EXIT_FAILURE);
  }
  if (weight_fd == -1) {
    clog << "Cannot find " << kWeightFile << endl;
    exit(EXIT_FAILURE);
  }
  if (bias_fd == -1) {
    clog << "Cannot find " << kBiasFile << endl;
    exit(EXIT_FAILURE);
  }

  auto input_in = reinterpret_cast<float(*)[kInImSize][kInImSize]>(mmap(
      nullptr, sizeof(float) * kNum * kInImSize * kInImSize,
      PROT_READ, MAP_SHARED, input_fd, 0));
  if (input_in == MAP_FAILED) {
    clog << "Incomplete " << kInputFile << endl;
    close(input_fd);
    exit(EXIT_FAILURE);
  }

  auto weight_in = reinterpret_cast<float(*)[kNum][kKernel][kKernel]>(mmap(
      nullptr, sizeof(float) * kNum * kNum * kKernel * kKernel,
      PROT_READ, MAP_SHARED, weight_fd, 0));
  if (weight_in == MAP_FAILED) {
    clog << "Incomplete " << kWeightFile << endl;
    close(weight_fd);
    exit(EXIT_FAILURE);
  }

  float* bias_in = reinterpret_cast<float*>(mmap(
      nullptr, sizeof(float) * kNum, PROT_READ, MAP_SHARED, bias_fd, 0));
  if (bias_in == MAP_FAILED) {
    clog << "Incomplete " << kBiasFile << endl;
    close(bias_fd);
    exit(EXIT_FAILURE);
  }

  // normalized to 0 ~ 63
  for (int i = 0; i < kNum; i++)
    for (int h = 0; h < kInImSize; h++)
      for (int w = 0; w < kInImSize; w++) {
        input[i][h][w] = input_in[i][h][w] / 4;
      }

  // normalized to -63 ~ 63
  for (int i = 0; i < kNum; i++)
    for (int j = 0; j < kNum; j++)
      for (int p = 0; p < kKernel; p++)
        for (int q = 0; q < kKernel; q++) {
          weight[i][j][p][q] = weight_in[i][j][p][q] * 64;
        }

  // normalized to 0 ~ 63
  for (int i = 0; i < kNum; i++) {
    bias[i] = bias_in[i] * 64;
  }

  munmap(input_in, sizeof(*input) * kNum);
  munmap(weight_in, sizeof(*weight) * kNum);
  munmap(bias_in, sizeof(*bias) * kNum);
  close(input_fd);
  close(weight_fd);
  close(bias_fd);
}

bool IsError(char a, float b) {
  return fabs(a - b) > 4;
}

int Verify(const string& data_dir,
           const char output[kNum][kOutImSize][kOutImSize]) {
  int error = 0;
  const char kOutputFile[] = "lib/testdata/output.bin";
  int fd = open((data_dir + kOutputFile).c_str(), O_RDONLY);
  if (fd == -1) {
    clog << "Cannot find " << kOutputFile << endl;
    return EXIT_FAILURE;
  }
  auto ground_truth = reinterpret_cast<float(*)[kOutImSize][kOutImSize]>(mmap(
      nullptr, sizeof(float) * kNum * kOutImSize * kOutImSize,
      PROT_READ, MAP_SHARED, fd, 0));
  if (ground_truth == MAP_FAILED) {
    clog << "Incomplete " << kOutputFile << endl;
    close(fd);
    return EXIT_FAILURE;
  }
  bool first = true;
  int min = 0, max = 0;
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        if (IsError(output[i][h][w], ground_truth[i][h][w] / 512)) {
          if (ground_truth[i][h][w] / 512 < min) min = ground_truth[i][h][w] / 512;
          if (ground_truth[i][h][w] / 512 > max) max = ground_truth[i][h][w] / 512;
          if (first) {
            clog << "First error: get " << float(output[i][h][w]) << ", expecting "
                 << ground_truth[i][h][w] / 512 << " @ i = " << i << ", h = " << h
                 << ", w = " << w << endl;
            //first = false;
          }
          ++error;
        }
      }
    }
  }
  printf("%d %d\n", min, max);
  munmap(ground_truth, sizeof(*output) * kNum);
  close(fd);
  return error;
}

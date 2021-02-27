#include "lib/cnn-krnl.h"

void CnnKernel_YourCode(
    const input_t *input, const weight_t *weight,
    const bias_t  *bias,        output_t *output) {

  compute_t C[kImSize][kImSize];

  for (int i = 0; i < kNum; ++i) {
    // You can use printf in software simulation for debugging
    fprintf(stderr, "Finished %d%% channel(s) #%d/#%d\r", 100*i/kNum, i, kNum);

    // Set Bias
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w)
        C[h][w] = Bias(i);
    }

    // Convolution
    for (int j = 0; j < kNum; ++j) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q)
              C[h][w] += Weight(i, j, p, q) *
                         Input(j, h + p, w + q);
          }
        }
      }
    }

    // ReLU
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        if (C[h][w] < 0) C[h][w] = 0;
      }
    }

    // Max pooling
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        Output(i, h, w) = max(
            max(C[h * 2][w * 2    ], C[h * 2 + 1][w * 2    ]),
            max(C[h * 2][w * 2 + 1], C[h * 2 + 1][w * 2 + 1]));
      }
    }
  }

  fprintf(stderr, "\nDone!\n");
}

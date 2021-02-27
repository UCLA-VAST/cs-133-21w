#include <cstdio>
#include <ap_fixed.h>

const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

inline float max(float a, float b) { return a > b ? a : b; }
#define Input(x,y,z)    \
    (input[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)])
#define Weight(x,y,z,i) \
    (weight[(x)*kNum*kKernel*kKernel+(y)*kKernel*kKernel+(z)*kKernel+(i)])
#define Bias(x)         \
    (bias[(x)])
#define Output(x,y,z)   \
    (output[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+z])

void CnnKernel_YourCode(
    const ap_ufixed<8,8> *input, const ap_fixed <8, 1> *weight,
    const ap_ufixed<8,0> *bias,        ap_ufixed<8,15> *output) {

  ap_fixed<17,16> C[kImSize][kImSize];

  for (int i = 219; i < 220; ++i) {
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
        C[h][w] = max(0.f, C[h][w]);
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

/* Magics :-)  Do not touch unless you understand what you are doing */
extern "C" void CnnKernel(
    const ap_ufixed<8,8> *input, const ap_fixed <8, 1> *weight,
    const ap_ufixed<8,0> *bias,        ap_ufixed<8,15> *output) {
#pragma HLS interface m_axi port=input offset=slave bundle=gmem
#pragma HLS interface m_axi port=weight offset=slave bundle=gmem
#pragma HLS interface m_axi port=bias offset=slave bundle=gmem
#pragma HLS interface m_axi port=output offset=slave bundle=gmem
#pragma HLS interface s_axilite port=input bundle=control
#pragma HLS interface s_axilite port=weight bundle=control
#pragma HLS interface s_axilite port=bias bundle=control
#pragma HLS interface s_axilite port=output bundle=control
#pragma HLS interface s_axilite port=return bundle=control
    CnnKernel_YourCode(input, weight, bias, output);
}

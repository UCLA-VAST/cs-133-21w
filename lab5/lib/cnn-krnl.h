#ifndef CNN_KRNL_H_
#define CNN_KRNL_H_

// Used for faster software simulation
#pragma GCC target ("arch=skylake")
#pragma GCC optimize ("-O3,-ffast-math")

#include <cstdio>
#include <ap_fixed.h>
#include <ap_int.h>

#define kNum            (256)
#define kKernel         (5)
#define kImSize         (224)
#define kInImSize       (228)
#define kOutImSize      (112)

#ifndef kTileH
#define kTileH          (112)
#endif

#ifndef kTileW
#define kTileW          (112)
#endif

template <class T>
inline T max(T a, T b) { return a > b ? a : b; }

#ifdef FASTSIM

// These code are soly for accelerating software emulation
// and are not used for hardware generation.

#define Input(x,y,z)    \
    (input_g[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)])
#define Weight(x,y,z,i) \
    (weight_g[(x)*kNum*kKernel*kKernel+(y)*kKernel*kKernel+(z)*kKernel+(i)])
#define Bias(x)         \
    (bias_g[(x)])
#define Output(x,y,z)   \
    (output_g[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+z])

typedef float input_t;
typedef float weight_t;
typedef float bias_t;
typedef float output_t;
typedef float compute_t;

typedef float input_g_t;
typedef float weight_g_t;
typedef float bias_g_t;
typedef float output_g_t;

inline void read_weight_from_memory(
    const weight_g_t *weight_g,
    weight_t          weight[kNum][kNum][kKernel][kKernel]) {
  read_weight:
  for (int i = 0; i < kNum; i++)
    for (int j = 0; j < kNum; j++)
      for (int k = 0; k < kKernel; k++)
        for (int l = 0; l < kKernel; l++)
          weight[i][j][k][l] = Weight(i, j, k, l);
}

inline void read_bias_from_memory(
    const bias_g_t *bias_g,
    bias_t          bias[kNum]) {
  read_bias:
  for (int i = 0; i < kNum; i++)
    bias[i] = Bias(i);
}

inline void read_input_from_memory(int hh, int ww,
    const input_g_t *input_g,
    input_t          input[kNum][kTileH+4][kTileW+4]) {
  read_input:
  for (int j = 0; j < kNum; j++)
    for (int h = 0; h < kTileH + 4; h++)
      for (int w = 0; w < kTileW + 4; w++)
        input[j][h][w] = Input(j, hh + h, ww + w);
}

inline void write_output_to_memory(int hh, int ww,
    output_g_t      *output_g,
    output_t         output[kNum][kTileH/2][kTileW/2]) {
  write_output:
  for (int i = 0; i < kNum; i++)
    for (int h = 0; h < kTileH/2; h++)
      for (int w = 0; w < kTileW/2; w++)
        Output(i, hh/2 + h, ww/2 + w) = output[i][h][w];
}

#else

// These code are the actual code used for for hardware generation.

typedef ap_ufixed<8,8>   input_t;
typedef ap_fixed <8,1>   weight_t;
typedef ap_ufixed<8,0>   bias_t;
typedef ap_ufixed<8,15>  output_t;
typedef ap_fixed <17,16> compute_t;

typedef ap_uint<512>     input_g_t;
typedef ap_uint<512>     weight_g_t;
typedef ap_uint<512>     bias_g_t;
typedef ap_uint<512>     output_g_t;

const int item_per_read = 512 / 8;

void read_weight_from_memory(
    const weight_g_t *weight_g,
    weight_t          weight[kNum][kNum][kKernel][kKernel]) {

  const int total_read = kNum * kNum * kKernel * kKernel / item_per_read;

  read_weight:
  for (int idx = 0; idx < total_read; idx++) {
    weight_g_t data = weight_g[idx];

    for (int item = 0; item < item_per_read; item++) {
#pragma HLS pipeline
      int real_index = idx * item_per_read + item;
      int l = real_index % kKernel;  real_index /= kKernel;
      int k = real_index % kKernel;  real_index /= kKernel;
      int j = real_index % kNum;     real_index /= kNum;
      int i = real_index;

      ap_uint<8> temp = (data >> (item * 8))(7, 0);
      weight[i][j][k][l] = *((weight_t *)&temp);
    }
  }
}

void read_bias_from_memory(
    const bias_g_t *bias_g,
    bias_t          bias[kNum]) {

  const int total_read = kNum / item_per_read;

  read_bias:
  for (int idx = 0; idx < total_read; idx++) {
    bias_g_t data = bias_g[idx];

    for (int item = 0; item < item_per_read; item++) {
#pragma HLS pipeline
      int real_index = idx * item_per_read + item;
      ap_uint<8> temp = (data >> (item * 8))(7, 0);
      bias[real_index] = *((bias_g_t *)&temp);
    }
  }
}

void read_input_from_memory(int hh, int ww,
    const input_g_t *input_g,
    input_t          input[kNum][kTileH+4][kTileW+4]) {

  read_input:
  for (int j = 0; j < kNum; j++)
    for (int h = 0; h < kTileH + 4; h++) {

      const int start_real = j*kInImSize*kInImSize+(hh+h)*kInImSize + ww;
      const int end_real   = j*kInImSize*kInImSize+(hh+h)*kInImSize + ww+kTileW+4;
      const int start      = start_real / item_per_read;
      const int till       = (end_real - 1) / item_per_read;

      for (int idx = start; idx <= till; idx++) {
// (kTileW+4) / item_per_read <= 116/64 <= 2
#pragma HLS loop_tripcount min=2 max=2 avg=2
        input_g_t data = input_g[idx];

        for (int item = 0; item < item_per_read; item++) {
#pragma HLS pipeline
          int real_index = idx * item_per_read + item;
          if (real_index < end_real && real_index >= start_real) {
            int w = real_index % kInImSize;
            ap_uint<8> temp = (data >> (item * 8))(7, 0);
            input[j][h][w-ww] = *((input_t *)&temp);
          }
        }
      }
    }
}

void write_output_to_memory(int hh, int ww,
    output_g_t      *output_g,
    output_t         output[kNum][kTileH/2][kTileW/2]) {

  hh /= 2;  ww /= 2;

  write_output:
  for (int i = 0; i < kNum; i++)
    for (int h = 0; h < kTileH/2; h++) {

      const int start_real = i*kOutImSize*kOutImSize+(hh+h)*kOutImSize + ww;
      const int end_real   = i*kOutImSize*kOutImSize+(hh+h)*kOutImSize + ww+kTileW/2;
      const int start      = start_real / item_per_read;
      const int till       = (end_real - 1) / item_per_read;

      for (int idx = start; idx <= till; idx++) {
// (kTileW / 2) / item_per_read <= 56/64 <= 1
#pragma HLS loop_tripcount min=1 max=1 avg=1
        output_g_t data = output_g[idx];

        for (int item = 0; item < item_per_read; item++) {
#pragma HLS pipeline
          int real_index = idx * item_per_read + item;
          if (real_index < end_real && real_index >= start_real) {
            int w = real_index % kOutImSize;
            ap_uint<8> temp = *((ap_uint<8> *)&output[i][h][w-ww]);
            data(item*8+7, item*8) = temp;
          }
        }

        output_g[idx] = data;
      }
    }
}


#endif

void CnnKernel_YourCode(
    const input_g_t *input_g, const weight_g_t *weight_g,
    const bias_g_t  *bias_g,        output_g_t *output_g);

// This is the actual top level function, you need to specify the same
// interface pragmas if you are writing your own.  Besides, remember "C".

extern "C" void CnnKernel(
    const input_g_t *input, const weight_g_t *weight,
    const bias_g_t  *bias,        output_g_t *output) {
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

#endif

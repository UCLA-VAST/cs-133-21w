#ifndef CNN_KRNL_H_
#define CNN_KRNL_H_

#pragma GCC target ("arch=skylake")
#pragma GCC optimize ("-O3,-ffast-math")

#include <cstdio>
#include <ap_fixed.h>
#include <ap_int.h>

const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

#define H_TILE_SIZE     (112)
#define W_TILE_SIZE     (112)
#define H_TILE_COUNT    (2)
#define W_TILE_COUNT    (2)

template <class T>
inline T max(T a, T b) { return a > b ? a : b; }

#define Input(x,y,z)    \
    (input_g[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)])
#define Weight(x,y,z,i) \
    (weight_g[(x)*kNum*kKernel*kKernel+(y)*kKernel*kKernel+(z)*kKernel+(i)])
#define Bias(x)         \
    (bias_g[(x)])
#define Output(x,y,z)   \
    (output_g[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+z])

#ifdef FASTSIM
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
    bias_t          bias[kNum])
  read_bias:
  for (int i = 0; i < kNum; i++)
    bias[i] = Bias(i);
}

inline void read_input_from_memory(int hh, int ww,
    const input_g_t *input_g,
    input_t          input[kNum][H_TILE_SIZE+4][W_TILE_SIZE+4]) {
  read_input:
  for (int j = 0; j < kNum; ++j)
    for (int h = 0; h < H_TILE_SIZE + 4; h++)
      for (int w = 0; w < W_TILE_SIZE + 4; w++)
        input[j][h][w] = Input(j, hh + h, ww + w);
}

inline void write_output_to_memory(int hh, int ww,
    output_g_t      *output_g,
    output_t        output_t  output[kNum][H_TILE_SIZE/2][W_TILE_SIZE/2] {
  write_output:
  for (int i = 0; i < kNum; i++)
    for (int h = 0; h < H_TILE_SIZE/2; h++)
      for (int w = 0; w < W_TILE_SIZE/2; w++)
        Output(i, hh/2 + h, ww/2 + w) = output[i][h][w];
}

#else
typedef ap_ufixed<8,8>   input_t;
typedef ap_fixed <8,1>   weight_t;
typedef ap_ufixed<8,0>   bias_t;
typedef ap_ufixed<8,15>  output_t;
typedef ap_fixed <17,16> compute_t;

typedef ap_uint<512>     input_g_t;
typedef ap_uint<512>     weight_g_t;
typedef ap_uint<512>     bias_g_t;
typedef ap_uint<512>     output_g_t;

#endif

void CnnKernel_YourCode(
    const input_g_t *input_g, const weight_g_t *weight_g,
    const bias_g_t  *bias_g,        output_g_t *output_g);

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

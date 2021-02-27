#ifndef CNN_KRNL_H_
#define CNN_KRNL_H_

#pragma GCC target ("arch=skylake")
#pragma GCC optimize ("-O3,-ffast-math")

#include <cstdio>
#include <ap_fixed.h>

const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

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
#else
typedef ap_ufixed<8,8>   input_t;
typedef ap_fixed <8,1>   weight_t;
typedef ap_ufixed<8,0>   bias_t;
typedef ap_ufixed<8,15>  output_t;
typedef ap_fixed <17,16> compute_t;
#endif

void CnnKernel_YourCode(
    const input_t *input, const weight_t *weight,
    const bias_t  *bias,        output_t *output);

extern "C" void CnnKernel(
    const input_t *input, const weight_t *weight,
    const bias_t  *bias,        output_t *output) {
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

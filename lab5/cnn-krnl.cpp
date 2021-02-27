#include "lib/cnn-krnl.h"

void CnnKernel_YourCode(
    const input_g_t *input_g, const weight_g_t *weight_g,
    const bias_g_t  *bias_g,        output_g_t *output_g) {

  static input_t   input [kNum][H_TILE_SIZE+4][W_TILE_SIZE+4];
  static weight_t  weight[kNum][kNum][kKernel][kKernel];
  static bias_t    bias  [kNum];
  static output_t  output[kNum][H_TILE_SIZE/2][W_TILE_SIZE/2];

  static compute_t C[H_TILE_SIZE][W_TILE_SIZE];

  // read the whole arrays from memory to device
  read_weight_from_memory(weight_g, weight);
  read_bias_from_memory  (bias_g,   bias);

  main_loop_tile_h:
  for (int hh = 0; hh < kImSize; hh += H_TILE_SIZE) {

    main_loop_tile_w:
    for (int ww = 0; ww < kImSize; ww += W_TILE_SIZE) {

      // read input[j][h][w] = Input(j, hh + h, ww + w);
      read_input_from_memory(hh, ww, input_g, input);

      main_loop_i:
      for (int i = 0; i < kNum; ++i) {

        // You can use printf in software simulation for debugging
        fprintf(stderr, "Finished %d%% channel(s) #%d/#%d\r",
                100*i/kNum, i, kNum);

        // Set Bias
        set_bias:
        for (int h = 0; h < H_TILE_SIZE; ++h) {
          for (int w = 0; w < W_TILE_SIZE; ++w)
            C[h][w] = bias[i];
        }

        // Convolution
        conv:
        for (int j = 0; j < kNum; ++j) {
          for (int h = 0; h < H_TILE_SIZE; ++h) {
            for (int w = 0; w < W_TILE_SIZE; ++w) {
              for (int p = 0; p < kKernel; ++p) {
                for (int q = 0; q < kKernel; ++q)
                  C[h][w] += weight[i][j][p][q] *
                             input[j][h + p][w + q];
              }
            }
          }
        }

        // ReLU
        relu:
        for (int h = 0; h < H_TILE_SIZE; ++h) {
          for (int w = 0; w < W_TILE_SIZE; ++w) {
            if (C[h][w] < 0) C[h][w] = 0;
          }
        }

        // Max pooling
        maxpool:
        for (int h = 0; h < H_TILE_SIZE/2; ++h) {
          for (int w = 0; w < W_TILE_SIZE/2; ++w) {
            output[i][h][w] = max(
                max(C[h * 2][w * 2    ], C[h * 2 + 1][w * 2    ]),
                max(C[h * 2][w * 2 + 1], C[h * 2 + 1][w * 2 + 1]));
          }
        }
      }

      // write Output(i, hh/2 + h, ww/2 + w) = output[i][h][w];
      write_output_to_memory(hh, ww, output_g, output);

      fprintf(stderr, "\Computation for tile (%d, %d) is completed.\n",
              hh, ww) ;
    }
  }

}

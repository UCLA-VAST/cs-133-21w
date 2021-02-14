__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;


__kernel
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here
}

__kernel
void VaddKernel(__global const float* a, __global const float* b,
                __global float* c) {
  int i = get_global_id(0);

  c[i] = a[i] + b[i];
}

#!/bin/bash
if [ "$(uname -s)" == 'Darwin' ]; then
  xcode-select --install
else
  echo 'deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /' | \
      sudo tee /etc/apt/sources.list.d/cuda.list
  sudo apt-key adv --fetch-keys \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
  sudo apt update
  sudo env DEBIAN_FRONTEND=noninteractive apt install opencl-headers build-essential ocl-icd-libopencl1 ocl-icd-opencl-dev libnuma1 libpciaccess0 cuda -y --no-install-recommends
  sudo nvidia-smi -i 0 -ac 2505,1177
fi

#!/bin/bash
if [ "$(uname -s)" == 'Darwin' ]; then
  xcode-select --install
else
  sudo apt update
  sudo env DEBIAN_FRONTEND=noninteractive apt install aria2 opencl-headers build-essential cpio ocl-icd-libopencl1 libnuma1 libpciaccess0 -y
  tmp=$(mktemp -d)
  pushd "$tmp"
  aria2c -x16 -c https://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/17206/intel_sdk_for_opencl_applications_2020.3.494.tar.gz
  tar xf intel_sdk_for_opencl_applications_2020.3.494.tar.gz
  pushd intel_sdk_for_opencl_applications_2020.3.494
  sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/' silent.cfg
  sudo ./install.sh --silent silent.cfg
  popd; popd
  rm -rf "$tmp"
fi

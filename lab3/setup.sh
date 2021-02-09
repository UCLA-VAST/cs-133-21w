#!/bin/bash
if [ "$(uname -s)" == 'Darwin' ]; then
  xcode-select --install
else
  sudo apt update
  sudo env DEBIAN_FRONTEND=noninteractive apt install aria2 opencl-headers build-essential cpio -y
  tmp=$(mktemp -d)
  pushd "$tmp"
  aria2c -x16 -c http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/13793/l_opencl_p_18.1.0.013.tgz
  tar xf l_opencl_p_18.1.0.013.tgz
  pushd l_opencl_p_18.1.0.013
  sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/' silent.cfg
  sudo ./install.sh --silent silent.cfg
  popd; popd
  rm -rf "$tmp"
fi

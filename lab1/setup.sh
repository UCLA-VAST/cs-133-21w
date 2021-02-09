#!/bin/bash
if [ "$(uname -s)" == 'Darwin' ]; then
  xcode-select --install
  [ -f /usr/local/bin/brew ] || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  /usr/local/bin/brew install llvm zip
else
  sudo apt update
  sudo env DEBIAN_FRONTEND=noninteractive apt install build-essential zip -y
fi

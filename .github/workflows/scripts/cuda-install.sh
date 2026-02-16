#!/bin/bash

# Replace '.' with '-' ex: 11.8 -> 11-8
cuda_version=$(echo $1 | tr "." "-")
# Removes '-' and '.' ex: ubuntu-20.04 -> ubuntu2004
OS=$(echo $2 | tr -d ".\-")

sudo apt -qq update
sudo apt install -y wget gnupg2

# Installs CUDA
wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt -qq update
sudo apt --no-install-recommends -y install cuda-nvcc-${cuda_version} cuda-libraries-dev-${cuda_version}
sudo apt clean

# Test nvcc
PATH=/usr/local/cuda-$1/bin:${PATH}
nvcc --version

# Log gcc, g++, c++ versions
gcc --version
g++ --version
c++ --version

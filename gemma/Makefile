NVCC ?= /usr/local/cuda-12.8/bin/nvcc
ARCH ?= sm_100

kernel.so: kernel.cu ops.cuh
	$(NVCC) -O3 -maxrregcount=64 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC kernel.cu -o $@

NVCC ?= /usr/local/cuda-12.8/bin/nvcc
ARCH ?= sm_100a
TK := ../ThunderKittens

kernel.so: native.cu ops.cuh ../include/megakernel.cuh ../demos/low-latency-llama/matvec_pipeline.cuh
	$(NVCC) -O3 -std=c++20 -arch=$(ARCH) -DKITTENS_HOPPER -DKITTENS_BLACKWELL --expt-relaxed-constexpr --expt-extended-lambda -Xptxas=-dlcm=cg -I$(TK)/include -I../include -shared -Xcompiler -fPIC native.cu -lcuda -o $@

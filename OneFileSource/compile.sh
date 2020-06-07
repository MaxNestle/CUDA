#!/bin/sh

nvcc -o cuda_mult cuda_mult.cu -L/usr/local/cuda/lib -lcudart -lcufft -lcrypto
./cuda_mult

#!/bin/sh

nvcc -c -o cuda_FFT_mult.o cuda_FFT_mult.cu -I/usr/local/cuda/inc -L/usr/local/cuda/lib -lcufft
gcc -c -o cuda_mult.o cuda_mult.c -lcrypto

g++ -o cuda_mult cuda_mult.o cuda_FFT_mult.o -L/usr/local/cuda/lib -lcudart -lcufft -lcrypto
./cuda_mult

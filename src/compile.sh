#!/bin/sh

nvcc -c -g -o cuda_FFT_mult.o cuda_FFT_mult.cu -I/usr/local/cuda/inc -L/usr/local/cuda/lib -lcufft -lcrypto
gcc -c -g -o cuda_mult.o cuda_mult.c -lcrypto

g++ -o cuda_mult cuda_mult.o cuda_FFT_mult.o -L/usr/local/cuda/lib -lcudart -lcufft -lcrypto
./cuda_mult -g

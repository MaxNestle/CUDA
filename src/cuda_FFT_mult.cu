#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <string>
#include <iostream>
#include <openssl/bn.h>
#include <time.h>




/*The multicplication algorithm with cuFFT is from the following source:
Source: https://programmer.group/implementing-large-integer-multiplication-with-cufft.html
The multWithFFT-function is edited by Max & Johannes*/

const auto BATCH = 1;

__global__ void ComplexPointwiseMulAndScale(cufftComplex *a, cufftComplex *b, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    float scale = 1.0f / (float)size;
    cufftComplex c;
    for (int i = threadID; i < size; i += numThreads)
    {
        c = cuCmulf(a[i], b[i]);
        b[i] = make_cuFloatComplex(scale*cuCrealf(c), scale*cuCimagf(c));
    }
}

__global__ void ConvertToInt(cufftReal *a, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    auto b = (int*)a;
    for (int i = threadID; i < size; i += numThreads)
        b[i] = static_cast<int>(round(a[i]));
}

std::vector<int> multiply(const std::vector<float> &a, const std::vector<float> &b)
{

	//clock_t t;

	//t = clock();

    const auto NX = a.size();
    cufftHandle plan_a, plan_b, plan_c;
    cufftComplex *data_a, *data_b;
    std::vector<int> c(a.size() + 1);
    c[0] = 0;

    //Allocate graphics card memory and initialize, assuming sizeof(int)==sizeof(float), sizeof(cufftComplex)==2*sizeof(float)
    cudaMalloc((void**)&data_a, sizeof(cufftComplex) * (NX / 2 + 1) * BATCH);
    cudaMalloc((void**)&data_b, sizeof(cufftComplex) * (NX / 2 + 1) * BATCH);
    cudaMemcpy(data_a, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(data_b, b.data(), sizeof(float) * b.size(), cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "Cuda error: Failed to allocate\n"); return c; }

    if (cufftPlan1d(&plan_a, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFT error: Plan creation failed"); return c; }
    if (cufftPlan1d(&plan_b, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFT error: Plan creation failed"); return c; }
    if (cufftPlan1d(&plan_c, NX, CUFFT_C2R, BATCH) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFT error: Plan creation failed"); return c; }

	//t = clock() - t;
	//double time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
	//printf("Memory and plan: %f s\n", time_taken_GPU);


	//t = clock();

    //Converting A(x) to Frequency Domain
    if (cufftExecR2C(plan_a, (cufftReal*)data_a, data_a) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return c;
    }

    //Converting B(x) to Frequency Domain
    if (cufftExecR2C(plan_b, (cufftReal*)data_b, data_b) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
        return c;
    }

    //Point multiplication
    ComplexPointwiseMulAndScale<<<NX / 256 + 1, 256>>>(data_a, data_b, NX);

    //Converting C(x) back to time domain
    if (cufftExecC2R(plan_c, data_b, (cufftReal*)data_b) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecC2R Forward failed");
        return c;
    }

    //Converting the results of floating-point numbers to integers
    ConvertToInt<<<NX / 256 + 1, 256>>>((cufftReal*)data_b, NX);

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        return c;
    }

    //t = clock() - t;
    //time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
    //printf("Calc: %f s\n", time_taken_GPU);


    //t = clock();
    cudaMemcpy(&c[1], data_b, sizeof(float) * b.size(), cudaMemcpyDeviceToHost);

    cufftDestroy(plan_a);
    cufftDestroy(plan_b);
    cufftDestroy(plan_c);
    cudaFree(data_a);
    cudaFree(data_b);

    //t = clock() - t;
    //time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
    //printf("Cleaning: %f s\n", time_taken_GPU);

    return c;
}

void print(std::vector<float> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i);
	}
	printf("\n");
}

extern "C" int multWithFFT(char* a_String, char* b_String, char **c_String)
{

    const auto base = 10;


    //printf("a = %s\n",a_String);
    //printf("b = %s\n",b_String);


    //Set base

    int lengthA = strlen(a_String);
    int lengthB = strlen(b_String);

    int new_length = lengthA + lengthB;

    std::vector<float> a{};
    std::vector<float> b{};



    for(int i=0; i<lengthA; ++i){
    	a.push_back((float)(a_String[i])-'0');
    }

    for(int i=0; i<lengthB; ++i){
    	b.push_back((float)(b_String[i])-'0');
    }

    while (a.size() != new_length){
    	 a.insert(a.begin(),(float) 0);
    }

    while (b.size() != new_length){
    	b.insert(b.begin(),(float) 0);
    }

    a.resize(new_length,(float) 0);
    b.resize(new_length,(float) 0);

    //print(a);
    //printf("\n");

    //print(b);
    //printf("\n");

    std::vector<int> c = multiply(a, b);


    //Processing carry
    for (int i = c.size() - 1; i > 0; i--)
    {
        if (c[i] >= base)
        {
            c[i - 1] += c[i] / base;
            c[i] %= base;
        }
    }


    //Remove excess zeros
    c.pop_back();
    auto o = 0;
    if (c[0] == 0)
        o++;

    //To output the final result, we need to change the mode of output, such as the decimal system is "% 2d" and the decimal system is "% 3d".

    int i = o;

   /* printf("Old ");
    for (; i < c.size(); i++)
        printf("%d", c[i]);
    printf("\n");

*/
    char tmp[c.size()];
    int k = 0;
    i = o;
	//convert integer vector to string
    for (; i < c.size(); i++){
    	tmp[k] = (char) c.at(i) + '0';
    	k++;
    }
    tmp[k] = '\0';
	//printf("a: %s\n",tmp);

    //transfer result to cuda_mult.c
    memcpy(*c_String,tmp,strlen(tmp));

    return k;
}

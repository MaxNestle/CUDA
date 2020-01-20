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

	clock_t t;

	t = clock();

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

	t = clock() - t;
	double time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
	printf("Memory and plan: %f s\n", time_taken_GPU);


	t = clock();

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

    t = clock() - t;
    time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("Calc: %f s\n", time_taken_GPU);


    t = clock();
    cudaMemcpy(&c[1], data_b, sizeof(float) * b.size(), cudaMemcpyDeviceToHost);

    cufftDestroy(plan_a);
    cufftDestroy(plan_b);
    cufftDestroy(plan_c);
    cudaFree(data_a);
    cudaFree(data_b);

    t = clock() - t;
    time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("Cleaning: %f s\n", time_taken_GPU);

    return c;
}

void print(std::vector<float> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
}

extern "C" void multWithFFT(BIGNUM* a, BIGNUM *b, BIGNUM **c)
{

    const int base = 10;
    char* a_String = BN_bn2dec(a);
    char* b_String = BN_bn2dec(b);

    int lengthA = strlen(a_String);
    int lengthB = strlen(b_String);

	printf("a: %s\n",a_String);
	printf("b: %s\n",b_String);

    //length of multiplication result has the size of the sum of the two factors
    int result_length = lengthA + lengthB;

    //factors are stored in these vectors
    std::vector<float> av{};
    std::vector<float> bv{};

    //fill vectors step by step
    for(int i=0; i<lengthA; ++i){
    	av.push_back((float)(a_String[i])-'0');
    }
    for(int i=0; i<lengthB; ++i){
    	bv.push_back((float)(b_String[i])-'0');
    }

    //vectors need to be same size
    while (av.size() != result_length){
    	 av.insert(av.begin(),(float) 0);
    }
    while (bv.size() != result_length){
    	bv.insert(bv.begin(),(float) 0);
    }

	clock_t t;

	t = clock();
    //call cuda-kernel-function
    std::vector<int> cv = multiply(av, bv);
	t = clock() - t;
	double time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
	printf("Cuda_multiply: %f s\n", time_taken_GPU);

    //Processing carry
    for (int i = cv.size() - 1; i > 0; i--)
    {
        if (cv[i] >= base)
        {
            cv[i - 1] += cv[i] / base;
            cv[i] %= base;
        }
    }


    //Remove excess zeros
    cv.pop_back();
    auto i = 0;

    //For some multiplications the result has a zero as a first digit (for example 999*1 = 999 will be 0999)
    if (cv[0] == 0)
        i++;

    //If i++ will be executed the array still has to begin at element tmp[0]
    int k = 0;

	char tmp[cv.size()];

	//convert integer vector to string
    for (; i < cv.size(); i++){
    	tmp[k] = (char) cv.at(i) + '0';
    	k++;
    }
    tmp[k] = '\0';

    //transfer result to cuda_mult.c
    memcpy(*c,tmp,sizeof(tmp));

    return;
}

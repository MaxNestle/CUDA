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

void print(std::vector<float> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i);
	}
	printf("\n");
}


std::vector<int> multiply(const std::vector< std::vector<float>> at, std::vector< std::vector<float> > bt)
{

	clock_t t;
	t = clock();

	int amount = at.size();
	printf("Data Size: %d\n",amount);
	printf("at: %d\n",sizeof(at));
	printf("b: %d\n",sizeof(at[0]));


	int biggest_len = 0;
	for(int i=0; i<at.size(); ++i){
		if (biggest_len < at[i].size()){
			biggest_len = at[i].size();
		}
	}

    const auto NX = biggest_len;
    cufftHandle plan_a, plan_b, plan_c;
    cufftComplex *data_a[amount];
    cufftComplex *data_b[amount];
    std::vector<int> c(NX + 1);
    c[0] = 0;

    //Allocate graphics card memory and initialize, assuming sizeof(int)==sizeof(float), sizeof(cufftComplex)==2*sizeof(float)

	for(int i=0; i<amount; ++i){
		cudaMalloc((void**)&data_a[i],sizeof(cufftComplex) * (NX / 2 + 1) * BATCH);
		cudaMalloc((void**)&data_b[i],sizeof(cufftComplex) * (NX / 2 + 1) * BATCH);
	}

	for(int i=0; i<amount; ++i){
		cudaMemcpy(data_a[i], at[i].data(), sizeof(float) * at[i].size(), cudaMemcpyHostToDevice);
		cudaMemcpy(data_b[i], bt[i].data(), sizeof(float) * bt[i].size(), cudaMemcpyHostToDevice);
		if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "Cuda error: Failed to allocate\n"); return c; }
	}

    if (cufftPlan1d(&plan_a, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFT error: Plan creation failed"); return c; }
    if (cufftPlan1d(&plan_b, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFT error: Plan creation failed"); return c; }
    if (cufftPlan1d(&plan_c, NX, CUFFT_C2R, BATCH) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFT error: Plan creation failed"); return c; }

	t = clock() - t;
	double time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
	printf("Memory and plan: %f s\n", time_taken_GPU);


	for(int i=0; i<amount; ++i){
		t = clock();
		//Converting A(x) to Frequency Domain
		if (cufftExecR2C(plan_a, (cufftReal*)data_a[i], data_a[i]) != CUFFT_SUCCESS)
		{
			fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
			return c;
		}

		//Converting B(x) to Frequency Domain
		if (cufftExecR2C(plan_b, (cufftReal*)data_b[i], data_b[i]) != CUFFT_SUCCESS)
		{
			fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
			return c;
		}

		//Point multiplication
		ComplexPointwiseMulAndScale<<<NX / 256 + 1, 256>>>(data_a[i], data_b[i], NX);

		//Converting C(x) back to time domain
		if (cufftExecC2R(plan_c, data_b[i], (cufftReal*)data_b[i]) != CUFFT_SUCCESS)
		{
			fprintf(stderr, "CUFFT error: ExecC2R Forward failed");
			return c;
		}

		//Converting the results of floating-point numbers to integers
		ConvertToInt<<<NX / 256 + 1, 256>>>((cufftReal*)data_b[i], NX);

		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
			return c;
		}

		t = clock() - t;
		time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
		printf("Calc: %f s\n", time_taken_GPU);
	}

    t = clock();
    cudaMemcpy(&c[1], data_b[0], sizeof(float) * bt[0].size(), cudaMemcpyDeviceToHost);


    cufftDestroy(plan_a);
    cufftDestroy(plan_b);
    cufftDestroy(plan_c);

	for(int i=0; i<amount; ++i){
	    cudaFree(data_a[i]);
	    cudaFree(data_b[i]);
	}

    t = clock() - t;
    time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("Cleaning: %f s\n", time_taken_GPU);

    return c;
}


extern "C" int multWithFFT(char** data, char **c_String, int amount)
{

	int k = 0;
    const auto base = 10;

    //printf("a = %s\n",data[0]);
    //printf("b = %s\n",data[1]);

    int lengthA = strlen(data[0]);
    int lengthB = strlen(data[1]);

    int new_length = lengthA + lengthB;

	std::vector< std::vector<float> > a(amount/2);
	std::vector< std::vector<float> > b(amount/2);

	int t = 0;
    for(int l =0; l<amount/2; l++){

		//std::vector<float> a[sizeof(data)/2]{};
		//std::vector<float> b[sizeof(data)/2]{};


    	//TODO
		for(int i=0; i<lengthA; ++i){
			a[l].push_back((float)(data[t][i])-'0');
		}
		t++;
		for(int i=0; i<lengthB; ++i){
			b[l].push_back((float)(data[t][i])-'0');
		}
		t++;

		while (a[l].size() != new_length){
			a[l].insert(a[l].begin(),(float) 0);
		}

		while (b[l].size() != new_length){
			b[l].insert(b[l].begin(),(float) 0);
		}

		a[l].resize(new_length,(float) 0);
		b[l].resize(new_length,(float) 0);

		//print(a);
		//printf("\n");

		//print(b);
		//printf("\n");


    }

    int l = 0;
	std::vector< std::vector<int>> c(100);
	c[l] = multiply(a,b);


	//Processing carry
	for (int i = c[l].size() - 1; i > 0; i--)
	{
		if (c[l][i] >= base)
		{
			c[l][i - 1] += c[l][i] / base;
			c[l][i] %= base;
		}
	}


	//Remove excess zeros
	c[l].pop_back();
	auto o = 0;
	if (c[l][0] == 0)
		o++;

	//To output the final result, we need to change the mode of output, such as the decimal system is "% 2d" and the decimal system is "% 3d".

	int i = o;

   /* printf("Old ");
	for (; i < c.size(); i++)
		printf("%d", c[i]);
	printf("\n");

*/
	char tmp[c[l].size()];
	k = 0;
	i = o;
	//convert integer vector to string
	for (; i < c[l].size(); i++){
		tmp[k] = (char) c[l].at(i) + '0';
		k++;
	}
	tmp[k] = '\0';
	//printf("a: %s\n",tmp);
	memcpy(*c_String,tmp,strlen(tmp));

    //transfer result to cuda_mult.c

    return k;
}

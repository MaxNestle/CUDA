#include <stdio.h>
#include <stdlib.h>
#include <openssl/bn.h>
#include <string>
#include <time.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <curand_kernel.h>
#include <curand.h>


/*The multicplication algorithm with cuFFT is from the following source:
Source: https://programmer.group/implementing-large-integer-multiplication-with-cufft.html
The multWithFFT-function is edited by Max & Johannes*/

const auto BATCH = 1;
BIGNUM *Result_Multiplication_GPU = BN_new();
BIGNUM *Result_Multiplication_CPU = BN_new();
double cleanin_time_average = 0;
double calc_time_average = 0;
double prep_time_average = 0;
clock_t t_pre_cuda;
double time_test_average = 0;
double pre_cuda = 0;
clock_t t_post_cuda;
double post_cuda = 0;
clock_t time_test;

std::vector<int> c;


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
    clock_t t_tmp;
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
	  prep_time_average = prep_time_average+((double)t)/CLOCKS_PER_SEC; // in seconds
	  //printf("Memory and plan: %f s\n", time_taken_GPU);


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
    calc_time_average = calc_time_average + ((double)t)/CLOCKS_PER_SEC; // in seconds
    //printf("Calc: %f s\n", calc_time_average);


    t = clock();
    cudaMemcpy(&c[1], data_b, sizeof(float) * b.size(), cudaMemcpyDeviceToHost);

    cufftDestroy(plan_a);
    cufftDestroy(plan_b);
    cufftDestroy(plan_c);
    cudaFree(data_a);
    cudaFree(data_b);

    t = clock() - t;
    cleanin_time_average = cleanin_time_average + ((double)t)/CLOCKS_PER_SEC; // in seconds
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


void multWithFFT(BIGNUM* a_BN, BIGNUM* b_BN)
{
    t_pre_cuda = clock();
    const int base = 10;

    // BIGNUM to String
    char* a_String = "";
    char* b_String = "";
    
    a_String = BN_bn2dec(a_BN);
    b_String = BN_bn2dec(b_BN);

    //printf("a = %s\n",a_String);
    //printf("b = %s\n",b_String);



    // String to float Array
 
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
 
    
    time_test = clock();
    //while (a.size() != new_length){
    //	 a.insert(a.begin(),(float) 0);
    //}
    //while (b.size() != new_length){
    //	b.insert(b.begin(),(float) 0);
    //}
 
    int amount = new_length - a.size();
    a.insert(a.begin(),amount,(float) 0);
 
    amount = new_length - b.size();
    b.insert(b.begin(),amount,(float) 0);

 
    time_test = clock() - time_test;
    time_test_average = time_test_average + ((double)time_test)/CLOCKS_PER_SEC; // in seconds


    a.resize(new_length,(float) 0);
    b.resize(new_length,(float) 0);

    //print(a);
    //printf("\n");
    //print(b);
    //printf("\n");
 

    t_pre_cuda = clock() - t_pre_cuda;
    pre_cuda = pre_cuda + ((double)t_pre_cuda)/CLOCKS_PER_SEC; // in seconds


    // Multiply on GPU
    c = multiply(a, b);


    t_post_cuda = clock();

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
 
    char tmp[c.size()];
    int k = 0;
    i = o;
 
	  //convert integer vector to string

    for (; i < c.size(); i++){
    	tmp[k] = (char) c.at(i) + '0';
    	k++;
    }
    tmp[k] = '\0';
	  //printf("tmp: %s\n",tmp);

 
    // string to BIGNUM
    BN_dec2bn(&Result_Multiplication_GPU,tmp);
 
    t_post_cuda = clock() - t_post_cuda;
    post_cuda = post_cuda + ((double)t_post_cuda)/CLOCKS_PER_SEC; // in seconds

    return;
}


void multWithCPU(BIGNUM* a, BIGNUM* b, BN_CTX *bn_ctx){
	BN_mul(Result_Multiplication_CPU,a,b,bn_ctx);
}



void multWithGPU(BIGNUM* a, BIGNUM* b){
	multWithFFT(a, b);
}



int main (int argc, const char * argv[])
{
	int bits_per_number  = 24;
  int amount_of_multiplications = 100;

	/*Erzeugung der notwendigen BIGNUMs*/
	BIGNUM *Prime_1 = BN_new();
	BIGNUM *Prime_2 = BN_new();
	BN_CTX *bn_ctx = BN_CTX_new();
	double time_taken_CPU = 0;
	double time_taken_GPU = 0;
	float result_right = 0;
	float result_false = 0;
  cleanin_time_average = 0;
  calc_time_average = 0;
  prep_time_average = 0;
 time_test_average = 0;
 	clock_t time;
  int runs = 800;
  std::string csv = "Bits,Prep,Calc,Clean,Pre_Cuda, Post_CUDA,GPU,CPU,,,\n"; 

  for (int w = 0; w < runs; w++){
    time_taken_CPU = 0;
	  time_taken_GPU = 0;
	  result_right = 0;
	  result_false = 0;
    cleanin_time_average = 0;
    calc_time_average = 0;
    prep_time_average = 0;
    pre_cuda = 0;
    post_cuda = 0;
    time_test_average = 0;

    for (int l = 0; l < amount_of_multiplications; l++){

      BN_rand(Prime_1,bits_per_number,BN_RAND_TOP_ANY,BN_RAND_BOTTOM_ANY);
      BN_rand(Prime_2,bits_per_number,BN_RAND_TOP_ANY,BN_RAND_BOTTOM_ANY);

      //int res_length = strlen(BN_bn2dec(Prime_1))+strlen(BN_bn2dec(Prime_2));

      time = clock();
      multWithCPU(Prime_1,Prime_2,bn_ctx);
      time = clock() - time;
      time_taken_CPU = time_taken_CPU+ (((double)time)/CLOCKS_PER_SEC); // in seconds

      time = clock();
      multWithGPU(Prime_1,Prime_2);
      time = clock() - time;
      time_taken_GPU = time_taken_GPU+(((double)time)/CLOCKS_PER_SEC); // in seconds


      BN_sub(Prime_1, Result_Multiplication_GPU, Result_Multiplication_CPU);
      if (BN_is_zero(Prime_1) == 1){
        //printf("Result: %s\n",BN_bn2dec(Result_Multiplication_GPU));
        result_right++;
      }
      else {
        printf("\n\nResult GPU: %s\n",BN_bn2dec(Result_Multiplication_GPU));
        printf("Result CPU: %s\n",BN_bn2dec(Result_Multiplication_CPU));
        printf("\n");
        result_false++;
      }
    }
   
    printf("Bits: %d   average over %d runs\n", bits_per_number, amount_of_multiplications);
    printf("GPU preperation average: \t%f s\n", prep_time_average/amount_of_multiplications);
    printf("GPU calc average: \t\t%f s\n", calc_time_average/amount_of_multiplications);

    printf("Pre_Cuda average: \t\t%f s\n", pre_cuda/amount_of_multiplications);
    printf("Post_Cuda average: \t\t%f s\n", post_cuda/amount_of_multiplications);


    printf("GPU cleaning average: \t\t%f s\n", cleanin_time_average/amount_of_multiplications);
    printf("GPU total average: \t\t%f s\n", time_taken_GPU/amount_of_multiplications);
    printf("CPU total average: \t\t%f s\n", time_taken_CPU/amount_of_multiplications);

    printf("Test: \t\t\t\t%f s\n", time_test_average/amount_of_multiplications);

  
    float ratio = result_right/(result_right+result_false);
    printf("false: %f right: %f ratio: %f \n\n", result_false,result_right,ratio);

    csv = csv + std::to_string(bits_per_number)+",\""+std::to_string(prep_time_average/amount_of_multiplications)+"\",\""+std::to_string(calc_time_average/amount_of_multiplications)+"\",\""+
     std::to_string(cleanin_time_average/amount_of_multiplications)+"\",\""+std::to_string(pre_cuda/amount_of_multiplications)+"\",\""+std::to_string(post_cuda/amount_of_multiplications)+"\",\""+std::to_string(time_taken_GPU/amount_of_multiplications)+"\",\""+std::to_string(time_taken_CPU/amount_of_multiplications)+"\",,,\n";

    bits_per_number = bits_per_number + 24;
  }
  std::cout << csv ;

	BN_free(Prime_1);
	BN_free(Prime_2);
	BN_CTX_free(bn_ctx);
	BN_free(Result_Multiplication_GPU);
	BN_free(Result_Multiplication_CPU);


	return 0;
}


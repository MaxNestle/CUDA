#include <stdio.h>
#include <stdlib.h>
#include <openssl/bn.h>
#include <string.h>
#include <time.h>
#include "cuda_FFT_mult.h"

void multWithCPU(char** data, BN_CTX *bn_ctx ,BIGNUM** Result_Multiplication){
	BIGNUM *DiffCpuGpu = BN_new();
	BIGNUM *Prime_2 = BN_new();
	BN_dec2bn(&DiffCpuGpu,data[0]);
	BN_dec2bn(&Prime_2,data[1]);
	BN_mul(*Result_Multiplication,DiffCpuGpu,Prime_2,bn_ctx);
	BN_free(DiffCpuGpu);
	BN_free(Prime_2);
}

void multWithGPU(char** data, char** Result_Multiplication, int res_length, int amount){
	//The size of the required memory space is allocated
	char *result = (char*) malloc(res_length+1);

	int k = multWithFFT(data, &result, amount);

	result[k] = '\0';

	//printf("r: %s\n",result);

	BN_dec2bn(Result_Multiplication, result);
	free(result);
}

void getData(int n,int bits,BIGNUM** data){
    for (int i = 0; i > n; i++)
    {
    	data[n] = BN_new();
    	BN_rand(data[n],bits,BN_RAND_TOP_ANY,BN_RAND_BOTTOM_ANY);
    }
}

int main (int argc, const char * argv[])
{

	int amount_of_multiplications = 1;
	int amount_of_data = 4;
	int bits  = 4;

	BIGNUM *DiffCpuGpu = BN_new();
	BN_CTX *bn_ctx = BN_CTX_new();
	BIGNUM *Result_Multiplication_GPU = BN_new();
	BIGNUM *Result_Multiplication_CPU = BN_new();

	float right = 0;
	float false = 0;

	for (int l = 0; l < amount_of_multiplications; l++){

		BN_rand(DiffCpuGpu,bits,BN_RAND_TOP_ANY,BN_RAND_BOTTOM_ANY);

		char* data[amount_of_data];
		BIGNUM *tmp = BN_new();

		for (int i = 0; i < amount_of_data; i++)
		{
			BN_rand(tmp,bits,BN_RAND_TOP_ANY,BN_RAND_BOTTOM_ANY);
			data[i] = BN_bn2dec(tmp);
			printf("%s\n",data[i]);
		}
		BN_free(tmp);

		//clock_t t;
		//t = clock();
		multWithCPU(data,bn_ctx,&Result_Multiplication_CPU);
		//t = clock() - t;
		//double time_taken_CPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
		//printf("CPU: %f s\n", time_taken_CPU);

		//t = clock();
		//result_length zuweisen
		multWithGPU(data,&Result_Multiplication_GPU,10, amount_of_data);
		//t = clock() - t;
		//double time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds
		//printf("GPU: %f s\n", time_taken_GPU);


		BN_sub(DiffCpuGpu, Result_Multiplication_GPU, Result_Multiplication_CPU);
		if (BN_is_zero(DiffCpuGpu) == 1){
			printf("Result: %s\n",BN_bn2dec(Result_Multiplication_GPU));
			right++;
		}
		else {
			printf("\n\nResult GPU: %s\n",BN_bn2dec(Result_Multiplication_GPU));
			printf("Result CPU: %s\n",BN_bn2dec(Result_Multiplication_CPU));
			printf("\n");
			false++;
		}
	}

	float ratio = right/(right+false);
	printf("false: %f right: %f ratio: %f \n", false,right,ratio);


	BN_free(DiffCpuGpu);
	BN_CTX_free(bn_ctx);
	BN_free(Result_Multiplication_GPU);
	BN_free(Result_Multiplication_CPU);

	return 0;
}

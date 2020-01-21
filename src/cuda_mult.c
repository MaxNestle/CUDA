#include <stdio.h>
#include <stdlib.h>
#include <openssl/bn.h>
#include <string.h>
#include <time.h>
#include "cuda_FFT_mult.h"

void multWithCPU(BIGNUM* a, BIGNUM* b, BN_CTX *bn_ctx ,BIGNUM** Result_Multiplication){
	BN_mul(*Result_Multiplication,a,b,bn_ctx);
}

void multWithGPU(char* a, char* b, char** Result_Multiplication, int res_length){
	//The size of the required memory space is allocated
	//int len = sizeof(char)*( sizeof(Prime_1_Value)/sizeof(char) + sizeof(Prime_2_Value)/sizeof(char) +1);
	//int len = BN_num_bits(a)/8 + BN_num_bits(b)/8;
	char *result = (char*) malloc(res_length+1);
	//printf("length: %d\n", res_length);

	int k = multWithFFT(a, b, &result);

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

	char Prime_1_Value[] = "108165236279178312660610114131826512483935470542850824183737259708197206310322";
	char Prime_2_Value[] = "566862902526025592026168860088627596750028651678658181187001187611221298622578";

	int n = 100;
	int bits  = 25600;


	/*Erzeugung der notwendigen BIGNUMs*/
	BIGNUM *Prime_1 = BN_new();
	BIGNUM *Prime_2 = BN_new();
	BN_CTX *bn_ctx = BN_CTX_new();
	BIGNUM *Result_Multiplication_GPU = BN_new();
	BIGNUM *Result_Multiplication_CPU = BN_new();

	float r = 0;
	float f = 0;

	for (int l = 0; l < 1000; l++){

	//BN_dec2bn(&Prime_1, Prime_1_Value);
	//BN_dec2bn(&Prime_2, Prime_2_Value);

	BN_rand(Prime_1,bits,BN_RAND_TOP_ANY,BN_RAND_BOTTOM_ANY);
	BN_rand(Prime_2,bits,BN_RAND_TOP_ANY,BN_RAND_BOTTOM_ANY);

	int res_length = strlen(BN_bn2dec(Prime_1))+strlen(BN_bn2dec(Prime_2));

	int la = strlen(BN_bn2dec(Prime_1));
	int lb = strlen(BN_bn2dec(Prime_2));

	//BIGNUM* data[1000];
	//getData(n,bits,data);

	//clock_t t;

	//t = clock();
	multWithCPU(Prime_1,Prime_2,bn_ctx,&Result_Multiplication_CPU);
	//t = clock() - t;
	//double time_taken_CPU = ((double)t)/CLOCKS_PER_SEC; // in seconds


	//t = clock();
	multWithGPU(BN_bn2dec(Prime_1),BN_bn2dec(Prime_2),&Result_Multiplication_GPU,res_length);
	//t = clock() - t;
	//double time_taken_GPU = ((double)t)/CLOCKS_PER_SEC; // in seconds


	//printf("GPU: %f s\n", time_taken_GPU);
	//printf("CPU: %f s\n", time_taken_CPU);




	BN_sub(Prime_1, Result_Multiplication_GPU, Result_Multiplication_CPU);
	if (BN_is_zero(Prime_1) == 1){
		//printf("k");
		printf("Result: %s\n",BN_bn2dec(Result_Multiplication_GPU));

		r++;
	}
	else {
		//printf("la: %d lb: %d\n",la,lb);

		printf("\n\nResult GPU: %s\n",BN_bn2dec(Result_Multiplication_GPU));
		printf("Result CPU: %s\n",BN_bn2dec(Result_Multiplication_CPU));
		printf("\n");
		f++;
	}




	}

	float ratio = r/(r+f);
	printf("false: %f right: %f ratio: %f \n", f,r,ratio);


	BN_free(Prime_1);
	BN_free(Prime_2);
	BN_CTX_free(bn_ctx);
	BN_free(Result_Multiplication_GPU);
	BN_free(Result_Multiplication_CPU);

	return 0;
}

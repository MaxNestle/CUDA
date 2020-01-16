#include <stdio.h>
#include <stdlib.h>
#include <openssl/bn.h>



int main (int argc, const char * argv[])
{

	char Prime_1_Value[] = "18218455622870713193";
	char Prime_2_Value[] = "16785410813118998279";

	/*Erzeugung der notwendigen BIGNUMs*/
	BIGNUM *Prime_1 = BN_new();
	BIGNUM *Prime_2 = BN_new();
	BIGNUM *Result_Multiplication = BN_new();

	//BN_dec2bn(&Prime_1, Prime_1_Value);
	//BN_dec2bn(&Prime_2, Prime_2_Value);

	//The size of the required memory space is allocated
	int len = sizeof(char)*( sizeof(Prime_1_Value)/sizeof(char) + sizeof(Prime_2_Value)/sizeof(char) +1);
	char *result = (char*) malloc(len);

	multWithFFT(Prime_1_Value, Prime_2_Value, &result);

	printf("Result: %s\n",result);

	BN_dec2bn(&Result_Multiplication, result);

	BN_free(Prime_1);
	BN_free(Prime_2);
	BN_free(Result_Multiplication);

	free(result);
	return 0;
}

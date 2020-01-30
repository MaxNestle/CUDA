################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Aykan/cuda_FFT_mult.cu 

C_SRCS += \
../src/Aykan/cuda_mult.c 

OBJS += \
./src/Aykan/cuda_FFT_mult.o \
./src/Aykan/cuda_mult.o 

CU_DEPS += \
./src/Aykan/cuda_FFT_mult.d 

C_DEPS += \
./src/Aykan/cuda_mult.d 


# Each subdirectory must supply rules for building sources it contributes
src/Aykan/%.o: ../src/Aykan/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "src/Aykan" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/Aykan/%.o: ../src/Aykan/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "src/Aykan" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



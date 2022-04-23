#include "MinMaxStretching.cuh"
#include <array>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void stretchKernel(unsigned char* output, unsigned char* input, unsigned char* values, int count)
{
    unsigned int a = blockDim.x * blockIdx.x + threadIdx.x;
    output[a] = values[(int)input[a]];
}

__global__ void minMaxKernel(unsigned char* input, int count, unsigned char* _minMax)
{
    unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;

    if (_minMax[0] > input[a])
    {
        _minMax[0] = input[a];
    }
    else if (_minMax[1] < input[a])
    {
        _minMax[1] = input[a];
    }
}

__global__ void calculateDifferenceKernel(unsigned char* _minMax, float* difference)
{
    *difference = 255.0f / (_minMax[1] - _minMax[0]);
}

__global__ void stretchValuesKernel(unsigned char* values, unsigned char* _min, float* difference)
{
    unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
    values[a] = (a - _min[0]) * (*difference);
}

MinMaxStretching::MinMaxStretching(unsigned char* values, int count, bool fromGpuValues)
{
    _count = count;
    _threadsPerBlock = 1024;
    _blockCount = _count / _threadsPerBlock;

    if (fromGpuValues)
    {
        _values = values;
    }
    else
    {
        cudaMalloc(&_values, _count * sizeof(unsigned char));
        cudaMemcpy(_values, values, count * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }
}

unsigned char* MinMaxStretching::stretch()
{
    unsigned char* gpuOutput = stretchGpu();
    unsigned char* output = new unsigned char[_count];
    //Move output from GPU to RAM
    cudaMemcpy(output, gpuOutput, _count * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(gpuOutput);

    return output;
}

unsigned char* MinMaxStretching::stretchGpu()
{
    float* gpuDifference;
    unsigned char* gpuDarkestBrightest;
    unsigned char* gpuStretchValues;
    unsigned char* gpuOutput;

    //Allocating & filling
    cudaMalloc(&gpuDarkestBrightest, 2 * sizeof(unsigned char));
    cudaMemcpy(gpuDarkestBrightest, _values, 2 * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
    cudaMalloc(&gpuDifference, sizeof(float));
    cudaMalloc(&gpuOutput, _count * sizeof(unsigned char));

    //GPU calculations
    minMaxKernel << <_blockCount, _threadsPerBlock >> > (_values, _count, gpuDarkestBrightest);
    calculateDifferenceKernel << <1, 1 >> > (gpuDarkestBrightest, gpuDifference);
    cudaMalloc(&gpuStretchValues, 256 * sizeof(unsigned char));
    stretchValuesKernel << <1, _threadsPerBlock >> > (gpuStretchValues, gpuDarkestBrightest, gpuDifference);
    stretchKernel << <_blockCount, _threadsPerBlock >> > (gpuOutput, _values, gpuStretchValues, _count);

    //Free GPU memory
    cudaFree(gpuDifference);
    cudaFree(gpuDarkestBrightest);
    cudaFree(gpuStretchValues);

    return gpuOutput;
}
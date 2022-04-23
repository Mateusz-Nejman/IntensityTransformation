
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <array>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "OldAlgo.h"
#include "MinMaxStretching.cuh"

int main()
{
    //Loading data
    auto mat = cv::imread("image.jpg");
    auto gray = mat.clone();
    int count = gray.rows * gray.cols;
    cv::cvtColor(mat, gray, cv::COLOR_RGB2GRAY);
    cv::Mat cpuMat = mat.clone();
    cv::Mat gpuMat = mat.clone();

    unsigned char* image = new unsigned char[count];

    int index = 0;
    for (int y = 0; y < gray.rows; y++)
    {
        for (int x = 0; x < gray.cols; x++)
        {
            image[index] = gray.at<uchar>(y, x);
            index++;
        }
    }

    //CPU

    OldAlgo algo(image, count);
    auto algoStart = std::chrono::high_resolution_clock::now();
    auto outputImage = algo.stretch();
    auto algoEnd = std::chrono::high_resolution_clock::now();

    std::cout << "OldAlgo: " << std::chrono::duration_cast<std::chrono::microseconds>(algoEnd - algoStart).count() << " microseconds" << std::endl;

    index = 0;
    for (int y = 0; y < gray.rows; y++)
    {
        for (int x = 0; x < gray.cols; x++)
        {
            cpuMat.at<uchar>(y, x) = outputImage[index];
            index++;
        }
    }

    cv::imwrite("outputImage.jpg", gray);
    

    //CUDA


    MinMaxStretching gpuAlgo(image, count);
    algoStart = std::chrono::high_resolution_clock::now();
    auto gpuOutputImage = gpuAlgo.stretchGpu();
    algoEnd = std::chrono::high_resolution_clock::now();
    std::cout << "NewAlgo: " << std::chrono::duration_cast<std::chrono::microseconds>(algoEnd - algoStart).count() << " microseconds" << std::endl;

    unsigned char* gpuOutputImage1 = new unsigned char[count];

    //Move output from GPU to RAM
    cudaMemcpy(gpuOutputImage1, gpuOutputImage, count * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(gpuOutputImage);
    cudaDeviceReset();

    index = 0;
    for (int y = 0; y < gray.rows; y++)
    {
        for (int x = 0; x < gray.cols; x++)
        {
            gpuMat.at<uchar>(y, x) = gpuOutputImage1[index];
            index++;
        }
    }

    cv::imwrite("outputCuda.jpg", gray);

    return 0;
}
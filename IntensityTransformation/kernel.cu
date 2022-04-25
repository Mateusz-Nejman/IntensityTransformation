
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <array>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "OldAlgo.h"
#include "MinMaxStretching.cuh"

int outputPerformanceTest()
{
  //Loading data
  auto mat = cv::imread("CorrectedImage.bmp");
  auto gray = mat.clone();
  int count = gray.rows * gray.cols;
  cv::cvtColor(mat, gray, cv::COLOR_RGB2GRAY);
  cv::Mat cpuMat = gray.clone();
  cv::Mat gpuMat = gray.clone();

  unsigned char* image = gray.data;

  //CPU

  OldAlgo algo(image, count);
  auto algoStart = std::chrono::high_resolution_clock::now();
  auto outputImage = algo.stretch();
  auto algoEnd = std::chrono::high_resolution_clock::now();

  std::cout << "OldAlgo: " << std::chrono::duration_cast<std::chrono::microseconds>(algoEnd - algoStart).count() << " microseconds" << std::endl;

  cpuMat.data = outputImage;

  cv::imwrite("outputImage.jpg", cpuMat);


  //CUDA


  MinMaxStretching gpuAlgo(image, count);
  algoStart = std::chrono::high_resolution_clock::now();
  auto gpuOutputImage = gpuAlgo.stretchGpu();
  algoEnd = std::chrono::high_resolution_clock::now();
  std::cout << "NewAlgo: " << std::chrono::duration_cast<std::chrono::microseconds>(algoEnd - algoStart).count() << " microseconds" << std::endl;

  //Move output from GPU to RAM
  gpuMat.data = gpuOutputImage.getData();
  cudaDeviceReset();

  cv::imwrite("outputCuda.jpg", gpuMat);

  return 0;
}

int outputGpuCyclePerformanceTest()
{
  auto algoStart = std::chrono::high_resolution_clock::now();
  //Loading data
  auto gpuMat = cv::imread("CorrectedImage.bmp", cv::IMREAD_GRAYSCALE);
  int count = gpuMat.rows * gpuMat.cols;

  MinMaxStretching* gpuAlgo = new MinMaxStretching(gpuMat.data, count);
  auto gpuOutputImage = gpuAlgo->stretchGpu();

  //Move output from GPU to RAM
  gpuMat.data = gpuOutputImage.getData();
  cv::imwrite("outputCuda.jpg", gpuMat);
  delete gpuAlgo;
  auto algoEnd = std::chrono::high_resolution_clock::now();
  std::cout << "NewAlgo: " << std::chrono::duration_cast<std::chrono::microseconds>(algoEnd - algoStart).count() << " microseconds" << std::endl;
  cudaDeviceReset();
  return 0;
}

int main()
{
    return outputPerformanceTest();
}
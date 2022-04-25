#pragma once
class MinMaxStretching
{
public:
	MinMaxStretching(unsigned char* values, int count, bool fromGpuValues = false);
	~MinMaxStretching();
	unsigned char* stretch();
	MinMaxStretching stretchGpu();
	unsigned char* getData();
	void minMaxToCheck();

private:
	unsigned char* _values;
	int _count;
	int _threadsPerBlock;
	int _blockCount;
	float *gpuDifference;
	unsigned char* gpuDarkestBrightest;
	unsigned char* gpuStretchValues;
	unsigned char* gpuOutput;
};


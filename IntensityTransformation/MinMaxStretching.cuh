#pragma once
class MinMaxStretching
{
public:
	MinMaxStretching(unsigned char* values, int count, bool fromGpuValues = false);
	unsigned char* stretch();
	unsigned char* stretchGpu();

private:
	unsigned char* _values;
	int _count;
	int _threadsPerBlock;
	int _blockCount;
};


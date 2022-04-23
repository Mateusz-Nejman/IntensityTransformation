#pragma once
class OldAlgo
{
public:
	OldAlgo(unsigned char* values, int count);
	unsigned char* stretch();

private:
	unsigned char* _values;
	int _count;
};


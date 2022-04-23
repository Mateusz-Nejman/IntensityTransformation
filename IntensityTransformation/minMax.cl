void kernel stretchKernel(global unsigned char* output, unsigned char* input, unsigned char* values, int count)
{
    unsigned int a = get_global_id(0);
    output[a] = values[(int)input[a]];
}

void kernel minMaxKernel(unsigned char* input, int count, unsigned char* _minMax)
{
    unsigned int a = get_global_id(0);

    if (_minMax[0] > input[a])
    {
        _minMax[0] = input[a];
    }
    else if (_minMax[1] < input[a])
    {
        _minMax[1] = input[a];
    }
}

void kernel calculateDifferenceKernel(unsigned char* _minMax, float* difference)
{
    *difference = 255.0f / (_minMax[1] - _minMax[0]);
}

void kernel stretchValuesKernel(unsigned char* values, unsigned char* _min, float* difference)
{
    unsigned int a = get_global_id(0);
    values[a] = (a - _min[0]) * (*difference);
}
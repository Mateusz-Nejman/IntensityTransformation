#include "OldAlgo.h"
#include <array>
#include <iostream>

OldAlgo::OldAlgo(unsigned char* values, int count)
{
	_values = values;
	_count = count;
}

unsigned char* OldAlgo::stretch()
{
    unsigned char* output = new unsigned char[_count];
    unsigned char darkest = _values[0];
    unsigned char brightest = _values[0];

    for (int a = 0; a < _count; a++)
    {
        if (darkest > _values[a])
        {
            darkest = _values[a];
        }
        else if (brightest < _values[a])
        {
            brightest = _values[a];
        }
    }

    float difference = 255.0f / (brightest - darkest);

    std::array<unsigned char, 256> histo;

    for (int a = 0; a < 256; a++)
    {
        histo[a] = (a - darkest) * difference;
    }

    for (int a = 0; a < _count; a++)
    {
        output[a] = histo[(int)_values[a]];
    }

    return output;
}
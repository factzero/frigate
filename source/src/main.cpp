#include "examples/test_squeezenet.h"


int main()
{
    {
        test_squeezenet();
        test_squeezenet_int8();
    }

    return 0;
}

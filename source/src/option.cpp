#include "option.h"


namespace ACNN
{
    Option::Option()
    {
        use_sgemm_convolution = false;
        use_int8_inference = false;
    }
}
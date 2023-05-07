#pragma once



namespace ACNN
{
    class Option
    {
    public:
        Option();
    
    public:
        int use_sgemm_convolution;
        int use_int8_inference;
    };
}
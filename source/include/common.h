#pragma once
#include "amat.h"
#include "option.h"


namespace ACNN
{
    void copy_make_border(const aMat& src, aMat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt);
}
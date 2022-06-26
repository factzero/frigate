#pragma once
#include "layer.h"


namespace ACNN
{
	class Dropout : public Layer
	{
	public:
		Dropout(const LayerParam& layer_param);

		virtual int load_param(const ParamDict& pd) override;
		virtual int forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const override;

	private:
		float scale;
	};
}
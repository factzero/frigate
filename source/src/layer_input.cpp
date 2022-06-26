#include "layer_input.h"
#include "layer_factory.h"


namespace ACNN
{
	Input::Input(const LayerParam& layer_param)
		: Layer(layer_param)
	{}

	int Input::load_param(const ParamDict& pd)
	{
		w = pd.get(0, 0);
		h = pd.get(1, 0);
		d = pd.get(11, 0);
		c = pd.get(2, 0);

		return 0;
	}

	int Input::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
	{
		return 0;
	}

	REGISTER_LAYER_CLASS(Input, Input);
}
#include "layer_relu.h"
#include "layer_factory.h"
#include "platform.h"


namespace ACNN
{
	Relu::Relu(const LayerParam& layer_param)
		: Layer(layer_param)
	{}

	int Relu::load_param(const ParamDict& pd)
	{
		slope = pd.get(0, 0.f);

		return 0;
	}

	int Relu::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
	{
		if (bottom_blobs.size() != top_blobs.size())
		{
			ACNN_LOGE("Relu ERROR: bottom_blobs size(%d) != top_blobs size(%d)", (int)bottom_blobs.size(), (int)top_blobs.size());
			return -1;
		}

		for (size_t i = 0; i < bottom_blobs.size(); i++)
		{
			const aMat& bottom_blob = bottom_blobs[i];
			aMat& top_blob = top_blobs[i];
			int w = bottom_blob.m_w;
			int h = bottom_blob.m_h;
			int channels = bottom_blob.m_c;

			top_blob.create(w, h, channels, bottom_blob.m_elemsize, bottom_blob.m_allocator);
			for (int q = 0; q < channels; q++)
			{
				const float* inptr = bottom_blob.channel(q);
				float* outptr = top_blob.channel(q);
				for (int j = 0; j < w * h; j++)
				{
					if (inptr[j] < 0)
					{
						outptr[j] = inptr[j] * slope;
					}
					else
					{
						outptr[j] = inptr[j];
					}					
				}
			}
		}

		return 0;
	}

	REGISTER_LAYER_CLASS(ReLU, Relu);
}
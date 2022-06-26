#include "layer_dropout.h"
#include "layer_factory.h"
#include "platform.h"


namespace ACNN
{
	Dropout::Dropout(const LayerParam& layer_param)
		: Layer(layer_param)
	{}

	int Dropout::load_param(const ParamDict& pd)
	{
		scale = pd.get(0, 1.f);

		return 0;
	}

	int Dropout::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
	{
		if (bottom_blobs.size() != top_blobs.size())
		{
			ACNN_LOGE("Dropout ERROR: bottom_blobs size(%d) != top_blobs size(%d)", (int)bottom_blobs.size(), (int)top_blobs.size());
			return -1;
		}

		for (size_t i = 0; i < bottom_blobs.size(); i++)
		{
			const aMat& bottom_blob = bottom_blobs[i];
			aMat& top_blob = top_blobs[i];
			int inw = bottom_blob.m_w;
			int inh = bottom_blob.m_h;
			int inch = bottom_blob.m_c;

			top_blob.create(inw, inh, inch, bottom_blob.m_elemsize, bottom_blob.m_allocator);
			for (int q = 0; q < inch; q++)
			{
				const float* inptr = bottom_blob.channel(q);
				float* outptr = top_blob.channel(q);
				for (int j = 0; j < inw * inh; j++)
				{
					outptr[j] = inptr[j] * scale;
				}
			}
		}

		return 0;
	}

	REGISTER_LAYER_CLASS(Dropout, Dropout);
}
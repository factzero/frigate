#include "layer_softmax.h"
#include "layer_factory.h"
#include "platform.h"


namespace ACNN
{
	Softmax::Softmax(const LayerParam& layer_param)
		: Layer(layer_param)
	{}

	int Softmax::load_param(const ParamDict& pd)
	{
		axis = pd.get(0, 0);

		return 0;
	}

	int Softmax::forward(const std::vector<aMat>& bottom_blobs, std::vector<aMat>& top_blobs, const Option& opt) const
	{
		if (bottom_blobs.size() != top_blobs.size())
		{
			ACNN_LOGE("Softmax ERROR: bottom_blobs size(%d) != top_blobs size(%d)", (int)bottom_blobs.size(), (int)top_blobs.size());
			return -1;
		}

		for (size_t i = 0; i < bottom_blobs.size(); i++)
		{
			const aMat& bottom_blob = bottom_blobs[i];
			aMat& top_blob = top_blobs[i];
			int dims = bottom_blob.m_dims;

			if (dims == 1) // positive_axis == 0
			{
				int w = bottom_blob.m_w;
				const float* ptr = bottom_blob;
				float max = -FLT_MAX;
				for (int j = 0; j < w; j++)
				{
					max = std::max(max, ptr[j]);
				}

				top_blob.create(w, bottom_blob.m_elemsize, bottom_blob.m_allocator);
				float* outptr = top_blob;
				float sum = 0.f;
				for (int j = 0; j < w; j++)
				{
					outptr[j] = static_cast<float>(exp(ptr[j] - max));
					sum += outptr[j];
				}

				for (int j = 0; j < w; j++)
				{
					outptr[j] /= sum;
				}
			}
			else
			{
				ACNN_LOGE("Softmax Type is not supported");
			}
		}

		return 0;
	}

	REGISTER_LAYER_CLASS(Softmax, Softmax);
}
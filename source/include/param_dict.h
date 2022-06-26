#pragma once
#include "data_reader.h"
#include "amat.h"


// at most 32 parameters
#define ACNN_MAX_PARAM_COUNT 32

namespace ACNN
{
	class ParamDict
	{
	public:
		ParamDict();
		virtual ~ParamDict() {}

		int load_param(const DataReader& dr);

		// get int
		int get(int id, int def) const;
		// get float
		float get(int id, float def) const;
		// get array
		aMat get(int id, const aMat& def) const;

	private:
		struct
		{
			// 0 = null
			// 1 = int/float
			// 2 = int
			// 3 = float
			// 4 = array of int/float
			// 5 = array of int
			// 6 = array of float
			int type;
			union
			{
				int i;
				float f;
			};
			aMat v;
		} params[ACNN_MAX_PARAM_COUNT];

		void clear();
	};

}
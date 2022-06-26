#pragma once
#include <string>
#include "amat.h"


namespace ACNN
{
	class Blob
	{
	public:
		Blob();

		void set_data(const aMat& data) { m_data = data; }
		aMat get_data() { return m_data; }
	
	public:
		std::string name;
		int producer;
		int consumer;

	private:
		aMat m_data;
	};
}
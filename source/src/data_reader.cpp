#include "data_reader.h"
#include "platform.h"


namespace ACNN
{
	DataReaderFromFile::DataReaderFromFile(const char* file_path)
		: DataReaderAPI()
	{
		if (fopen_s(&m_fp, file_path, "rb"))
		{
			ACNN_LOGE("fopen %s failed", file_path);
		}
	}

	DataReaderFromFile::~DataReaderFromFile()
	{
		if (m_fp)
		{
			fclose(m_fp);
		}
	}

	int DataReaderFromFile::scan(const char* format, void* p) const
	{
		return fscanf(m_fp, format, p);
	}

	size_t DataReaderFromFile::read(void* buf, size_t size) const
	{
		return fread(buf, 1, size, m_fp);
	}
}
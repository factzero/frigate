#pragma once
#include <memory>
#include <stdio.h>


namespace ACNN
{
	class DataReaderAPI
	{
	public:
		DataReaderAPI() {}
		virtual~DataReaderAPI() {}

		virtual int scan(const char* format, void* p) const = 0;
		virtual size_t read(void* buf, size_t size) const = 0;
	};

	typedef std::shared_ptr<DataReaderAPI> DataReader;

	class DataReaderFromFile : public DataReaderAPI
	{
	public:
		explicit DataReaderFromFile(const char* file_path);
		~DataReaderFromFile();

		virtual int scan(const char* format, void* p) const override;
		virtual size_t read(void* buf, size_t size) const override;

	private:
		FILE* m_fp;
	};

}
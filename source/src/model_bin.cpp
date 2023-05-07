#include "model_bin.h"
#include "logger.h"


namespace ACNN
{
    aMat ModelBin::load(int w, int h, int type) const
    {
        aMat m = load(w * h, type);

        return m.reshape(w, h);
    }

    aMat ModelBin::load(int w, int h, int c, int type) const
    {
        aMat m = load(w * h * c, type);

        return m.reshape(w, h, c);
    }


    ModelBinFromDataReader::ModelBinFromDataReader(const DataReader& dr)
        : dr(dr)
    {}

    aMat ModelBinFromDataReader::load(int w, int type) const
    {
        aMat m;

        if (type == 0)
        {
            size_t nread;

            union
            {
                struct
                {
                    unsigned char f0;
                    unsigned char f1;
                    unsigned char f2;
                    unsigned char f3;
                };
                unsigned int tag;
            } flag_struct;

            nread = dr->read(&flag_struct, sizeof(flag_struct));
            if (nread != sizeof(flag_struct))
            {
                ConsoleELog << "ModelBin read flag_struct failed " << nread;
                return aMat();
            }

            unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

            if (flag_struct.tag == 0x01306B47)
            {
                // half-precision data
                ConsoleELog << "ModelBin load half-precision data not supported";
                return aMat();
            }
            else if (flag_struct.tag == 0x000D4B38)
            {
                // int8 data
                size_t align_data_size = alignSize(w, 4);

                m.create((int)align_data_size, (size_t)1u);
                if (m.empty())
                {
                    return m;
                }

                nread = dr->read(m, align_data_size);
                if (nread != align_data_size)
                {
                    ConsoleELog << "ModelBin read flag_struct failed " << nread;
                    return aMat();
                }

                return m;
            }
            else if (flag_struct.tag == 0x0002C056)
            {
                m.create(w);
                if (m.empty())
                {
                    return m;
                }

                // raw data with extra scaling
                nread = dr->read(m, w * sizeof(float));
                if (nread != w * sizeof(float))
                {
                    ConsoleELog << "ModelBin read weight_data failed " << nread;
                    return aMat();
                }

                return m;
            }

            if (flag != 0)
            {
                // quantized data
                m.create(w);
                if (m.empty())
                {
                    return m;
                }

                float quantization_value[256];
                nread = dr->read(quantization_value, 256 * sizeof(float));
                if (nread != 256 * sizeof(float))
                {
                    ConsoleELog << "ModelBin read quantization_value failed " << nread;
                    return aMat();
                }

                size_t align_weight_data_size = alignSize(w * sizeof(unsigned char), 4);
                std::vector<unsigned char> index_array;
                index_array.resize(align_weight_data_size);
                nread = dr->read(index_array.data(), align_weight_data_size);
                if (nread != align_weight_data_size)
                {
                    ConsoleELog << "ModelBin read index_array failed " << nread;
                    return aMat();
                }

                float* ptr = m;
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = quantization_value[index_array[i]];
                }
            }
            else if (flag_struct.f0 == 0)
            {
                m.create(w);
                if (m.empty())
                {
                    return m;
                }

                // raw data
                nread = dr->read(m, w * sizeof(float));
                if (nread != w * sizeof(float))
                {
                    ConsoleELog << "ModelBin read weight_data failed " << nread;
                    return aMat();
                }
            }

            return m;
        }
        else if (type == 1)
        {
            m.create(w);
            if (m.empty())
            {
                return m;
            }

            // raw data
            size_t nread = dr->read(m, w * sizeof(float));
            if (nread != w * sizeof(float))
            {
                ConsoleELog << "ModelBin read weight_data failed " << nread;
                return aMat();
            }

            return m;
        }
        else
        {
            ConsoleELog << "ModelBin load type " << type << " not implemented";
            return aMat();
        }

        return aMat();
    }

    ModelBinFromMatArray::ModelBinFromMatArray(std::vector<aMat>& d)
        :ModelBin(), m_data(d)
    {}

    aMat ModelBinFromMatArray::load(int w, int type) const
    {
        if (m_data.empty())
        {
            return aMat();
        }

        aMat m = m_data.front();
        m_data.erase(m_data.begin());

        return m;
    }
}
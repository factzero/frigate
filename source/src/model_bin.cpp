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
                ConsoleELog << "ModelBin load int8 data not supported";
                return aMat();
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
                ConsoleELog << "ModelBin load quantized data not supported";
                return aMat();
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
}
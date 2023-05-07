#include <ctype.h>
#include "param_dict.h"
#include "logger.h"


namespace ACNN
{
    static bool vstr_is_float(const char vstr[16])
    {
        // look ahead for determine isfloat
        for (int j = 0; j < 16; j++)
        {
            if (vstr[j] == '\0')
                break;

            if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
                return true;
        }

        return false;
    }

    static float vstr_to_float(const char vstr[16])
    {
        double v = 0.0;

        const char* p = vstr;

        // sign
        bool sign = *p != '-';
        if (*p == '+' || *p == '-')
        {
            p++;
        }

        // digits before decimal point or exponent
        unsigned int v1 = 0;
        while (isdigit(*p))
        {
            v1 = v1 * 10 + (*p - '0');
            p++;
        }

        v = (double)v1;

        // digits after decimal point
        if (*p == '.')
        {
            p++;

            unsigned int pow10 = 1;
            unsigned int v2 = 0;

            while (isdigit(*p))
            {
                v2 = v2 * 10 + (*p - '0');
                pow10 *= 10;
                p++;
            }

            v += v2 / (double)pow10;
        }

        // exponent
        if (*p == 'e' || *p == 'E')
        {
            p++;

            // sign of exponent
            bool fact = *p != '-';
            if (*p == '+' || *p == '-')
            {
                p++;
            }

            // digits of exponent
            unsigned int expon = 0;
            while (isdigit(*p))
            {
                expon = expon * 10 + (*p - '0');
                p++;
            }

            double scale = 1.0;
            while (expon >= 8)
            {
                scale *= 1e8;
                expon -= 8;
            }
            while (expon > 0)
            {
                scale *= 10.0;
                expon -= 1;
            }

            v = fact ? v * scale : v / scale;
        }

        return sign ? (float)v : (float)-v;
    }

    ParamDict::ParamDict()
    {
        clear();
    }

    int ParamDict::load_param(const DataReader& dr)
    {
        clear();

        // parse each key=value pair
        int id = 0;
        while (dr->scan("%d=", &id) == 1)
        {
            bool is_array = id <= -23300;
            if (is_array)
            {
                id = -id - 23300;
            }

            if (id >= ACNN_MAX_PARAM_COUNT)
            {
                ConsoleELog << "id < NCNN_MAX_PARAM_COUNT failed (id=;" << id << ", NCNN_MAX_PARAM_COUNT=" << ACNN_MAX_PARAM_COUNT << ")";
                return -1;
            }

            if (is_array)
            {
                int len = 0;
                int nscan = dr->scan("%d", &len);
                if (nscan != 1)
                {
                    ConsoleELog << "ParamDict read array length failed";
                    return -1;
                }

                params[id].v.create(len);

                for (int j = 0; j < len; j++)
                {
                    char vstr[16];
                    nscan = dr->scan(",%15[^,\n ]", vstr);
                    if (nscan != 1)
                    {
                        ConsoleELog << "ParamDict read array element failed";
                        return -1;
                    }

                    bool is_float = vstr_is_float(vstr);

                    if (is_float)
                    {
                        float* ptr = params[id].v;
                        ptr[j] = vstr_to_float(vstr);
                    }
                    else
                    {
                        int* ptr = params[id].v;
                        nscan = sscanf_s(vstr, "%d", &ptr[j]);
                        if (nscan != 1)
                        {
                            ConsoleELog << "ParamDict parse array element failed";
                            return -1;
                        }
                    }

                    params[id].type = is_float ? 6 : 5;
                }
            }
            else
            {
                char vstr[16];
                int nscan = dr->scan("%15s", vstr);
                if (nscan != 1)
                {
                    ConsoleELog << "ParamDict read value failed";
                    return -1;
                }

                bool is_float = vstr_is_float(vstr);

                if (is_float)
                {
                    params[id].f = vstr_to_float(vstr);
                }
                else
                {
                    nscan = sscanf_s(vstr, "%d", &params[id].i);
                    if (nscan != 1)
                    {
                        ConsoleELog << "ParamDict parse value failed";
                        return -1;
                    }
                }

                params[id].type = is_float ? 3 : 2;
            }
        }

        return 0;
    }

    int ParamDict::get(int id, int def) const
    {
        return params[id].type ? params[id].i : def;
    }
    
    float ParamDict::get(int id, float def) const
    {
        return params[id].type ? params[id].f : def;
    }
    
    aMat ParamDict::get(int id, const aMat& def) const
    {
        return params[id].type ? params[id].v : def;
    }

    void ParamDict::set(int id, int i)
    {
        params[id].type = 2;
        params[id].i = i;
    }

    void ParamDict::set(int id, float f)
    {
        params[id].type = 3;
        params[id].f = f;
    }

    void ParamDict::set(int id, const aMat& v)
    {
        params[id].type = 4;
        params[id].v = v;
    }

    void ParamDict::clear()
    {
        for (int i = 0; i < ACNN_MAX_PARAM_COUNT; i++)
        {
            params[i].type = 0;
            params[i].v = aMat();
        }

        return;
    }
}
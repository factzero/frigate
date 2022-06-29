#include "amat.h"


namespace ACNN
{
    aMat::aMat()
        : m_w(0), m_h(0), m_c(0), m_cstep(0), m_dims(0), m_elemsize(0), m_pvdata(nullptr), m_allocator(nullptr), m_pirefcount(nullptr)
    {}

    aMat::aMat(int w, int elemsize, Allocator allocator)
        : m_w(0), m_h(0), m_c(0), m_cstep(0), m_dims(0), m_elemsize(0), m_pvdata(nullptr), m_allocator(nullptr), m_pirefcount(nullptr)
    {
        create(w, elemsize, allocator);
    }

    aMat::aMat(int w, int h, int elemsize, Allocator allocator)
        : m_w(0), m_h(0), m_c(0), m_cstep(0), m_dims(0), m_elemsize(0), m_pvdata(nullptr), m_allocator(nullptr), m_pirefcount(nullptr)
    {
        create(w, h, elemsize, allocator);
    }

    aMat::aMat(int w, int h, int c, int elemsize, Allocator allocator)
        : m_w(0), m_h(0), m_c(0), m_cstep(0), m_dims(0), m_elemsize(0), m_pvdata(nullptr), m_allocator(nullptr), m_pirefcount(nullptr)
    {
        create(w, h, c, elemsize, allocator);
    }

    aMat::aMat(const aMat& m)
        : m_w(m.m_w), m_h(m.m_h), m_c(m.m_c), m_cstep(m.m_cstep), m_dims(m.m_dims), m_elemsize(m.m_elemsize), m_pvdata(m.m_pvdata), m_allocator(m.m_allocator), m_pirefcount(m.m_pirefcount)
    {
        addrefcount();
    }

    aMat::aMat(int w, int h, void* data, int elemsize, Allocator allocator)
        : m_w(w), m_h(h), m_c(1), m_dims(2), m_elemsize(elemsize), m_pvdata(data), m_allocator(allocator), m_pirefcount(nullptr)
    {
        m_cstep = w * h;
    }

    aMat::aMat(int w, int h, int c, void* data, int elemsize, Allocator allocator)
        : m_w(w), m_h(h), m_c(c), m_dims(3), m_elemsize(elemsize), m_pvdata(data), m_allocator(allocator), m_pirefcount(nullptr)
    {
        m_cstep = w * h;
    }

    aMat::~aMat()
    {
        release();
    }

    aMat& aMat::operator=(const aMat& m)
    {
        if (this == &m)
        {
            return *this;
        }

        if (m.m_pirefcount)
        {
            ACNN_XADD(m.m_pirefcount, 1);
        }

        release();

        m_w = m.m_w;
        m_h = m.m_h;
        m_c = m.m_c;
        m_cstep = m.m_cstep;
        m_dims = m.m_dims;
        m_elemsize = m.m_elemsize;
        m_pvdata = m.m_pvdata;
        m_pirefcount = m.m_pirefcount;
        m_allocator = m.m_allocator;

        return *this;
    }

    void aMat::create(int w, int elemsize, Allocator allocator)
    {
        if (1 == m_dims && m_w == w && m_allocator == allocator)
        {
            return;
        }

        release();

        m_w = w;
        m_h = 1;
        m_c = 1;
        m_dims = 1;
        m_cstep = w;
        m_elemsize = elemsize;
        m_allocator = allocator;

        if (total() > 0)
        {
            int total_size = alignSize(total() * m_elemsize, 4);
            if (m_allocator)
            {
                m_pvdata = m_allocator->fastMalloc(total_size + sizeof(*m_pirefcount));
            }
            else
            {
                m_pvdata = fastMalloc(total_size + sizeof(*m_pirefcount));
            }
            m_pirefcount = (int*)((unsigned char*)m_pvdata + total_size);
            *m_pirefcount = 1;
        }

        return;
    }

    void aMat::create(int w, int h, int elemsize, Allocator allocator)
    {
        if (2 == m_dims && m_w == w && m_h == h && m_elemsize == elemsize && m_allocator == allocator)
        {
            return;
        }

        release();

        m_w = w;
        m_h = h;
        m_c = 1;
        m_dims = 2;
        m_cstep = w * h;
        m_elemsize = elemsize;
        m_allocator = allocator;

        if (total() > 0)
        {
            int total_size = alignSize(total() * m_elemsize, 4);
            if (m_allocator)
            {
                m_pvdata = m_allocator->fastMalloc(total_size + sizeof(*m_pirefcount));
            }
            else
            {
                m_pvdata = fastMalloc(total_size + sizeof(*m_pirefcount));
            }
            m_pirefcount = (int*)((unsigned char*)m_pvdata + total_size);
            *m_pirefcount = 1;
        }

        return;
    }

    void aMat::create(int w, int h, int c, int elemsize, Allocator allocator)
    {
        if (2 == m_dims && m_w == w && m_h == h && m_c == c && m_elemsize == elemsize && m_allocator == allocator)
        {
            return;
        }

        release();

        m_w = w;
        m_h = h;
        m_c = c;
        m_dims = 3;
        m_cstep = alignSize(w * h * elemsize, 16) / elemsize;;
        m_elemsize = elemsize;
        m_allocator = allocator;

        if (total() > 0)
        {
            int total_size = alignSize(total() * m_elemsize, 4);
            if (m_allocator)
            {
                m_pvdata = m_allocator->fastMalloc(total_size + sizeof(*m_pirefcount));
            }
            else
            {
                m_pvdata = fastMalloc(total_size + sizeof(*m_pirefcount));
            }
            m_pirefcount = (int*)((unsigned char*)m_pvdata + total_size);
            *m_pirefcount = 1;
        }

        return;
    }

    aMat aMat::reshape(int w, Allocator allocator) const
    {
        if (m_w * m_h * m_c != w)
        {
            return aMat();
        }

        if (m_dims >= 3 && m_cstep != m_w * m_h)
        {
            aMat m;
            m.create(w, m_elemsize, allocator);

            // flatten
            for (int i = 0; i < m_c; i++)
            {
                const void* ptr = (unsigned char*)m_pvdata + i * m_cstep * m_elemsize;
                void* mptr = (unsigned char*)m.m_pvdata + i * m_w * m_h * m_elemsize;
                memcpy(mptr, ptr, m_w * m_h * m_elemsize);
            }

            return m;
        }

        aMat m = *this;

        m.m_dims = 1;
        m.m_w = w;
        m.m_h = 1;
        m.m_c = 1;
        m.m_cstep = w;

        return m;
    }

    aMat aMat::reshape(int w, int h, Allocator allocator) const
    {
        if ( m_w * m_h * m_c != w * h)
        {
            return aMat();
        }

        if (m_dims >= 3 && m_cstep != w * h)
        {
            aMat m;
            m.create(w, h, m_elemsize, allocator);

            // flatten
            for (int i = 0; i < m_c; i++)
            {
                const void* ptr = (unsigned char*)m_pvdata + i * m_cstep * m_elemsize;
                void* mptr = (unsigned char*)m.m_pvdata + i * m_w * m_h * m_elemsize;
                memcpy(mptr, ptr, m_w * m_h * m_elemsize);
            }

            return m;
        }

        aMat m = *this;

        m.m_dims = 2;
        m.m_w = w;
        m.m_h = h;
        m.m_c = 1;
        m.m_cstep = w * h;

        return m;
    }

    aMat aMat::reshape(int w, int h, int c, Allocator allocator) const
    {
        if (m_w * m_h * m_c != w * h * c)
        {
            return aMat();
        }

        if (m_dims < 3)
        {
            if (w * h != alignSize(w * h * m_elemsize, 16) / m_elemsize)
            {
                aMat m;
                m.create(w, h, c, m_elemsize, allocator);

                // align channel
                for (int i = 0; i < c; i++)
                {
                    const void* ptr = (unsigned char*)m_pvdata + i * w * h * m_elemsize;
                    void* mptr = (unsigned char*)m.m_pvdata + i * m.m_cstep * m.m_elemsize;
                    memcpy(mptr, ptr, w * h * m_elemsize);
                }

                return m;
            }
        }
        else if (m_c != c)
        {
            // flatten and then align
            aMat tmp = reshape(w * h * c, allocator);
            return tmp.reshape(w, h, c, allocator);
        }

        aMat m = *this;

        m.m_dims = 3;
        m.m_w = w;
        m.m_h = h;
        m.m_c = c;
        m.m_cstep = alignSize(w * h * m_elemsize, 16) / m_elemsize;

        return m;
    }

    bool aMat::empty() const
    {
        return m_pvdata == nullptr || total() == 0;
    }

    aMat aMat::channel(int c) const
    {
        return aMat(m_w, m_h, (unsigned char*)m_pvdata + m_cstep * c * m_elemsize, m_elemsize, m_allocator);
    }

    float& aMat::operator[](int i)
    {
        return ((float*)m_pvdata)[i];
    }

    const float& aMat::operator[](int i) const
    {
        return ((const float*)m_pvdata)[i];
    }

    void aMat::release()
    {
        if (m_pirefcount && ACNN_XADD(m_pirefcount, -1) == 1)
        {
            if (m_allocator)
            {
                m_allocator->fastFree(m_pvdata);
            }
            else
            {
                fastFree(m_pvdata);
            }
        }

        m_w = 0;
        m_h = 0;
        m_c = 0;
        m_cstep = 0;
        m_dims = 0;
        m_elemsize = 0;
        m_pvdata = nullptr;
        m_pirefcount = nullptr;
        m_allocator = nullptr;

        return;
    }

    void aMat::addrefcount()
    {
        if (m_pirefcount)
        {
            ACNN_XADD(m_pirefcount, 1);
        }

        return;
    }

    void copy_make_border(const aMat& src, aMat& dst, int top, int bottom, int left, int right, int type, float v)
    {
        int w = src.m_w;
        int h = src.m_h;
        int ch = src.m_c;
        int w_new = w + left + right;
        int h_new = h + top + bottom;

        dst.create(w_new, h_new, ch, src.m_elemsize, src.m_allocator);
        for (int q = 0; q < ch; q++)
        {
            const float* inptr = src.channel(q);
            float* outptr = dst.channel(q);
            // top
            for (int y = 0; y < top; y++)
            {
                for (int x = 0; x < w_new; x++)
                {
                    outptr[y * w_new + x] = v;
                }
            }
            for (int y = top; y < (h + top); y++)
            {
                // left
                for (int x = 0; x < left; x++)
                {
                    outptr[y * w_new + x] = v;
                }
                // right
                for (int x = w + left; x < w_new; x++)
                {
                    outptr[y * w_new + x] = v;
                }
            }
            // bottom
            for (int y = h + top; y < h_new; y++)
            {
                for (int x = 0; x < w_new; x++)
                {
                    outptr[y * w_new + x] = v;
                }
            }

            for (int y = 0; y < h; y++)
            {
                memcpy((void*)&outptr[top*w_new + left + y*w_new], (void*)&inptr[y*w], w*src.m_elemsize);
            }
        }

        return;
    }
}
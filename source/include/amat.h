#pragma once
#include "allocate.h"


namespace ACNN
{
    class aMat
    {
    public:
        aMat();
        aMat(int w, size_t elemsize = 4, Allocator allocator = nullptr);
        aMat(int w, int h, size_t elemsize = 4, Allocator allocator = nullptr);
        aMat(int w, int h, int c, size_t elemsize = 4, Allocator allocator = nullptr);
        aMat(const aMat& m);
        aMat(int w, int h, void* data, size_t elemsize = 4, Allocator allocator = nullptr);
        aMat(int w, int h, int c, void* data, size_t elemsize = 4, Allocator allocator = nullptr);
        ~aMat();

        aMat& operator=(const aMat& m);

        void fill(float v);
        void fill(int v);

        void create(int w, size_t elemsize = 4, Allocator allocator = nullptr);
        void create(int w, int h, size_t elemsize = 4, Allocator allocator = nullptr);
        void create(int w, int h, int c, size_t elemsize = 4, Allocator allocator = nullptr);

        aMat reshape(int w, Allocator allocator = nullptr) const;
        aMat reshape(int w, int h, Allocator allocator = nullptr) const;
        aMat reshape(int w, int h, int c, Allocator allocator = nullptr) const;

        size_t total() const { return m_cstep * m_c; }
        bool empty() const;

        aMat channel(int c) const;

        // access raw data
        template<typename T>
        operator T* ();
        template<typename T>
        operator const T* () const;

        // convenient access float vec element
        float& operator[](int i);
        const float& operator[](int i) const;

    private:
        void release();
        void addrefcount();

    public:
        int m_w;
        int m_h;
        int m_c;
        size_t m_cstep;
        int m_dims;
        size_t m_elemsize;
        void* m_pvdata;
        int* m_pirefcount;
        Allocator m_allocator;

    };

    template<typename T>
    aMat::operator T* ()
    {
        return (T*)m_pvdata;
    }

    template<typename T>
    aMat::operator const T* () const
    {
        return (const T*)m_pvdata;
    }
}
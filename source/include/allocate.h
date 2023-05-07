#pragma once
#include <memory>
#include <malloc.h>


namespace ACNN
{
#define ACNN_MALLOC_ALIGN 64

    inline size_t alignSize(size_t sz, int n)
    {
        return (sz + n - 1) & -n;
    }

    inline void* fastMalloc(size_t size)
    {
        return _aligned_malloc(size, ACNN_MALLOC_ALIGN);
    }

    inline void fastFree(void* ptr)
    {
        if (ptr)
        {
            _aligned_free(ptr);
        }
    }

    inline int ACNN_XADD(int* addr, int delta)
    {
        int tmp = *addr;
        *addr += delta;
        return tmp;
    }

    class AllocatorAPI
    {
    public:
        virtual ~AllocatorAPI() {}
        virtual void* fastMalloc(size_t size) = 0;
        virtual void fastFree(void* ptr) = 0;
    };

    typedef std::shared_ptr<AllocatorAPI> Allocator;
}
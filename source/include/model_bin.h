#pragma once
#include <vector>
#include "amat.h"
#include "data_reader.h"


namespace ACNN
{
    class ModelBin
    {
    public:
        ModelBin() {}
        virtual ~ModelBin() {}
        // element type
        // 0 = auto
        // 1 = float32
        // 2 = float16
        // 3 = int8
        // load vec
        virtual aMat load(int w, int type) const = 0;
        // load image
        virtual aMat load(int w, int h, int type) const;
        // load dim
        virtual aMat load(int w, int h, int c, int type) const;
    };

    class ModelBinFromDataReader : public ModelBin
    {
    public:
        explicit ModelBinFromDataReader(const DataReader& dr);
        virtual ~ModelBinFromDataReader() {}

        virtual aMat load(int w, int type) const override;

    private:
        ModelBinFromDataReader(const ModelBinFromDataReader&);
        ModelBinFromDataReader& operator=(const ModelBinFromDataReader&);

    private:
        const DataReader& dr;
    };

    class ModelBinFromMatArray : public ModelBin
    {
    public:
        explicit ModelBinFromMatArray(std::vector<aMat>& d);
        virtual ~ModelBinFromMatArray() {}

        virtual aMat load(int w, int type) const override;

    private:
        ModelBinFromMatArray(const ModelBinFromMatArray&);
        ModelBinFromMatArray& operator=(const ModelBinFromMatArray&);

    private:
        mutable std::vector<aMat> m_data;
    };
}
#include "layer_factory.h"
#include "logger.h"


namespace ACNN
{
    LayerRegistry::CreatorRegistry& LayerRegistry::Registry()
    {
        static CreatorRegistry g_creator_registry;
        return g_creator_registry;
    }

    void LayerRegistry::AddCreator(const std::string& type, Creator creator)
    {
        CreatorRegistry& registry = Registry();
        if (registry.count(type) != 0)
        {
            ConsoleELog << "Layer type " << type << " already registered.";
        }
        registry[type] = creator;

        return;
    }

    std::shared_ptr<Layer> LayerRegistry::CreateLayer(const LayerParam& param)
    {
        CreatorRegistry& registry = Registry();
        const std::string& type = param.layer_type;
        if (registry.count(type) != 1)
        {
            ConsoleELog << "Unknown layer type: " << type << " (known types: " << LayerTypeListString() << ")";
        }

        return registry[type](param);
    }

    std::vector<std::string> LayerRegistry::LayerTypeList()
    {
        CreatorRegistry& registry = Registry();
        std::vector<std::string> layer_types;
        for (auto& iter : registry) 
        {
            layer_types.push_back(iter.first);
        }

        return layer_types;
    }

    std::string LayerRegistry::LayerTypeListString()
    {
        std::vector<std::string> layer_types = LayerTypeList();
        std::string layer_types_str;
        for (std::vector<std::string>::iterator iter = layer_types.begin(); iter != layer_types.end(); ++iter)
        {
            if (iter != layer_types.begin()) {
                layer_types_str += ", ";
            }
            layer_types_str += *iter;
        }

        return layer_types_str;
    }

    LayerRegisterer::LayerRegisterer(const std::string& type, std::shared_ptr<Layer>(*creator)(const LayerParam&))
    {
        LayerRegistry::AddCreator(type, creator);
    }
}
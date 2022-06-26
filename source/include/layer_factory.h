#pragma once
#include <map>
#include "layer.h"


namespace ACNN
{
	class LayerRegistry 
	{
	public:
		typedef std::shared_ptr<Layer>(*Creator)(const LayerParam&);
		typedef std::map<std::string, Creator> CreatorRegistry;

		static CreatorRegistry& Registry();

		// Adds a creator.
		static void AddCreator(const std::string& type, Creator creator);

		// Get a layer using a LayerParameter.
		static std::shared_ptr<Layer> CreateLayer(const LayerParam& param);

		static std::vector<std::string> LayerTypeList();

	private:
		LayerRegistry();

		static std::string LayerTypeListString();
	};

	class LayerRegisterer 
	{
	public:
		LayerRegisterer(const std::string& type, std::shared_ptr<Layer>(*creator)(const LayerParam&));
	};

#define REGISTER_LAYER_CLASS(type, layer)                                       \
	std::shared_ptr<Layer> Creator_##type##layer(const LayerParam& param)       \
	{                                                                           \
		return std::make_shared<layer>(param);                                  \
	}                                                                           \
	static LayerRegisterer g_creator_f_##type(#type, Creator_##type##layer)
}
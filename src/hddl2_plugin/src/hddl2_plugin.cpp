//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

// System include
#include <map>
#include <memory>
#include <string>

// Inference Engine include
#include <graph_transformer.h>
#include <cnn_network_impl.hpp>
#include <convert_function_to_cnn_network.hpp>
#include <details/ie_irelease.hpp>
#include <fstream>
#include <generic_ie.hpp>
#include <ie_icore.hpp>
#include <ie_itt.hpp>
#include <ie_metric_helpers.hpp>
#include <ie_util_internal.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>

// Plugin include
#include "vpux.hpp"
#include "hddl2_exceptions.h"
#include "file_reader.h"
#include "hddl2_executable_network.h"
#include "hddl2_params.hpp"
#include "hddl2_plugin.h"
#include "hddl2_metrics.h"
#include "ie_macro.hpp"
#include "hddl2_remote_context.h"
// Subplugin
#include "subplugin/hddl2_backend.h"

namespace vpux {
namespace HDDL2 {

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static VPUXConfig mergePluginAndNetworkConfigs(
    const VPUXConfig& pluginConfig, const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = pluginConfig;
    parsedConfigCopy.update(config);
    return parsedConfigCopy;
}

//------------------------------------------------------------------------------
Engine::Engine(): _backends(std::make_shared<VPUXBackends>(_parsedConfig)),
                  _metrics(_backends) {
    _pluginName = DEVICE_NAME;  // "VPUX"
    // TODO Different backends can require different compilers in future
    _compiler = vpux::Compiler::create(vpux::CompilerType::MCMCompiler);
    _parsedConfig.expandSupportedOptions(_compiler->getSupportedOptions());
}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
ExecutableNetworkInternal::Ptr Engine::LoadExeNetwork(
    const ICNNNetwork& network, std::shared_ptr<vpux::IDevice>& device, const VPUXConfig& networkConfig) {
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "LoadExeNetwork");
    std::shared_ptr<ICNNNetwork> clonedNetwork = cloneNetwork(network);

    auto implNetwork = std::dynamic_pointer_cast<CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
    }

    return std::make_shared<vpu::HDDL2Plugin::ExecutableNetwork>(*clonedNetwork, device, networkConfig);
}

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
    const ICNNNetwork& network, const std::map<std::string, std::string>& config) {
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDeviceToUse(networkConfig.deviceId());
    return LoadExeNetwork(network, device, networkConfig);
}

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
    const ICNNNetwork& network, RemoteContext::Ptr context, const std::map<std::string, std::string>& config) {
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDeviceFromContext(context);
    return LoadExeNetwork(network, device, networkConfig);
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
IExecutableNetwork::Ptr Engine::ImportNetwork(
    const std::string& modelFileName, const std::map<std::string, std::string>& config) {
    std::ifstream blobStream(modelFileName, std::ios::binary);
    return ImportNetworkImpl(vpu::KmbPlugin::utils::skipMagic(blobStream), config);
}

InferenceEngine::ExecutableNetwork Engine::ImportNetworkImpl(
    std::istream& networkModel, const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "ImportNetwork");
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    // TODO This backend instance should be replaced with VPUX after backend refactoring
    auto device = _backends->getDeviceToUse(networkConfig.deviceId());
    const auto executableNetwork = std::make_shared<vpu::HDDL2Plugin::ExecutableNetwork>(networkModel, device, networkConfig);
    return InferenceEngine::make_executable_network(executableNetwork);
}

InferenceEngine::ExecutableNetwork Engine::ImportNetworkImpl(
    std::istream& networkModel, const RemoteContext::Ptr& context, const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "ImportNetwork");
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDeviceFromContext(context);
    const auto executableNetwork = std::make_shared<vpu::HDDL2Plugin::ExecutableNetwork>(networkModel, device, networkConfig);
    return InferenceEngine::make_executable_network(executableNetwork);
}

//------------------------------------------------------------------------------
void Engine::SetConfig(const std::map<std::string, std::string>& config) {
    _parsedConfig.update(config);
    _backends->setup(_parsedConfig);
    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

void Engine::QueryNetwork(const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config,
    InferenceEngine::QueryNetworkResult& res) const {
    UNUSED(network);
    UNUSED(config);
    UNUSED(res);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED;
}

RemoteContext::Ptr Engine::CreateContext(const ParamMap& map) {
    // Device in this case will be searched inside RemoteContext creation
    return std::make_shared<vpu::HDDL2Plugin::HDDL2RemoteContext>(map, _parsedConfig);
}

InferenceEngine::Parameter Engine::GetMetric(
    const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& /*options*/) const {
    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics.GetAvailableDevicesNames());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, _metrics.SupportedMetrics());
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _metrics.GetFullDevicesNames());
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, _metrics.GetSupportedConfigKeys());
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, _metrics.GetOptimizationCapabilities());
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics.GetRangeForAsyncInferRequest());
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, _metrics.GetRangeForStreams());
    }
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "VPUXPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)

}  // namespace HDDL2
}  // namespace vpux


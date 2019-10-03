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

#pragma once

#include <map>
#include <string>

#include "inference_engine.hpp"
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"

namespace vpu {
namespace HDDL2Plugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine();

    InferenceEngine::ExecutableNetworkInternal::Ptr

    LoadExeNetworkImpl(const InferenceEngine::ICore *core,
                       InferenceEngine::ICNNNetwork &network,
                       const std::map <std::string, std::string> &config) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;

    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      const std::map<std::string, std::string> &config,
                      InferenceEngine::QueryNetworkResult &res) const override;

    InferenceEngine::IExecutableNetwork::Ptr ImportNetwork(
                        const std::string &modelFileName,
                        const std::map <std::string, std::string> &config) override;
};

}  //  namespace HDDL2Plugin
}  //  namespace vpu

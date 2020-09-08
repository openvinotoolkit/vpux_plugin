//
// Copyright 2020 Intel Corporation.
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

#include <Inference.h>

#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "hddl2_config.h"
#include "hddl2_executor.h"

namespace vpu {
namespace HDDL2Plugin {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<ExecutableNetwork>;

    explicit ExecutableNetwork(InferenceEngine::ICNNNetwork& network, const vpu::HDDL2Config& config,
        const InferenceEngine::RemoteContext::Ptr& context = nullptr);
    explicit ExecutableNetwork(std::istream& networkModel, const vpu::HDDL2Config& config,
        const InferenceEngine::RemoteContext::Ptr& context = nullptr);
    ~ExecutableNetwork() override = default;

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(
        const InferenceEngine::InputsDataMap networkInputs,
        const InferenceEngine::OutputsDataMap networkOutputs) override;
    void ExportImpl(std::ostream& model) override;

    using InferenceEngine::ExecutableNetworkInternal::Export;
    void Export(const std::string& modelFileName) override;

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr& asyncRequest) override;

private:
    explicit ExecutableNetwork(const vpu::HDDL2Config& config);

private:
    const HDDL2Config _config;
    const Logger::Ptr _logger;

    vpux::Compiler::Ptr _compiler = nullptr;
    vpux::NetworkDescription::Ptr _networkPtr = nullptr;
    vpux::Executor::Ptr _executorPtr;
};

}  //  namespace HDDL2Plugin
}  //  namespace vpu

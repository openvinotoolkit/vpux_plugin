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

#include <ie_common.h>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <threading/ie_executor_manager.hpp>
#include <vector>

#include "ie_remote_context.hpp"
#include "kmb_async_infer_request.h"
#include "kmb_config.h"
#include "kmb_executor.h"
#include "kmb_infer_request.h"
#include "kmb_remote_context.h"
#include "mcm_adapter.hpp"

namespace vpu {
namespace KmbPlugin {
namespace ie = InferenceEngine;

class ExecutableNetwork final : public ie::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<ExecutableNetwork>;

    explicit ExecutableNetwork(ie::ICNNNetwork& network, const KmbConfig& config, const ie::RemoteContext::Ptr& ctx);
    explicit ExecutableNetwork(std::istream& strm, const KmbConfig& config, const ie::RemoteContext::Ptr& ctx);

    ~ExecutableNetwork() = default;

    void GetMetric(const std::string& name, ie::Parameter& result, ie::ResponseDesc* resp) const override;
    void SetConfig(const std::map<std::string, ie::Parameter>& config, ie::ResponseDesc* resp) override;
    void GetConfig(const std::string& name, ie::Parameter& result, ie::ResponseDesc* resp) const override;

    ie::InferRequestInternal::Ptr CreateInferRequestImpl(
        ie::InputsDataMap networkInputs, ie::OutputsDataMap networkOutputs) override {
        // TODO: it would be better to use some interface for context
        // instead of a concrete KmbRemoteContext class
        const auto& kmbContext = std::dynamic_pointer_cast<KmbRemoteContext>(_remoteContext);
        if (kmbContext == nullptr) {
            THROW_IE_EXCEPTION << "Cannot cast context to KmbRemoteContext.";
        }
        return std::make_shared<KmbInferRequest>(
            networkInputs, networkOutputs, _stagesMetaData, _config, _executor, kmbContext->getAllocator(), _netName);
    }

    void CreateInferRequest(ie::IInferRequest::Ptr& asyncRequest) override {
        const auto& kmbContext = std::dynamic_pointer_cast<KmbRemoteContext>(_remoteContext);
        if (kmbContext == nullptr) {
            THROW_IE_EXCEPTION << "Cannot cast context to KmbRemoteContext.";
        }
        auto syncRequestImpl = std::make_shared<KmbInferRequest>(
            _networkInputs, _networkOutputs, _stagesMetaData, _config, _executor, kmbContext->getAllocator(), _netName);

        syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        auto taskExecutorGetResult = getNextTaskExecutor();
        auto asyncTreadSafeImpl = std::make_shared<KmbAsyncInferRequest>(
            syncRequestImpl, _taskExecutor, taskExecutorGetResult, _callbackExecutor, _logger);
        asyncRequest.reset(new ie::InferRequestBase<ie::AsyncInferRequestThreadSafeDefault>(asyncTreadSafeImpl),
            [](ie::IInferRequest* p) {
                p->Release();
            });
        asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    }

    void ExportImpl(std::ostream& model) override { model.write(_graphBlob.data(), _graphBlob.size()); }

    void Export(const std::string& modelFileName) override {
        if (!_graphBlob.empty()) {
            std::ofstream modelFile(modelFileName, std::ios::out | std::ios::binary);

            if (modelFile.is_open()) {
                ExportImpl(modelFile);
            } else {
                THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export";
            }
        }
    }

    void Export(std::ostream& networkModel) override { ExportImpl(networkModel); }

private:
    void ConfigureExecutor(const std::string& networkName);
    void LoadBlob();

    ie::ITaskExecutor::Ptr getNextTaskExecutor();

    Logger::Ptr _logger;
    KmbExecutor::Ptr _executor;
    std::vector<char> _graphBlob;
    std::vector<StageMetaInfo> _stagesMetaData;
    KmbConfig _config;
    std::map<std::string, ie::Parameter> _parsedConfig;
    std::vector<std::string> _supportedMetrics;

    const size_t _maxTaskExecutorGetResultCount = 1;
    std::queue<std::string> _taskExecutorGetResultIds;

    std::string _netName;
    ie::RemoteContext::Ptr _remoteContext = nullptr;
};

}  // namespace KmbPlugin
}  // namespace vpu

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "infer_request.hpp"
#include "npu_private_properties.hpp"

namespace vpux {

class IMDDevice final : public intel_npu::IDevice {
public:
    explicit IMDDevice(const std::string_view platform);

public:
    std::shared_ptr<intel_npu::IExecutor> createExecutor(
            const std::shared_ptr<const intel_npu::NetworkDescription>& network,
            const intel_npu::Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;

    std::shared_ptr<intel_npu::SyncInferRequest> createInferRequest(
            const std::shared_ptr<const intel_npu::ICompiledModel>& compiledModel,
            const std::shared_ptr<intel_npu::IExecutor>& executor, const intel_npu::Config& config) override {
        return std::make_shared<IMDInferRequest>(compiledModel, executor, config);
    }

private:
    std::string _platform;
};

}  // namespace vpux

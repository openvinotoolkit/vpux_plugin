//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "infer_request.hpp"
#include "npu_private_properties.hpp"

namespace intel_npu {

class IMDDevice final : public IDevice {
public:
    explicit IMDDevice(const std::string_view platform);

public:
    std::shared_ptr<IExecutor> createExecutor(const std::shared_ptr<const NetworkDescription>& network,
                                              const Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;
    uint32_t getSubDevId() const override;
    uint32_t getMaxNumSlices() const override;

    std::shared_ptr<SyncInferRequest> createInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel,
                                                         const std::shared_ptr<IExecutor>& executor,
                                                         const Config& config) override {
        return std::make_shared<IMDInferRequest>(compiledModel, executor, config);
    }

private:
    std::string _platform;
};

}  // namespace intel_npu

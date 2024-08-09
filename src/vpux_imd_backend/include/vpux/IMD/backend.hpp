//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "npu.hpp"

namespace intel_npu {

class IMDBackend final : public IEngineBackend {
public:
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& name) const override;
    const std::shared_ptr<IDevice> getDevice(const ov::AnyMap& params) const override;

    const std::vector<std::string> getDeviceNames() const override;

    const std::string getName() const override;

    void registerOptions(OptionsDesc& options) const override;

    bool isBatchingSupported() const override;
    bool isCommandQueueExtSupported() const override;
};

}  // namespace intel_npu

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "npu.hpp"

namespace vpux {

class IMDBackend final : public intel_npu::IEngineBackend {
public:
    const std::shared_ptr<intel_npu::IDevice> getDevice() const override;
    const std::shared_ptr<intel_npu::IDevice> getDevice(const std::string& name) const override;
    const std::shared_ptr<intel_npu::IDevice> getDevice(const ov::AnyMap& params) const override;

    const std::vector<std::string> getDeviceNames() const override;

    const std::string getName() const override;

    void registerOptions(intel_npu::OptionsDesc& options) const override;

    bool isBatchingSupported() const override;
};

}  // namespace vpux

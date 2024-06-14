//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <npu.hpp>

namespace vpux {
/**
 * @brief This is a class which emulates behavior of a backend without a device. Provided only for unit tests purposes.
 */
class NoDevicesTestBackend final : public vpux::IEngineBackend {
public:
    NoDevicesTestBackend() = default;

    const std::string getName() const override {
        return "OneDeviceTestBackend";
    }

    const std::shared_ptr<IDevice> getDevice() const override {
        return nullptr;
    }
    const std::shared_ptr<IDevice> getDevice(const std::string& /*deviceId*/) const override {
        return nullptr;
    }

    const std::shared_ptr<IDevice> getDevice(const ov::AnyMap& /*map*/) const override {
        return nullptr;
    }

    const std::vector<std::string> getDeviceNames() const override {
        return {};
    }
};

}  // namespace vpux

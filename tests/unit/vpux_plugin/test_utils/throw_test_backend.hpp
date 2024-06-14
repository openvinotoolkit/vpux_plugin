//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <npu.hpp>

namespace vpux {
/**
 * @brief This is a class which emulates behavior of a backend which throws exceptions. Provided only for unit tests
 * purposes.
 */
class ThrowTestBackend final : public vpux::IEngineBackend {
public:
    ThrowTestBackend() {
        OPENVINO_THROW("Error from ThrowTestBackend");
    }

    const std::string getName() const override {
        OPENVINO_THROW("Error from ThrowTestBackend");
        return "ThrowTest";
    }

    void registerOptions(OptionsDesc&) const override {
        OPENVINO_THROW("Error from ThrowTestBackend");
    }
    const std::shared_ptr<IDevice> getDevice() const override {
        OPENVINO_THROW("Error from ThrowTestBackend");
        return nullptr;
    }
    const std::shared_ptr<IDevice> getDevice(const std::string& /*deviceId*/) const override {
        OPENVINO_THROW("Error from ThrowTestBackend");
        return nullptr;
    }

    const std::shared_ptr<IDevice> getDevice(const ov::AnyMap& /*map*/) const override {
        OPENVINO_THROW("Error from ThrowTestBackend");
        return nullptr;
    }

    const std::vector<std::string> getDeviceNames() const override {
        OPENVINO_THROW("Error from ThrowTestBackend");
        return {};
    }
};

}  // namespace vpux

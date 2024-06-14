//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <npu.hpp>

namespace vpux {

/**
 * @brief These are a set of classes which emulates behavior of a backend with a single device. Provided only for unit
 * tests purposes.
 */
class DummyNPU3700Device final : public IDevice {
public:
    DummyNPU3700Device() {
    }
    std::shared_ptr<IExecutor> createExecutor(const std::shared_ptr<const NetworkDescription>& /*networkDescription*/,
                                              const Config& /*config*/) override {
        return nullptr;
    }

    std::string getName() const override {
        return "DummyNPU3700Device";
    }
    std::string getFullDeviceName() const override {
        return "Intel(R) NPU (DummyNPU3700Device)";
    }

    std::string dummyGetDeviceId() const {
        return "3700";
    }

    std::shared_ptr<SyncInferRequest> createInferRequest(
            const std::shared_ptr<const vpux::ICompiledModel>& /*compiledModel*/,
            const std::shared_ptr<IExecutor>& /*executor*/, const Config& /*config*/) override {
        return nullptr;
    }
};

class NPU3700TestBackend final : public vpux::IEngineBackend {
public:
    NPU3700TestBackend(): _dummyDevice(std::make_shared<DummyNPU3700Device>()) {
    }

    const std::string getName() const override {
        return "NPU3700TestBackend";
    }

    const std::shared_ptr<IDevice> getDevice() const override {
        return _dummyDevice;
    }

    const std::shared_ptr<IDevice> getDevice(const std::string& specificName) const override {
        if (specificName == _dummyDevice->getName())
            return _dummyDevice;
        else
            return nullptr;
    }

    const std::shared_ptr<IDevice> getDevice(const ov::AnyMap& paramMap) const override {
        const auto pm = paramMap.find(ov::device::id.name());
        std::string deviceId = pm->second.as<std::string>();

        if (deviceId == _dummyDevice->dummyGetDeviceId())
            return _dummyDevice;
        else
            return nullptr;
    }

    const std::vector<std::string> getDeviceNames() const override {
        return {_dummyDevice->getName(), "noOtherDevice"};
    }

private:
    std::shared_ptr<DummyNPU3700Device> _dummyDevice;
};

}  // namespace vpux

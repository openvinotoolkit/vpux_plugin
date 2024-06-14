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
class DummyNPU3720Device final : public IDevice {
public:
    DummyNPU3720Device() {
    }
    std::shared_ptr<IExecutor> createExecutor(const std::shared_ptr<const NetworkDescription>& /*networkDescription*/,
                                              const Config& /*config*/) override {
        return nullptr;
    }

    std::string getName() const override {
        return "3720.dummyDevice";
    }
    std::string getFullDeviceName() const override {
        return "Intel(R) NPU (DummyNPU3720Device)";
    }

    ov::device::UUID getUuid() const override {
        return {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x37, 0x20};
    }

    std::shared_ptr<SyncInferRequest> createInferRequest(
            const std::shared_ptr<const vpux::ICompiledModel>& /*compiledModel*/,
            const std::shared_ptr<IExecutor>& /*executor*/, const Config& /*config*/) override {
        return nullptr;
    }
};

class NPU3720TestBackend final : public vpux::IEngineBackend {
public:
    NPU3720TestBackend(): _dummyDevice(std::make_shared<DummyNPU3720Device>()) {
    }

    const std::string getName() const override {
        return "NPU3720TestBackend";
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

    const std::shared_ptr<IDevice> getDevice(const ov::AnyMap&) const override {
        return _dummyDevice;
    }

    const std::vector<std::string> getDeviceNames() const override {
        return {_dummyDevice->getName(), "noOtherDevice"};
    }

private:
    std::shared_ptr<DummyNPU3720Device> _dummyDevice;
};

}  // namespace vpux

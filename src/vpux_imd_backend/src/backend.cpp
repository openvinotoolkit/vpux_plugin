//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/backend.hpp"

#include "npu_private_properties.hpp"
#include "vpux/IMD/device.hpp"
#include "vpux/IMD/parsed_properties.hpp"
#include "vpux/utils/core/error.hpp"

#include "device_helpers.hpp"

namespace intel_npu {

const std::shared_ptr<IDevice> IMDBackend::getDevice() const {
    return std::make_shared<IMDDevice>(ov::intel_npu::Platform::NPU3720);
}

const std::shared_ptr<IDevice> IMDBackend::getDevice(const std::string& name) const {
    const auto platform = utils::getPlatformByDeviceName(name);
    return std::make_shared<IMDDevice>(platform);
}

const std::shared_ptr<IDevice> IMDBackend::getDevice(const ov::AnyMap& params) const {
    const auto it = params.find(ov::device::id.name());
    VPUX_THROW_WHEN(it == params.end(), "DEVICE_ID parameter was not provided");
    return getDevice(it->second.as<std::string>());
}

const std::vector<std::string> IMDBackend::getDeviceNames() const {
    if (const auto* name = std::getenv("IE_NPU_DEVICE_ID")) {
        return {name};
    }

    VPUX_THROW("Unknown device. Please set IE_NPU_DEVICE_ID!");
    return std::vector<std::string>();
}

const std::string IMDBackend::getName() const {
    return "IMD";
}

void IMDBackend::registerOptions(OptionsDesc& options) const {
    options.add<MV_TOOLS_PATH>();
    options.add<VPU4_SIMICS_DIR>();
    options.add<LAUNCH_MODE>();
    options.add<MV_RUN_TIMEOUT>();
}

bool IMDBackend::isBatchingSupported() const {
    return false;
}

bool IMDBackend::isCommandQueueExtSupported() const {
    return false;
}

OPENVINO_PLUGIN_API void CreateNPUEngineBackend(std::shared_ptr<IEngineBackend>& obj, const Config&) {
    obj = std::make_shared<IMDBackend>();
}

}  // namespace intel_npu

//
// Copyright 2020 Intel Corporation.
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

#include "zero_device.h"

#include "zero_allocator.h"
#include "zero_executor.h"

using namespace vpux;

std::shared_ptr<Allocator> ZeroDevice::getAllocator() const {
    std::shared_ptr<Allocator> result = std::make_shared<ZeroAllocator>(_driver_handle);
    return result;
}

std::shared_ptr<Executor> ZeroDevice::createExecutor(
    const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) {
    _config.parseFrom(config);
    std::shared_ptr<Executor> result;

    if (_config.ze_syncType() == InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE) {
        result = std::make_shared<ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>>(
                _driver_handle, _device_handle, _context, _graph_ddi_table_ext, _fence_ddi_table_ext,
                networkDescription, _config);
    } else {
        result = std::make_shared<ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>>(
                _driver_handle, _device_handle, _context, _graph_ddi_table_ext, _fence_ddi_table_ext,
                networkDescription, _config);
    }

    return result;
}

std::string ZeroDevice::getName() const { return std::string("VPU-0"); }

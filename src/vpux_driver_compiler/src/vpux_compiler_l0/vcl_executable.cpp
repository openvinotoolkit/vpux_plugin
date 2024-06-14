//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vcl_executable.hpp"

using namespace vpux;

namespace VPUXDriverCompiler {
VPUXExecutableL0::VPUXExecutableL0(const std::shared_ptr<const NetworkDescription>& networkDesc, bool enableProfiling,
                                   VCLLogger* vclLogger)
        : _networkDesc(networkDesc), enableProfiling(enableProfiling), _logger(vclLogger) {
}

vcl_result_t VPUXExecutableL0::serializeNetwork() {
    StopWatch stopWatch;
    if (enableProfiling) {
        stopWatch.start();
    }

    if (enableProfiling) {
        stopWatch.stop();
        _logger->info("getCompiledNetwork time: {0} ms", stopWatch.delta_ms());
    }
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXExecutableL0::getNetworkSize(uint64_t* blobSize) const {
    if (blobSize == nullptr) {
        _logger->outputError("Can not return blob size for NULL argument!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    const auto& blob = _networkDesc->compiledNetwork;
    *blobSize = blob.size();
    if (*blobSize == 0) {
        // The executable handle do not contain a legal network.
        _logger->outputError("No blob created! The compiled network is empty!");
        return VCL_RESULT_ERROR_UNKNOWN;
    } else {
        return VCL_RESULT_SUCCESS;
    }
}

vcl_result_t VPUXExecutableL0::exportNetwork(uint8_t* blobOut, uint64_t blobSize) const {
    const auto& blob = _networkDesc->compiledNetwork;
    if (!blobOut || blobSize != blob.size()) {
        _logger->outputError("Invalid argument to export network");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    StopWatch stopWatch;
    if (enableProfiling)
        stopWatch.start();

    memcpy(blobOut, blob.data(), blobSize);

    if (enableProfiling) {
        stopWatch.stop();
        _logger->info("exportNetwork time: {0} ms", stopWatch.delta_ms());
    }
    return VCL_RESULT_SUCCESS;
}

}  // namespace VPUXDriverCompiler

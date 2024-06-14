//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vcl_profiling.hpp"

#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

using namespace vpux;

namespace VPUXDriverCompiler {

vcl_result_t VPUXProfilingL0::getTaskInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        _logger->outputError("Null argument to get task info");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (_taskInfo.empty()) {
        try {
            _taskInfo = profiling::getTaskInfo(_blobData, _blobSize, _profData, _profSize,
                                               profiling::VerbosityLevel::HIGH, false);
        } catch (const std::exception& error) {
            _logger->outputError(error.what());
            return VCL_RESULT_ERROR_UNKNOWN;
        } catch (...) {
            _logger->outputError("Internal exception! Can't parse profiling information.");
            return VCL_RESULT_ERROR_UNKNOWN;
        }
    }

    profOutput->data = reinterpret_cast<uint8_t*>(_taskInfo.data());
    profOutput->size = _taskInfo.size() * sizeof(profiling::TaskInfo);
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXProfilingL0::getLayerInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        _logger->outputError("Null argument to get layer info");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (_layerInfo.empty()) {
        try {
            _layerInfo = profiling::getLayerProfilingInfoHook(_profData, _profSize, _blobData, _blobSize);
        } catch (const std::exception& error) {
            _logger->outputError(error.what());
            return VCL_RESULT_ERROR_UNKNOWN;
        } catch (...) {
            _logger->outputError("Internal exception! Can't parse profiling information.");
            return VCL_RESULT_ERROR_UNKNOWN;
        }
    }

    profOutput->data = reinterpret_cast<uint8_t*>(_layerInfo.data());
    profOutput->size = _layerInfo.size() * sizeof(profiling::LayerInfo);
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXProfilingL0::getRawInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        _logger->outputError("Null argument to get raw info");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    profOutput->data = _profData;
    profOutput->size = _profSize;
    return VCL_RESULT_SUCCESS;
}

vcl_profiling_properties_t VPUXProfilingL0::getProperties() const {
    vcl_profiling_properties_t prop;
    prop.version.major = VCL_PROFILING_VERSION_MAJOR;
    prop.version.minor = VCL_PROFILING_VERSION_MINOR;
    return prop;
}

}  // namespace VPUXDriverCompiler

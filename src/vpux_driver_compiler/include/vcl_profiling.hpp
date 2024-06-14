//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

/**
 * @file vcl_profiling.hpp
 * @brief Define VPUXProfilingL0 which parses profiling data
 */

#pragma once

#include "vcl_common.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

namespace VPUXDriverCompiler {

/**
 * @brief Parse the profiling output with blob.
 *
 * Check @ref how-to-use-profiling.md about how to collect the data
 */
class VPUXProfilingL0 final {
public:
    /**
     * @brief Construct a new VPUXProfilingL0 object
     *
     * @param profInput Include the blob and correspond profiling output
     * @param vclLogger
     */
    VPUXProfilingL0(p_vcl_profiling_input_t profInput, VCLLogger* vclLogger)
            : _blobData(profInput->blobData),
              _blobSize(profInput->blobSize),
              _profData(profInput->profData),
              _profSize(profInput->profSize),
              _logger(vclLogger) {
    }

    vcl_result_t getTaskInfo(p_vcl_profiling_output_t profOutput);
    vcl_result_t getLayerInfo(p_vcl_profiling_output_t profOutput);
    vcl_result_t getRawInfo(p_vcl_profiling_output_t profOutput);
    vcl_profiling_properties_t getProperties() const;
    VCLLogger* getLogger() const {
        return _logger;
    }

private:
    const uint8_t* _blobData;  ///< Pointer to the buffer with the blob
    uint64_t _blobSize;        ///< Size of the blob in bytes
    const uint8_t* _profData;  ///< Pointer to the raw profiling output
    uint64_t _profSize;        ///< Size of the raw profiling output

    std::vector<vpux::profiling::TaskInfo> _taskInfo;    ///< Per-task (DPU, DMA, SW) profiling info
    std::vector<vpux::profiling::LayerInfo> _layerInfo;  ///< Per-layer profiling info
    VCLLogger* _logger;                                  ///< Internal logger
};

}  // namespace VPUXDriverCompiler

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

/**
 * @file vcl_executable.hpp
 * @brief Define VPUXExecutableL0 which holds compiled blob
 */

#pragma once

#include "vcl_common.hpp"

using intel_npu::NetworkDescription;

namespace VPUXDriverCompiler {

/**
 * @brief Hold the compiled blob
 *
 */
class VPUXExecutableL0 final {
public:
    VPUXExecutableL0(const std::shared_ptr<const NetworkDescription>& networkDesc, bool enableProfiling,
                     VCLLogger* vclLogger);
    /**
     * @brief Get compiled blob from net description
     *
     * @return vcl_result_t
     */
    vcl_result_t serializeNetwork();

    /**
     * @brief Get the size of blob
     *
     * @param blobSize Store the size of blob if execute successfully
     * @return vcl_result_t
     */
    vcl_result_t getNetworkSize(uint64_t* blobSize) const;

    /**
     * @brief Copy blob data to user buffer
     *
     * @param blob Point to the buffer created by user
     * @param blobSize Need to be same with the result of getNetworkSize()
     * @return vcl_result_t
     */
    vcl_result_t exportNetwork(uint8_t* blob, uint64_t blobSize) const;

    VCLLogger* getLogger() const {
        return _logger;
    }

private:
    std::shared_ptr<const NetworkDescription> _networkDesc;  ///< The compilation result of MLIR compiler
    bool enableProfiling;                                    ///< Calc time cost on VCL level
    VCLLogger* _logger;
};

}  // namespace VPUXDriverCompiler

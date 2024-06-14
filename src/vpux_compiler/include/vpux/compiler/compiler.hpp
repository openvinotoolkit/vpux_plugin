//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "intel_npu/al/icompiler.hpp"

namespace vpux {

class CompilerImpl final : public intel_npu::ICompiler {
public:
    uint32_t getSupportedOpsetVersion() const final;

    // Mutable model variant for direct use with deserialized model in VCL
    intel_npu::NetworkDescription compile(const std::shared_ptr<ov::Model>& model,
                                          const intel_npu::Config& config) const;

    intel_npu::NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                                          const intel_npu::Config& config) const final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                              const intel_npu::Config& config) const final;

    intel_npu::NetworkMetadata parse(const std::vector<uint8_t>& network, const intel_npu::Config& config) const final;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const intel_npu::Config& config) const final;
};

/**
 * @enum IRPrintingOrder
 * @brief VPUX IR pass printing before/after or before and after
 */
enum class IRPrintingOrder {
    BEFORE,
    AFTER,
    BEFORE_AFTER,
};

bool isELFEnabled(const intel_npu::Config& configuration);

}  // namespace vpux

//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

namespace vpux::VPU {
/* @brief
 * Static class for generating MPEEngine attributes.
 */
class MPEEngineConfig {
public:
    static MPEEngineAttr retrieveMPEEngineAttribute(mlir::Operation* operation, VPU::ArchKind arch) {
        if (arch == VPU::ArchKind::NPU37XX || arch == VPU::ArchKind::NPU40XX) {
            return MPEEngine37XXAttr::get(operation->getContext(),
                                          MPEEngine37XXModeAttr::get(operation->getContext(), MPEEngine37XXMode::SCL));
        }
        return nullptr;
    }
};

}  // namespace vpux::VPU

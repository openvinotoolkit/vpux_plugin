//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

namespace vpux {
namespace VPU {

//
// ClusterBroadcastingOpModelNCEOp
//

class ClusterBroadcastingOpModelNCEOp final :
        public VPU::ClusterBroadcastingOpInterface::FallbackModel<ClusterBroadcastingOpModelNCEOp> {
public:
    bool isBroadcastCapable(mlir::Operation*, int64_t) const {
        return true;
    }
};

}  // namespace VPU
}  // namespace vpux

//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"

namespace vpux {
namespace VPUIP {
namespace arch37xx {

//
// ClusterSWRewriter
//

class ClusterSWRewriter {
public:
    ClusterSWRewriter(mlir::MLIRContext* ctx, mlir::ModuleOp module, Logger log)
            : _log(log), _ctx(ctx), _module(module) {
    }

    void matchAndRewrite(VPUIP::SwKernelOp swTask, mlir::OpBuilder& builder) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::ModuleOp _module;
};

}  // namespace arch37xx
}  // namespace VPUIP
}  // namespace vpux

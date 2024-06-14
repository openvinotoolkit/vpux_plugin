//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

namespace vpux::VPUIP {

class ICaptureWorkpointStrategy {
public:
    virtual void prepareDMACapture(mlir::OpBuilder& /* builder */, mlir::func::FuncOp& /* func */,
                                   const int64_t /* profOutputId */, mlir::func::ReturnOp /* returnOp */) {
        // if workpoint register is accesible by DMA - need to owerwrite this method
        // otherwise (if handled by FW) - keep this default empty implementation
    }

    virtual ~ICaptureWorkpointStrategy() = default;
};

}  // namespace vpux::VPUIP

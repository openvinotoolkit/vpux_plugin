//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/impl/capture_workpoint_strategy.hpp"

namespace vpux::VPUIP::arch37xx {

class CaptureWorkpointStrategy final : public vpux::VPUIP::ICaptureWorkpointStrategy {
public:
    virtual void prepareDMACapture(mlir::OpBuilder& builder, mlir::func::FuncOp& func, const int64_t profOutputId,
                                   mlir::func::ReturnOp returnOp) final override;
};

}  // namespace vpux::VPUIP::arch37xx

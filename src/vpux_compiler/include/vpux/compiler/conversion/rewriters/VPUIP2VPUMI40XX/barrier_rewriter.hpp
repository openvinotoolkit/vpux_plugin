//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace vpux::vpuip2vpumi40xx {

struct BarrierRewriter : mlir::OpConversionPattern<VPURT::ConfigureBarrierOp> {
    using OpConversionPattern::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(VPURT::ConfigureBarrierOp origOp, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpux::vpuip2vpumi40xx

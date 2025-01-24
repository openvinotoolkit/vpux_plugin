//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::vpuip2vpumi40xx {

struct M2IRewriter : mlir::OpConversionPattern<VPUIP::M2ITaskOp> {
    using OpConversionPattern::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(VPUIP::M2ITaskOp origOp, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpux::vpuip2vpumi40xx

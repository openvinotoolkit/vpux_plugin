//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::vpuip2vpumi40xx {

struct NCEClusterTaskRewriter : mlir::OpConversionPattern<VPUIP::NCEClusterTaskOp> {
    using OpConversionPattern::OpConversionPattern;
    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp origOp, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpux::vpuip2vpumi40xx

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <numeric>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {

struct AdjustConvShapeParams {
    Shape filterShape;     // New constructed filter shape after reshape conv's shape
    Shape inputShape;      // New conv's input shape after adjust
    Shape outputShape;     // New conv's output shape after adjust
    int64_t borrowFactor;  // The borrowed fator from W for C
    int64_t filterPading;  // The padding num for filter after construct new filter
    int64_t padNum;        // The padding num for aligned shape
};

void insertReorderForInput(mlir::Operation* op, mlir::OpOperand& input, DimsOrder dstOrder,
                           mlir::PatternRewriter& rewriter, Logger log);
IE::ReorderOp insertReorderForOutput(mlir::Operation* op, mlir::Value output, DimsOrder dstOrder,
                                     mlir::PatternRewriter& rewriter, Logger log);

void changeDimsOrder(mlir::Value value, DimsOrder newOrder, Logger log);

mlir::FailureOr<AdjustConvShapeParams> getAdjustConvShapeParameters(IE::ConvolutionOp convOp, mlir::Value filter,
                                                                    Shape outputShape, Logger _log);

}  // namespace vpux

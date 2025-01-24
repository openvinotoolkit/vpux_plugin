//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

//
// calcPadsEnd
//

Shape calcPadsEnd(ShapeRef origShape, ShapeRef extendedShape);
Shape calcPadsEnd(vpux::NDTypeInterface origType, int64_t channelAlignment);

bool needsPadding(const int64_t dim);

mlir::Value expandWithOffset(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, IE::SliceOp sliceOp,
                             mlir::Value expandValue, ShapeRef inPadsEnd, size_t expandDim);
mlir::Value paddingChannel(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::Value expandValue,
                           ShapeRef filterPadsEnd, size_t expandDim);
mlir::Value paddingFilter(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::Value expandValue,
                          Shape filterPadsEnd);
SmallVector<int64_t> extractMeaningfulOutput(mlir::Operation* origOp, ShapeRef outPadsEnd);

mlir::Value concatWithZeroConst(mlir::Location loc, mlir::Value filter, ShapeRef subInput, int64_t sliceChannelOffset,
                                mlir::PatternRewriter& rewriter);

mlir::Value padConvFilter(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, const int64_t inChanPadEnd,
                          const int64_t outChanPadEnd, const Logger& log);

bool beneficialToKeepExpand(ShapeRef unExpandedShape, ShapeRef expandedShape, mlir::Operation* op);

// convert expand op to convolution utils
int64_t calculateAlignmentRequirementForExpandOpConversion(const vpux::NDTypeInterface expandInType);
bool beneficialToPadHeight(IE::ExpandOp origOp);
bool beneficialToPadWidth(IE::ExpandOp origOp);
bool isEligibleConvertToConv(IE::ExpandOp expandOp, Logger log, StringRef debugName);
std::optional<vpux::Dim> getExpandAxis(IE::ExpandOp expandOp);
}  // namespace IE
}  // namespace vpux

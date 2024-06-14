//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/cast_utils.hpp"

using namespace vpux;

mlir::Value VPUIP::QuantizeCastOp::getViewSource() {
    return getInput();
}

mlir::OpFoldResult VPUIP::QuantizeCastOp::fold(FoldAdaptor) {
    return getInput().getType() == getOutput().getType() ? getInput() : mlir::TypedValue<mlir::MemRefType>{nullptr};
}

//
// FuseQuantizeCastOps
//

namespace {

class FuseQuantizeCastOps final : public mlir::OpRewritePattern<VPUIP::QuantizeCastOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::QuantizeCastOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseQuantizeCastOps::matchAndRewrite(VPUIP::QuantizeCastOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto producerQuantizeCastOp = origOp.getInput().getDefiningOp<VPUIP::QuantizeCastOp>();
    if (producerQuantizeCastOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::QuantizeCastOp>(origOp, origOp.getType(), producerQuantizeCastOp.getInput());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::QuantizeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseQuantizeCastOps>(ctx);
}

mlir::LogicalResult vpux::VPUIP::QuantizeCastOp::verify() {
    const auto op = getOperation();
    auto distributedInType = getInput().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributedOutType = getOutput().getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedInType && distributedOutType) {
        auto inputDistribution = distributedInType.getDistribution();
        auto outputDistribution = distributedOutType.getDistribution();
        if (inputDistribution != outputDistribution) {
            return errorAt(op, "QuantizeCastOp input and output must have the same distribution attribute");
        }
    }

    auto inputStrides = getStrides(getInput());
    auto outputStrides = getStrides(getOutput());
    if (inputStrides != outputStrides) {
        return errorAt(op, "QuantizeCastOp input and output must have the same strides, but got {0} and {1}",
                       inputStrides, outputStrides);
    }

    auto inputShape = getShape(getInput());
    auto outputShape = getShape(getOutput());
    if (inputShape != outputShape) {
        return errorAt(op, "QuantizeCastOp input and output must have the same shape, but got {0} and {1}", inputShape,
                       outputShape);
    }

    auto inElemType = getInput().getType().cast<NDTypeInterface>().getElementType();
    auto outElemType = getOutput().getType().cast<NDTypeInterface>().getElementType();
    return vpux::isQuantizeCastValid(getLoc(), inElemType, outElemType);
}

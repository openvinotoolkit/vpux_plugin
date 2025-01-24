//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::GenericReshapeOp::verify() {
    const auto op = getOperation();
    auto distributedInType = getInput().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributedOutType = getOutput().getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedInType && distributedOutType) {
        if (!isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(distributedInType,
                                                                                             distributedOutType)) {
            return errorAt(op, "Reshape has incompatible output shape as clustering: in type = {0}, out type = {1}",
                           distributedInType, distributedOutType);
        }
    }

    const auto inType = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outType = getOutput().getType().cast<vpux::NDTypeInterface>();

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "Reshape input and output must have the same number of elements");
    }

    const auto inReqs = StrideReqs::compact(inType.getRank());
    const auto outReqs = StrideReqs::compact(outType.getRank());

    if (!inReqs.checkStrides(inType)) {
        return errorAt(op, "Input strides do not match requirements '{0}'", inType);
    }
    if (!outReqs.checkStrides(outType)) {
        return errorAt(op, "Output strides do not match requirements '{0}'", inType);
    }

    return mlir::success();
}

mlir::Value VPUIP::GenericReshapeOp::getViewSource() {
    return getInput();
}

mlir::OpFoldResult VPUIP::GenericReshapeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return static_cast<Const::ContentAttr>(cst).transform().reshape(getShape(getOutput())).get();
    }

    return nullptr;
}

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<VPUIP::GenericReshapeOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::GenericReshapeOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(VPUIP::GenericReshapeOp origOp,
                                                  mlir::PatternRewriter& rewriter) const {
    auto producerReshapeOp = origOp.getInput().getDefiningOp<VPUIP::GenericReshapeOp>();
    if (producerReshapeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::GenericReshapeOp>(origOp, origOp.getOutput().getType(),
                                                         producerReshapeOp.getInput());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::GenericReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseReshapes>(ctx);
}

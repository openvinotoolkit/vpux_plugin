//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::CopyOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::CopyOpAdaptor copyOp(operands, attrs, prop);
    if (mlir::failed(copyOp.verify(loc))) {
        return mlir::failure();
    }

    const auto ndInType = copyOp.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    if (ndInType == nullptr) {
        return errorAt(loc, "IE::CopyOp operand must have vpux::NDTypeInterface type");
    }

    IndexedSymbolAttr outMemSpace = nullptr;
    if (copyOp.getOutMemSpace().has_value()) {
        outMemSpace = copyOp.getOutMemSpace().value();
    }
    const auto outType = ndInType.changeMemSpace(outMemSpace);

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::CopyOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// FuseCopies
//

namespace {

class FuseCopies final : public mlir::OpRewritePattern<VPU::CopyOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseCopies::matchAndRewrite(VPU::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerCopyOp = origOp.getInput().getDefiningOp<VPU::CopyOp>();
    if (producerCopyOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::CopyOp>(origOp, producerCopyOp.getInput(), origOp.getOutMemSpaceAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::CopyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseCopies>(ctx);
}

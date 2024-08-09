//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"

#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

namespace {
//
// ConvertToPermuteCast
//
class ConvertToPermuteCast final : public mlir::OpRewritePattern<VPU::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<VPU::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToPermuteCast::matchAndRewrite(VPU::MemPermuteOp memPermuteOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto inOrder = DimsOrder::fromValue(memPermuteOp.getInput());
    const auto inShape = getShape(memPermuteOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    if (!isTrivialPermute(inMemShape, memPermuteOp.getMemPerm())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::PermuteCastOp>(memPermuteOp, memPermuteOp.getInput(),
                                                    memPermuteOp.getDstOrderAttr(), memPermuteOp.getMemPermAttr());
    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::VPU::MemPermuteOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MemPermuteOpAdaptor mem_permute(operands, attrs, prop);
    if (mlir::failed(mem_permute.verify(loc))) {
        return mlir::failure();
    }

    VPU::inferPermuteReturnTypes(mem_permute.getInput(), mem_permute.getMemPerm(), mem_permute.getDstOrder(),
                                 inferredReturnTypes);

    return mlir::success();
}

InputTiling vpux::VPU::MemPermuteOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    mlir::AffineMap memPerm = getMemPerm();
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    const auto inShape = getShape(getInput());
    const auto inOrder = DimsOrder::fromValue(getInput());
    const auto outOrder = DimsOrder::fromValue(getOutput());
    auto curTile = outputTile;
    for (auto ind : irange(inShape.size())) {
        // take in consideration input and output shape vector order not map with memory order
        auto idxOrdIn = inOrder.dimAt(perm.dimAt(ind).ind());
        auto idxOrdOut = outOrder.dimAt(ind);
        curTile.shape[idxOrdIn] = outputTile.shape[idxOrdOut];
        curTile.offsets[idxOrdIn] = outputTile.offsets[idxOrdOut];
        curTile.axis[idxOrdIn] = outputTile.axis[idxOrdOut];
    }
    return TilingInfo{curTile};
}

void vpux::VPU::MemPermuteOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::MemPermuteOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// canonicalize
//

void vpux::VPU::MemPermuteOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<ConvertToPermuteCast>(context);
}

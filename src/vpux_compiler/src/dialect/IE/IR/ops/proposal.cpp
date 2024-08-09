//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ProposalOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ProposalOpAdaptor proposal(operands, attrs, prop);
    if (mlir::failed(proposal.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = proposal.getClassProbs().getType().cast<mlir::ShapedType>();

    // out shape must be [batch_size * post_nms_topn, 5]
    const SmallVector<int64_t> outShape{
            inType.getShape().front() * proposal.getProposalAttrs().getPostNmsTopN().getInt(), 5};
    const SmallVector<int64_t> probsShape{inType.getShape().front() *
                                          proposal.getProposalAttrs().getPostNmsTopN().getInt()};
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(probsShape, inType.getElementType());

    return mlir::success();
}

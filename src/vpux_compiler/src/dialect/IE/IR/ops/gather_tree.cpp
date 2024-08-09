//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::GatherTreeOp::verify() {
    const auto stepIdsType = getStepIds().getType().cast<mlir::ShapedType>();
    const auto stepIdsShape = stepIdsType.getShape();

    if (stepIdsType.getRank() != 3) {
        return errorAt(*this, "Wrong GatherTree step_ids rank {0}, step_ids should have rank 3", stepIdsType.getRank());
    }

    const auto parentIdsType = getParentIds().getType().cast<mlir::ShapedType>();
    const auto parentIdsShape = parentIdsType.getShape();

    if (parentIdsType.getRank() != 3) {
        return errorAt(*this, "Wrong GatherTree parent_ids rank {0}, parent_ids should have rank 3",
                       parentIdsType.getRank());
    }

    const auto maxSeqLenType = getMaxSeqLen().getType().cast<mlir::ShapedType>();
    const auto maxSeqLenShape = maxSeqLenType.getShape();

    if (maxSeqLenType.getRank() != 1) {
        return errorAt(*this, "Wrong GatherTree max_seq_len rank {0}, max_seq_len should have rank 1",
                       maxSeqLenType.getRank());
    }

    const auto endTokenType = getEndToken().getType().cast<mlir::ShapedType>();

    if (stepIdsShape != parentIdsShape) {
        return errorAt(*this, "GatherTree step_ids and parent_id got shape mismatch: {0} {1}", stepIdsShape,
                       parentIdsShape);
    }

    if (!(stepIdsShape[1] == parentIdsShape[1] && stepIdsShape[1] == maxSeqLenShape[0])) {
        return errorAt(*this, "GatherTree inputs batch_size mismatch {0} {1} {2}", stepIdsShape[1], parentIdsShape[1],
                       maxSeqLenShape[0]);
    }

    if (!(stepIdsType.getElementType() == parentIdsType.getElementType() &&
          stepIdsType.getElementType() == maxSeqLenType.getElementType() &&
          stepIdsType.getElementType() == endTokenType.getElementType())) {
        return errorAt(*this, "Wrong GatherTree Inputs Element Type {0} {1} {2} {3}", stepIdsType.getElementType(),
                       parentIdsType.getElementType(), maxSeqLenType.getElementType(), endTokenType.getElementType());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::GatherTreeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GatherTreeOpAdaptor gatherTree(operands, attrs, prop);
    if (mlir::failed(gatherTree.verify(loc))) {
        return mlir::failure();
    }

    const auto stepIdsType = gatherTree.getStepIds().getType().cast<mlir::ShapedType>();
    const auto stepIdsShape = stepIdsType.getShape();

    inferredReturnShapes.emplace_back(stepIdsShape, stepIdsType.getElementType());

    return mlir::success();
}

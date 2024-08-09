//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ProposalOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ProposalOpAdaptor proposal(operands, attrs, prop);
    if (mlir::failed(proposal.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = proposal.getClassProbs().getType().cast<vpux::NDTypeInterface>();

    // out shape must be [batch_size * post_nms_topn, 5]
    const SmallVector<int64_t> outShape{
            inType.getShape().front() * proposal.getProposalAttrs().getPostNmsTopN().getInt(), 5};
    const SmallVector<int64_t> probsShape{inType.getShape().front() *
                                          proposal.getProposalAttrs().getPostNmsTopN().getInt()};

    const auto outType = inType.changeShape(Shape(outShape));
    const auto probsType = inType.changeShape(Shape(probsShape));
    inferredReturnTypes.push_back(outType);
    inferredReturnTypes.push_back(probsType);

    return mlir::success();
}

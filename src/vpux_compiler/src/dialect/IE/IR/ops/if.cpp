//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace mlir;

mlir::LogicalResult vpux::IE::IfOp::verify() {
    auto regions = getRegions();
    if (regions.size() != 2) {
        return errorAt(*this, "IfOp should have 2 regions, actual: {0}", regions.size());
    }

    llvm::SmallVector<size_t> results;
    for (Region* region : regions) {
        if (!region->hasOneBlock()) {
            return errorAt(*this, "Region #{0} should have one block, actual: {1}", region->getRegionNumber(),
                           region->getBlocks().size());
        }
        auto& block = region->getBlocks().front();
        if (auto returnOp = dyn_cast<IE::YieldOp>(block.getTerminator())) {
            results.push_back(returnOp.getOperands().size());
        }
    }

    auto ifResultsNumber = getResults().size();
    if (ifResultsNumber == 0) {
        return errorAt(*this, "IfOp should have minimum 1 output, actual: {0}", ifResultsNumber);
    }
    if (results[0] != results[1] && results[0] != ifResultsNumber) {
        return errorAt(*this, "Region Then, Region Else and IfOp should have same number of outputs");
    }

    return mlir::success();
}

LogicalResult vpux::IE::IfOp::inferReturnTypeComponents(MLIRContext* ctx, std::optional<Location> optLoc,
                                                        ValueShapeRange operands, DictionaryAttr attrs,
                                                        OpaqueProperties props, RegionRange regions,
                                                        SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::IfOpAdaptor ifOperator(operands, attrs, props, regions);
    if (mlir::failed(ifOperator.verify(loc))) {
        return mlir::failure();
    }

    llvm::SmallVector<IE::YieldOp> yieldOps;
    mlir::Region* region = regions.front();
    for (auto& block : *region) {
        if (auto returnOp = dyn_cast<IE::YieldOp>(block.getTerminator())) {
            yieldOps.push_back(returnOp);
        }
    }

    if (yieldOps.empty()) {
        return failure();
    }

    for (auto& yield : yieldOps) {
        for (mlir::Value operand : yield.getOperands()) {
            auto inType = operand.getType().cast<RankedTensorType>();
            const auto outDesc = vpux::getTensorAttr(inType);
            inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);
        }
    }
    return success();
}

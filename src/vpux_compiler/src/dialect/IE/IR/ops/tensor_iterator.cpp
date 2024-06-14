//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace mlir;

mlir::LogicalResult vpux::IE::TensorIteratorOp::verify() {
    // Check if num_iterations is valid
    if (!getNumIterations()) {
        return errorAt(*this, "Attribute num_iterations is required by TensorIterator op");
    }

    if (getNumIterations() < 1) {
        return errorAt(*this, "The value of num_iterations for TensorIterator op should be more than 0, actual {0}",
                       getNumIterations());
    }

    return mlir::success();
}

LogicalResult vpux::IE::TensorIteratorOp::inferReturnTypeComponents(
        MLIRContext* ctx, std::optional<Location> optLoc, ValueShapeRange operands, DictionaryAttr attrs,
        OpaqueProperties props, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::TensorIteratorOpAdaptor tensorIteratorOperator(operands, attrs, props, regions);

    if (mlir::failed(tensorIteratorOperator.verify(loc))) {
        return mlir::failure();
    }

    // Get the output size of body module
    mlir::Region* region = regions.front();
    auto& block = region->getBlocks().front();
    llvm::SmallVector<IE::LoopTerminatorOp> loopTerminatorOps;
    if (auto returnOp = dyn_cast<IE::LoopTerminatorOp>(block.getTerminator())) {
        loopTerminatorOps.push_back(returnOp);
    }

    if (loopTerminatorOps.empty()) {
        return failure();
    }

    // collecting concating along axis cases
    auto numIterations = tensorIteratorOperator.getNumIterations();
    auto concatOutputDescAttrVector =
            parseCustomAttrArray<IE::ConcatOutputPortMapAttr>(tensorIteratorOperator.getConcatOutputDescsAttr());
    auto invariantOutputDescAttrVector =
            parseCustomAttrArray<IE::InvariantOutputPortMapAttr>(tensorIteratorOperator.getInvariantOutputDescsAttr());
    mlir::DenseMap<int64_t, int64_t> toConcatOutputAxisMap;
    mlir::DenseMap<int64_t, int64_t> toConcatOutputPordIdMap;
    mlir::DenseMap<int64_t, int64_t> invariantOutputPordIdMap;
    for (auto copm : concatOutputDescAttrVector) {
        auto exId = copm.getExternalPortId().getValue().getSExtValue();
        auto inId = copm.getInternalLayerId().getValue().getSExtValue();
        auto axis = copm.getAxis().getValue().getSExtValue();
        toConcatOutputAxisMap.insert({inId, axis});
        toConcatOutputPordIdMap.insert({exId, inId});
    }
    for (auto iopm : invariantOutputDescAttrVector) {
        auto exId = iopm.getExternalPortId().getValue().getSExtValue();
        auto inId = iopm.getInternalLayerId().getValue().getSExtValue();
        invariantOutputPordIdMap.insert({exId, inId});
    }

    // Infer return types
    for (auto& loopTerminator : loopTerminatorOps) {
        for (auto exId : irange(loopTerminator.getOperands().size())) {
            int64_t inId = toConcatOutputPordIdMap.count(exId) ? toConcatOutputPordIdMap[exId]
                                                               : invariantOutputPordIdMap[exId];
            mlir::Value operand = loopTerminator.getOperands()[inId];

            auto inType = operand.getType().cast<RankedTensorType>();
            const auto outDesc = vpux::getTensorAttr(inType);

            // deal with concating along axis cases
            const auto inputShape = inType.getShape();
            SmallVector<int64_t> outShape;
            for (size_t i = 0; i < inputShape.size(); ++i) {
                outShape.push_back(inputShape[i]);
            }

            if (toConcatOutputPordIdMap.count(exId)) {
                int64_t axis = toConcatOutputAxisMap[inId] < 0 ? toConcatOutputAxisMap[inId] + inputShape.size()
                                                               : toConcatOutputAxisMap[inId];
                VPUX_THROW_UNLESS(axis >= 0, "Wrong axis `{0}`, out of range [-Rank , Rank - 1]",
                                  toConcatOutputAxisMap[inId]);
                outShape[axis] = numIterations * outShape[axis];
            }

            inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);
        }
    }
    return success();
}

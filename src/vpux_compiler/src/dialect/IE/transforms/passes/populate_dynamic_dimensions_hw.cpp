//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reify_shape.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

using namespace vpux;

namespace {

// Adapter for mlir::bufferization::populateDynamicDimSizes
// StridedSlice is required to crop the output (which will eventually become static) to dynamic sizes.
// Since populateDynamicDimSizes returns only dynamic dimensions, the pass needs to concatenate them
// with static dimensions and then provide the result to StridedSlice.
// StridedSlice infers its output as tensor<?x?x?x?xf16> when any of begins, ends or strides are
// unknown at compile time. DynamicReshape eliminates the discrepancy between the output of
// StridedSlice and the input of mlir.return (which is not necessarily set to tensor<?x?x?x?xf16>)
void populateDynamicOperand(mlir::Operation* op, const unsigned operandIdx) {
    mlir::Value operand{op->getOperand(operandIdx)};
    if (mlir::isa<mlir::BlockArgument>(operand)) {
        return;
    }
    const auto operandShape = getShape(operand);
    if (operandShape.isStatic()) {
        return;
    }
    auto producer{operand.getDefiningOp()};
    if (!mlir::isa<mlir::ReifyRankedShapedTypeOpInterface>(producer)) {
        return;
    }
    SmallVector<mlir::Value> dynamicOperands{};
    mlir::OpBuilder builder(op);
    mlir::bufferization::populateDynamicDimSizes(builder, producer->getLoc(), operand, dynamicOperands);
    auto newShapeValue = buildConcat(producer->getLoc(), builder, getShape(producer->getResult(0)), dynamicOperands);
    auto newResult = repackDynamicTensor(builder, producer, operandShape, newShapeValue);

    op->setOperand(operandIdx, newResult);
}

void populateDynamicSizes(mlir::Operation* op) {
    if (!IE::needsStaticShape(op)) {
        return;
    }
    mlir::Value output = op->getResult(0);
    if (!output.hasOneUse()) {
        return;
    }
    mlir::Operation* consumer = *output.getUsers().begin();
    // Skip Convolution -> Add, MaxPool -> ReLU and other combinations.
    // Populate dynamic dimensions only when a consumer is not from the list.
    // For example: ReLU -> Reshape is a good candidate to become ReLU -> StridedSlice -> Reshape
    // In this example reshape kernel can handle dynamic shapes properly.
    // Convolution, MaxPool, Add and ReLU cannot.
    if (IE::needsStaticShape(consumer)) {
        return;
    }
    for (const unsigned idx : irange(consumer->getNumOperands())) {
        populateDynamicOperand(consumer, idx);
    }
}
}  // namespace

namespace {
class PopulateDynamicDimensionsHWPass final :
        public IE::PopulateDynamicDimensionsHWBase<PopulateDynamicDimensionsHWPass> {
public:
    explicit PopulateDynamicDimensionsHWPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PopulateDynamicDimensionsHWPass::safeRunOnFunc() {
    auto func = getOperation();
    func->walk(populateDynamicSizes);
}
};  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPopulateDynamicDimensionsHWPass(Logger log) {
    return std::make_unique<PopulateDynamicDimensionsHWPass>(log);
}

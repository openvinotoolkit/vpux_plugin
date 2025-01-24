//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {
void padInput(mlir::Operation* firstOp) {
    mlir::OpBuilder builder{firstOp};
    auto expand = builder.create<IE::DynamicExpandOp>(appendLoc(firstOp->getLoc(), "expand"), firstOp->getOperand(0));
    firstOp->setOperand(0, expand.getOutput());
}

SmallVector<mlir::Operation*> getDynamicOperations(mlir::Operation* op) {
    mlir::Operation* next = op;
    SmallVector<mlir::Operation*> dynamicOps;
    while (IE::needsStaticShape(next)) {
        auto bounds = getBounds(next->getResult(0));
        if (bounds == nullptr) {
            return {};
        }
        dynamicOps.push_back(next);
        // Only data operand (operand 0) must be dynamic. Other operands must be static.
        // FIXME generalize this approach to cover any combination of static and dynamic operands.
        if (getShape(next->getOperand(0)).isStatic()) {
            return {};
        }
        for (unsigned idx = 1; idx < next->getNumOperands(); idx++) {
            if (getShape(next->getOperand(idx)).isDynamic()) {
                return {};
            }
        }
        next = next->getOperand(0).getDefiningOp();
    }
    return dynamicOps;
}

void freezeOutputShape(mlir::Operation* op) {
    auto origType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());
    auto bounds = getBounds(op->getResult(0));
    const auto newShape = parseIntArrayAttr<int64_t>(bounds);
    const auto newType = mlir::RankedTensorType::get(newShape, origType.getElementType());
    op->getResult(0).setType(newType);
}

void traverseDynamicSubgraph(IE::DynamicReshapeOp dynReshape) {
    mlir::Value dynReshapeInput{dynReshape.getInput()};
    if (mlir::isa<mlir::BlockArgument>(dynReshapeInput)) {
        return;
    }
    auto slice = dynReshapeInput.getDefiningOp<IE::StridedSliceOp>();
    if (slice == nullptr) {
        return;
    }
    auto producer = slice.getInput().getDefiningOp();
    const auto dynamicOps = getDynamicOperations(producer);
    if (dynamicOps.empty()) {
        return;
    }
    std::for_each(dynamicOps.begin(), dynamicOps.end(), freezeOutputShape);

    mlir::Operation* firstOp = dynamicOps.back();
    padInput(firstOp);
}

class PadDynamicInputsPass final : public IE::PadDynamicInputsBase<PadDynamicInputsPass> {
public:
    explicit PadDynamicInputsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PadDynamicInputsPass::safeRunOnFunc() {
    getOperation()->walk(traverseDynamicSubgraph);
}
};  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPadDynamicInputsPass(Logger log) {
    return std::make_unique<PadDynamicInputsPass>(log);
}

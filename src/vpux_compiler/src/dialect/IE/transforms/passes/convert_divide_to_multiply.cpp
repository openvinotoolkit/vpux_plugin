//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

class ConvertDivideToMultiplyPass final : public IE::ConvertDivideToMultiplyBase<ConvertDivideToMultiplyPass> {
public:
    explicit ConvertDivideToMultiplyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDivideToMultiplyPass::safeRunOnFunc() {
    static_assert(IE::DivideOp::hasTrait<IE::EltwiseOp>(),
                  "This pass cannot replace IE.Divide with IE.Multiply when division is not element-wise: the "
                  "reciprocal must be calculated differently");

    // converts const 'value' to (1 / 'value') in IR
    const auto applyScalarMultInverse = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value) {
        auto cstOp = value.getDefiningOp<Const::DeclareOp>();
        const auto newCstAttr = cstOp.getContentAttr().scalarMultInverse();
        auto newCstOp = builder.create<Const::DeclareOp>(loc, newCstAttr.getType(), newCstAttr);
        return newCstOp.getOutput();
    };

    auto func = getOperation();
    func->walk([&](IE::DivideOp divideOp) {
        _log.trace("Found IE.Divide at {0}", divideOp->getLoc());
        // Note: assume element types of all operands are equal
        const auto elementType = divideOp.getOutput().getType().getElementType();
        const bool floatDivision = mlir::isa<mlir::FloatType>(elementType);
        if (!floatDivision) {
            _log.trace("Ignore: IE.Divide is not a floating-point division");
            return;
        }

        const auto constOp = divideOp.getInput2().getDefiningOp();
        const bool constDivisor = mlir::isa_and_nonnull<Const::DeclareOp>(constOp);
        if (!constDivisor) {
            _log.trace("Ignore: IE.Divide has no constant divisor");
            return;
        }

        _log.trace("Transforming IE.Divide to IE.Multiply");
        mlir::OpBuilder builder(constOp);
        const auto newInput2 = applyScalarMultInverse(builder, constOp->getLoc(), divideOp.getInput2());
        builder.setInsertionPoint(divideOp);
        const auto multiplyOp = builder.create<IE::MultiplyOp>(appendLoc(divideOp->getLoc(), "convert_to_multiply"),
                                                               divideOp.getInput1(), newInput2,
                                                               divideOp.getAutoBroadcastAttr(), nullptr, nullptr);

        divideOp->replaceAllUsesWith(multiplyOp);
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDivideToMultiplyPass(Logger log) {
    return std::make_unique<ConvertDivideToMultiplyPass>(log);
}

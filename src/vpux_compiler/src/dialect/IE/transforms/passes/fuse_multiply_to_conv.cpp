//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

#include <tuple>

using namespace vpux;

namespace {

// Returns all axes in which the shapes differ.
SmallVector<int64_t> deduceDiscrepancyAxes(ShapeRef inShape, ShapeRef outShape) {
    SmallVector<int64_t> res{};
    for (size_t idx = 0; idx < inShape.size(); idx++) {
        if (inShape[Dim(idx)] != outShape[Dim(idx)]) {
            res.push_back(checked_cast<int64_t>(idx));
        }
    }

    return res;
}

// Returns an "origin" operation of the specified type (one of Ops), ignoring
// pure view ops, for the given operation. This procedure assumes IR in question
// is a chain of single-use operations.
template <typename... Ops>
mlir::Operation* findOriginOp(const Logger& log, mlir::Operation* current) {
    // skip "pure view ops"
    while (current && IE::isPureViewOp(current)) {
        // Note: for the sake of this pass, only single-use op chains are
        // considered
        auto operands = current->getOperands();
        if (operands.size() != 1) {
            log.trace("{0} at {1} has unexpected number of operands {2}, expected 1", current->getName(),
                      current->getLoc(), operands.size());
            return nullptr;
        }

        current = operands[0].getDefiningOp();
    }

    if (mlir::isa_and_nonnull<Ops...>(current)) {
        return current;
    }
    return nullptr;
}

// Returns defining op of the specified type for some operand of the op.
template <typename Op>
Op findDefiningOp(mlir::Operation* op) {
    auto it = llvm::find_if(op->getOperands(), [](const mlir::Value& operand) {
        return mlir::isa_and_nonnull<Op>(operand.getDefiningOp());
    });
    return it == op->getOperands().end() ? nullptr : mlir::cast<Op>((*it).getDefiningOp());
}

// E#122893: consider moving this rewriter into
// InsertReorderBetweenLayerAndConcat pass
struct InsertMultiplyBeforeConcat : public mlir::OpRewritePattern<IE::MultiplyOp> {
    InsertMultiplyBeforeConcat(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitHigh), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

struct FuseMultiplyToConvolution : public mlir::OpRewritePattern<IE::MultiplyOp> {
    FuseMultiplyToConvolution(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitLow), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

mlir::LogicalResult InsertMultiplyBeforeConcat::matchAndRewrite(IE::MultiplyOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE.Multiply at {0}", origOp->getLoc());
    if (origOp.getPostOpAttr() != nullptr) {
        _log.trace("Ignore: IE.Multiply is not simple (has ppe)");
        return mlir::failure();
    }
    auto constOp = findDefiningOp<Const::DeclareOp>(origOp);
    if (constOp == nullptr || !constOp.getContentAttr().getBaseContent().isSplat()) {
        _log.trace("Ignore: IE.Multiply is not constant scalar multiplication");
        return mlir::failure();
    }
    auto concatOp = findDefiningOp<IE::ConcatOp>(origOp);
    if (concatOp == nullptr) {
        _log.trace("Ignore: IE.Multiply does not have preceding IE.Concat");
        return mlir::failure();
    }
    // Note: only put Multiply "inside" concat block when it is feasible
    for (auto [index, operand] : llvm::enumerate(concatOp->getOperands())) {
        // FIXME: so far it is not clear whether *each* concat branch should
        // have a convolution (so we could actually fuse multiply inside) or
        // just *some* branches is enough
        if (findOriginOp<IE::ConvolutionOp>(_log, operand.getDefiningOp()) == nullptr) {
            _log.trace("Ignore: IE.Multiply does not have preceding IE.Convolution in concat branch #{0} - no "
                       "possibility to fuse",
                       index);
            return mlir::failure();
        }
    }

    const auto concatInputShape = getShape(concatOp.getInputs().front());
    const auto axes = deduceDiscrepancyAxes(concatInputShape, getShape(origOp.getOutput()));
    if (axes.size() != 1) {
        _log.trace("Ignore: Concat operation has {0} dimensions of concatenation, expected 1", axes.size());
        return mlir::failure();
    }

    _log.trace("Reordering IE.Multiply at {0} and IE.Concat at {1}", origOp->getLoc(), concatOp->getLoc());
    const auto axis = axes[0];

    // slices the constant into a shape suitable for use inside of concat block
    const auto applySlice = [&](mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp cstOp,
                                int64_t axisOffset, int64_t axisShape) {
        const auto cstType = mlir::cast<NDTypeInterface>(cstOp.getOutput().getType());
        if (cstType.getNumElements() == 1) {
            return cstOp.getOutput();
        }

        // FIXME: perhaps we could just fold the constant once here and then
        // re-use the same -- so far we only work with splats
        auto offset = SmallVector<int64_t>(concatInputShape.size(), 0);
        offset[axis] = axisOffset;
        auto shape = SmallVector<int64_t>(concatInputShape.raw());
        shape[axis] = axisShape;

        const auto newCstAttr = cstOp.getContentAttr().subview(ShapeRef(offset), ShapeRef(shape));
        auto newCstOp = builder.create<Const::DeclareOp>(loc, newCstAttr.getType(), newCstAttr);
        return newCstOp.getOutput();
    };

    int64_t offset = 0;
    SmallVector<mlir::Value> newMultiplyResults;
    for (const auto& concatInput : concatOp.getInputs()) {
        // slice constant using subview
        const auto axisShape = getShape(concatInput)[Dim(axis)];
        const auto newConstResult = applySlice(rewriter, constOp->getLoc(), constOp, offset, axisShape);
        offset += axisShape;

        // create IE.Multiply with the new constant
        static_assert(IE::MultiplyOp::hasTrait<mlir::OpTrait::IsCommutative>(),
                      "The order of operands does not matter for IE.Multiply.");
        auto newMultiply = rewriter.create<IE::MultiplyOp>(concatOp->getLoc(), concatInput, newConstResult,
                                                           origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
                                                           origOp.getClampAttr());
        newMultiplyResults.push_back(newMultiply.getOutput());
    }

    auto newConcatOp =
            rewriter.create<IE::ConcatOp>(concatOp->getLoc(), concatOp.getOutput().getType(), newMultiplyResults,
                                          concatOp.getPerAxisAttr(), concatOp.getStaticOffsetsAttr());

    rewriter.replaceOp(concatOp, newConcatOp);
    rewriter.replaceAllUsesWith(origOp.getOutput(), newConcatOp.getOutput());

    return mlir::success();
}

mlir::LogicalResult FuseMultiplyToConvolution::matchAndRewrite(IE::MultiplyOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE.Multiply at {0}", origOp->getLoc());
    if (origOp.getPostOpAttr() != nullptr) {
        _log.trace("Ignore: IE.Multiply is not simple (has ppe)");
        return mlir::failure();
    }
    auto constOp = findDefiningOp<Const::DeclareOp>(origOp);
    if (constOp == nullptr || !constOp.getContentAttr().getBaseContent().isSplat()) {
        _log.trace("Ignore: IE.Multiply is not constant scalar multiplication");
        return mlir::failure();
    }
    const auto nonConstMultiplyOperand =
            origOp.getInput1().getDefiningOp() == constOp ? origOp.getInput2() : origOp.getInput1();
    auto maybeConvOp = findOriginOp<IE::ConvolutionOp>(_log, nonConstMultiplyOperand.getDefiningOp());
    if (maybeConvOp == nullptr) {
        _log.trace("Ignore: IE.Multiply has no preceding convolution - cannot fuse");
        return mlir::failure();
    }
    auto convOp = mlir::cast<IE::ConvolutionOp>(maybeConvOp);

    _log.trace("Fusing IE.Multiply at {0} into convolution at {1}", origOp->getLoc(), convOp->getLoc());
    // fuse IE.Multiply by specifying the static scale attribute
    const auto originalScale = convOp.getStaticScaleAttr() ? convOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
    const auto newScale = originalScale * constOp.getContent().getSplatValue<double>();
    convOp.setStaticScaleAttr(mlir::FloatAttr::get(mlir::Float32Type::get(origOp.getContext()), newScale));

    rewriter.replaceAllUsesWith(origOp.getOutput(), nonConstMultiplyOperand);

    return mlir::success();
}

class FuseMultiplyToConvPass final : public IE::FuseMultiplyToConvBase<FuseMultiplyToConvPass> {
public:
    explicit FuseMultiplyToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseMultiplyToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InsertMultiplyBeforeConcat>(&ctx, _log);
    patterns.add<FuseMultiplyToConvolution>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseMultiplyToConvPass(Logger log) {
    return std::make_unique<FuseMultiplyToConvPass>(log);
}

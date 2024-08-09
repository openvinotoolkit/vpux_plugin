//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

#include <tuple>

using namespace vpux;

namespace {

// Returns an "origin" operation of the specified type (one of Ops), ignoring
// pure view ops, for the given operation. This procedure assumes IR in question
// is a chain of single-use operations.
template <typename Op>
Op findOriginOp(const Logger& log, mlir::Operation* current) {
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

    return mlir::dyn_cast_or_null<Op>(current);
}

// Returns defining op of the specified type for some operand of the op.
template <typename Op>
Op findDefiningOp(mlir::Operation* op) {
    auto it = llvm::find_if(op->getOperands(), [](const mlir::Value& operand) {
        return mlir::isa_and_nonnull<Op>(operand.getDefiningOp());
    });
    return it == op->getOperands().end() ? nullptr : mlir::cast<Op>((*it).getDefiningOp());
}

bool isSuitableConstant(Const::DeclareOp op) {
    return op && op.getContentAttr().isSplat() &&
           mlir::isa<mlir::FloatType>(op.getContentAttr().getType().getElementType());
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
    if (!isSuitableConstant(constOp)) {
        _log.trace("Ignore: IE.Multiply is not floating-point constant scalar multiplication");
        return mlir::failure();
    }
    auto concatOp = findDefiningOp<IE::ConcatOp>(origOp);
    if (concatOp == nullptr) {
        _log.trace("Ignore: IE.Multiply does not have preceding IE.Concat");
        return mlir::failure();
    }
    // Note: only put IE.Multiply "inside" concat block when it is feasible:
    // *each* concat branch should have a convolution (so we could actually fuse
    // multiply in all the cases)
    for (auto [index, operand] : llvm::enumerate(concatOp->getOperands())) {
        if (findOriginOp<IE::ConvolutionOp>(_log, operand.getDefiningOp()) == nullptr) {
            _log.trace("Ignore: IE.Multiply does not have preceding IE.Convolution in concat branch #{0} - no "
                       "possibility to fuse",
                       index);
            return mlir::failure();
        }
    }

    _log.trace("Reordering IE.Multiply at {0} and IE.Concat at {1}", origOp->getLoc(), concatOp->getLoc());
    // create a new constant suitable for usage inside concat blocks
    const auto singleConstResult = [&]() {
        VPUX_THROW_WHEN(!constOp.getContentAttr().isSplat(), "Expected splat constant in IE.Multiply");

        const auto concatInputShape = getShape(concatOp.getInputs().front());
        const SmallVector<int64_t> newShape(concatInputShape.size(), 1);  // 1x1x1x1

        // Note: in order to preserve the original constant's splat value,
        // always create an FP32 storage and then convert the resulting content
        // attr to a suitable type (either FP32 or FP16). this way, the actual
        // splat value is preserved (FP32 -> FP32) or upcasted (FP16 -> FP32)
        // internally.
        const auto fp32TensorType = mlir::RankedTensorType::get(newShape, mlir::Float32Type::get(getContext()));
        const auto newConst = Const::createFloatConst(rewriter, appendLoc(constOp->getLoc(), "_to_mult_scalar"),
                                                      fp32TensorType, constOp.getContent().getSplatValue<float>());
        const auto actualElemType = mlir::cast<NDTypeInterface>(constOp.getType()).getElementType();
        const bool floatConstantHasCorrectType = (actualElemType == fp32TensorType.getElementType());
        if (floatConstantHasCorrectType) {
            return newConst;
        }
        const auto actualType = mlir::RankedTensorType::get(newShape, actualElemType);
        const auto patchedContentAttr =
                newConst.getDefiningOp<Const::DeclareOp>().getContentAttr().convertElemType(actualElemType);
        return rewriter.create<Const::DeclareOp>(newConst.getLoc(), actualType, patchedContentAttr).getOutput();
    }();

    SmallVector<mlir::Value> newMultiplyResults;
    for (const auto& concatInput : concatOp.getInputs()) {
        // create IE.Multiply with the new constant
        static_assert(IE::MultiplyOp::hasTrait<mlir::OpTrait::IsCommutative>(),
                      "The order of operands does not matter for IE.Multiply.");
        // Note: since we use 1x1x1x1 constant, the broadcast type has to be
        // overwritten just in case
        const auto numpyBroadcast = IE::AutoBroadcastTypeAttr::get(origOp.getContext(), IE::AutoBroadcastType::NUMPY);
        auto newMultiply =
                rewriter.create<IE::MultiplyOp>(concatOp->getLoc(), concatInput, singleConstResult, numpyBroadcast,
                                                origOp.getPostOpAttr(), origOp.getClampAttr());
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
    if (!isSuitableConstant(constOp)) {
        _log.trace("Ignore: IE.Multiply is not floating-point constant scalar multiplication");
        return mlir::failure();
    }
    const auto nonConstMultiplyOperand =
            origOp.getInput1().getDefiningOp() == constOp ? origOp.getInput2() : origOp.getInput1();
    auto convOp = findOriginOp<IE::ConvolutionOp>(_log, nonConstMultiplyOperand.getDefiningOp());
    if (convOp == nullptr) {
        _log.trace("Ignore: IE.Multiply has no preceding convolution - cannot fuse");
        return mlir::failure();
    }

    _log.trace("Fusing IE.Multiply at {0} into convolution at {1}", origOp->getLoc(), convOp->getLoc());
    // fuse IE.Multiply by specifying the static scale attribute
    const auto originalScale = convOp.getStaticScaleAttr() ? convOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
    const auto newScale = originalScale * constOp.getContent().getSplatValue<double>();
    convOp.setStaticScaleAttr(mlir::FloatAttr::get(mlir::Float32Type::get(origOp.getContext()), newScale));

    rewriter.replaceAllUsesWith(origOp.getOutput(), nonConstMultiplyOperand);

    return mlir::success();
}

class FuseMultiplyToConvPass final : public IE::arch37xx::FuseMultiplyToConvBase<FuseMultiplyToConvPass> {
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

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createFuseMultiplyToConvPass(Logger log) {
    return std::make_unique<FuseMultiplyToConvPass>(log);
}

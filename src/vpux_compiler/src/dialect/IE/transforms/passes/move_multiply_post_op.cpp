//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

//
// MoveMultiplyPostMatmul
//

//                  (1x32x1024x80)           (1x1x1x1)
//                              \               /
//      (1x32x1x80)         IE.Multiply (1x32x1024x80)
//               \            /
//            IE.Matmul(1x32x1x1024)

// To

//        (1x32x1x80)    1x32x1024x80)
//              \            /
//             IE.Matmul(1x32x1x1024)       (1x1x1x1)
//                         \                   /
//                        IE.Multiply (1x32x1x1024)

class MoveMultiplyPostMatmul final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MoveMultiplyPostMatmul(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("MoveMultiplyPostMatmul");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isBeneficialToConvert(ShapeRef inShape, ShapeRef outShape) {
    return inShape.totalSize() > outShape.totalSize();
}

mlir::Value getSingleDataInput(IE::MultiplyOp multiplyOp) {
    for (auto operand : multiplyOp->getOperands()) {
        if (auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(operand.getDefiningOp())) {
            if (IE::isBaseContentSplat(constOp)) {
                return operand;
            }
            return nullptr;
        }
        auto shape = getShape(operand);
        if (vpux::details::calcTotalShapeSize(shape.raw()) == 1) {
            return operand;
        }
    }
    return nullptr;
}

mlir::LogicalResult MoveMultiplyPostMatmul::matchAndRewrite(IE::MultiplyOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got multiply layer at '{1}'", origOp->getName(), origOp->getLoc());
    if (!origOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "multiply has more than one user");
    }

    if (origOp.getPostOpAttr() != nullptr) {
        return matchFailed(rewriter, origOp, "multiply has post op attr");
    }

    if (origOp.getClampAttr() != nullptr) {
        return matchFailed(rewriter, origOp, "multiply has clamp attr");
    }

    auto singleDataInput = getSingleDataInput(origOp);
    if (singleDataInput == nullptr) {
        return matchFailed(rewriter, origOp, "multiply doesn't have single data input");
    }

    const auto nonSingleDataOperand = origOp.getInput1() == singleDataInput ? origOp.getInput2() : origOp.getInput1();

    auto matmulOp = mlir::dyn_cast<IE::MatMulOp>(*origOp.getOutput().getUsers().begin());
    if (matmulOp == nullptr) {
        return matchFailed(rewriter, origOp, "multiply user is not a matmul");
    }

    if (!isBeneficialToConvert(getShape(origOp.getOutput()), getShape(matmulOp.getOutput()))) {
        return matchFailed(rewriter, origOp, "not benefical to swap multiply with matmul");
    }

    rewriter.setInsertionPoint(matmulOp);
    auto matmulInput1 = matmulOp.getInput1().getDefiningOp() == origOp ? nonSingleDataOperand : matmulOp.getInput1();
    auto matmulInput2 = matmulOp.getInput2().getDefiningOp() == origOp ? nonSingleDataOperand : matmulOp.getInput2();
    auto newMatMul = rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), matmulInput1, matmulInput2,
                                                   matmulOp.getTransposeA(), matmulOp.getTransposeB());

    auto multiplyInput1 = origOp.getInput1() == singleDataInput ? origOp.getInput1() : newMatMul.getOutput();
    auto multiplyInput2 = origOp.getInput2() == singleDataInput ? origOp.getInput2() : newMatMul.getOutput();

    auto newMultiply = rewriter.create<IE::MultiplyOp>(
            origOp->getLoc(), multiplyInput1, multiplyInput2, origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    rewriter.replaceOp(matmulOp, newMultiply.getOutput());
    _log.trace("Successfully swap multiply with matmul");

    return mlir::success();
}

//
// MoveMultiplyPostFC
//

class MoveMultiplyPostFC final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MoveMultiplyPostFC(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("MoveMultiplyPost");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isMultiplyUsedForQuantization(IE::MultiplyOp multiplyOp) {
    mlir::Value input = multiplyOp.getInput1();
    while (input) {
        auto inputBlock = mlir::dyn_cast_or_null<mlir::BlockArgument>(input);
        if (inputBlock == nullptr) {
            auto parentOp = input.getDefiningOp();
            if (mlir::isa_and_nonnull<IE::ConvertOp, IE::SubtractOp>(parentOp)) {
                input = parentOp->getOperand(0);
                continue;
            } else {
                return false;
            }
        } else {
            auto outType = mlir::cast<vpux::NDTypeInterface>(input.getType());
            auto outElementType = outType.getElementType();
            return outElementType.isInteger(8) || outElementType.isInteger(4) || outElementType.isUnsignedInteger(8) ||
                   outElementType.isUnsignedInteger(4);
        }
    }

    return false;
}

mlir::LogicalResult MoveMultiplyPostFC::matchAndRewrite(IE::MultiplyOp multiplyOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got multiply layer at '{1}'", multiplyOp->getName(), multiplyOp->getLoc());
    if (multiplyOp.getPostOpAttr() != nullptr || multiplyOp.getClampAttr() != nullptr ||
        multiplyOp.getOutputChannelsAttr() != nullptr || multiplyOp.getInputChannelsAttr() != nullptr) {
        return matchFailed(rewriter, multiplyOp, "multiply should not have extra attribute");
    }

    auto inputBlock = mlir::dyn_cast_or_null<mlir::BlockArgument>(multiplyOp.getInput2());
    if (inputBlock == nullptr) {
        return matchFailed(rewriter, multiplyOp, "second input of multiply is not a argument input");
    }

    if (!multiplyOp->hasOneUse()) {
        return matchFailed(rewriter, multiplyOp, "multiply has multi users");
    }

    if (!isMultiplyUsedForQuantization(multiplyOp)) {
        return matchFailed(rewriter, multiplyOp, "multiply is not used for quantization");
    }

    IE::FullyConnectedOp fcOp = nullptr;
    auto convertOp = mlir::dyn_cast<IE::ConvertOp>(*multiplyOp->user_begin());
    if (convertOp != nullptr && convertOp->hasOneUse()) {
        fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(*convertOp->user_begin());
    } else {
        fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(*multiplyOp->user_begin());
    }

    // MultiplyOp need to be the weights of FC
    if (fcOp == nullptr ||
        (fcOp.getWeights().getDefiningOp() != convertOp && fcOp.getWeights().getDefiningOp() != multiplyOp)) {
        return matchFailed(rewriter, multiplyOp, "multiply user is not FC");
    }

    auto fcOutShape = getShape(fcOp.getOutput());
    auto scaleShape = getShape(multiplyOp.getInput2());
    auto isChannelWiseQuant = [](ShapeRef shape) {
        const auto greaterThanOne = [](auto dim) {
            return dim > 1;
        };
        if (shape.size() != 2) {
            return false;
        }
        if (llvm::count_if(shape.raw(), greaterThanOne) == 1) {
            return true;
        }
        return false;
    };

    if (!isChannelWiseQuant(scaleShape)) {
        return matchFailed(rewriter, multiplyOp, "not channel wise quant");
    }

    if (fcOutShape.raw().back() != scaleShape.raw().front()) {
        return matchFailed(rewriter, multiplyOp, "scale shape doesn't match with FC output shape");
    }

    const auto memPerm = mlir::AffineMapAttr::get(
            mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{1, 0}, multiplyOp->getContext()));
    auto transpose = rewriter.create<IE::TransposeOp>(appendLoc(multiplyOp->getLoc(), "_transpose"), inputBlock,
                                                      nullptr, memPerm)
                             .getOutput();
    rewriter.setInsertionPointAfter(fcOp);
    auto argMultiply = rewriter.create<IE::MultiplyOp>(appendLoc(multiplyOp->getLoc(), "_arg_multiply"),
                                                       fcOp.getOutput(), transpose, multiplyOp.getAutoBroadcastAttr(),
                                                       nullptr, nullptr, nullptr, nullptr);
    fcOp.getOutput().replaceUsesWithIf(argMultiply.getOutput(), [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() != argMultiply;
    });
    rewriter.replaceOp(multiplyOp, multiplyOp.getInput1());

    _log.trace("Successfully move multiply post FC for Argument input");
    return mlir::success();
}

//
// MoveMultiplySubtractPostGather
//

class MoveMultiplySubtractPostGather final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    MoveMultiplySubtractPostGather(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("MoveMultiplySubtractPostGather");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveMultiplySubtractPostGather::matchAndRewrite(IE::GatherOp gatherOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}' ..{2} ", gatherOp->getName(), gatherOp->getLoc(), &rewriter);

    // benefit when gather output size is smaller than input size.
    if (!isBeneficialToConvert(getShape(gatherOp.getInput()), getShape(gatherOp.getOutput()))) {
        return matchFailed(rewriter, gatherOp, "not beneficial to do the convert");
    }

    if (gatherOp.getAxisValueAttr() != nullptr && gatherOp.getAxisValue().value() != 0) {
        return matchFailed(rewriter, gatherOp, "axis need to be the first dim");
    }

    IE::MultiplyOp multiplyOp = nullptr;
    IE::ConvertOp convertOpBeforeGather = mlir::dyn_cast_or_null<IE::ConvertOp>(gatherOp.getInput().getDefiningOp());
    if (convertOpBeforeGather != nullptr) {
        multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(convertOpBeforeGather.getInput().getDefiningOp());
    } else {
        multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(gatherOp.getInput().getDefiningOp());
    }

    if (multiplyOp == nullptr || multiplyOp.getPostOpAttr() != nullptr || multiplyOp.getClampAttr() != nullptr ||
        multiplyOp.getOutputChannelsAttr() != nullptr || multiplyOp.getInputChannelsAttr() != nullptr) {
        return matchFailed(rewriter, gatherOp, "not a required multiplyOp");
    }

    // multiply need to be used for quantization
    if (!isMultiplyUsedForQuantization(multiplyOp)) {
        return matchFailed(rewriter, multiplyOp, "multiply is not used for quantization");
    }

    if (mlir::dyn_cast<mlir::BlockArgument>(multiplyOp.getInput2()) == nullptr) {
        return matchFailed(rewriter, gatherOp, "multiply input2 is not a block argument");
    }

    IE::SubtractOp subtractOp = nullptr;
    IE::ConvertOp convertOpAfterInArg = nullptr;
    convertOpAfterInArg = mlir::dyn_cast_or_null<IE::ConvertOp>(multiplyOp.getInput1().getDefiningOp());
    if (convertOpAfterInArg == nullptr) {
        subtractOp = mlir::dyn_cast_or_null<IE::SubtractOp>(multiplyOp.getInput1().getDefiningOp());
        if (subtractOp == nullptr || subtractOp.getPostOpAttr() != nullptr || subtractOp.getClampAttr() != nullptr ||
            subtractOp.getOutputChannelsAttr() != nullptr || subtractOp.getInputChannelsAttr() != nullptr) {
            return matchFailed(rewriter, gatherOp, "not a required subtractOp");
        }

        if (mlir::dyn_cast<mlir::BlockArgument>(subtractOp.getInput2()) == nullptr) {
            return matchFailed(rewriter, gatherOp, "subtractOp input2 is not a block argument");
        }

        convertOpAfterInArg = mlir::dyn_cast_or_null<IE::ConvertOp>(subtractOp.getInput1().getDefiningOp());
    }

    if (convertOpAfterInArg == nullptr) {
        return matchFailed(rewriter, gatherOp, "there is not convertOp");
    }
    auto convertInArg = mlir::dyn_cast<mlir::BlockArgument>(convertOpAfterInArg.getInput());
    if (convertInArg == nullptr) {
        return matchFailed(rewriter, gatherOp, "convertOp input is not a block argument");
    }

    if (getShape(gatherOp.getInput()) != getShape(convertOpAfterInArg.getInput())) {
        return matchFailed(rewriter, gatherOp, "block argument shape doesn't match with gather input shape");
    }

    auto newGather = rewriter.create<IE::GatherOp>(gatherOp->getLoc(), convertInArg, gatherOp.getIndices(), nullptr,
                                                   gatherOp.getAxisValueAttr(), gatherOp.getBatchDims(),
                                                   gatherOp.getIndicesRankAttr());
    // newGather output type is int or uint, convert it to original one
    const auto convertOutType = mlir::dyn_cast<vpux::NDTypeInterface>(convertOpAfterInArg.getOutput().getType());
    auto multiplyInput = rewriter.create<IE::ConvertOp>(convertOpAfterInArg->getLoc(), newGather.getOutput(),
                                                        convertOutType.getElementType())
                                 .getOutput();

    if (subtractOp) {
        auto subtractGather = rewriter.create<IE::GatherOp>(
                appendLoc(gatherOp->getLoc(), "_subtract"), subtractOp.getInput2(), gatherOp.getIndices(), nullptr,
                gatherOp.getAxisValueAttr(), gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());
        multiplyInput =
                rewriter.create<IE::SubtractOp>(subtractOp->getLoc(), multiplyInput, subtractGather.getOutput(),
                                                subtractOp.getAutoBroadcast(), nullptr, nullptr, nullptr, nullptr);
    }

    auto multiplyGather = rewriter.create<IE::GatherOp>(
            appendLoc(gatherOp->getLoc(), "_multiply"), multiplyOp.getInput2(), gatherOp.getIndices(), nullptr,
            gatherOp.getAxisValueAttr(), gatherOp.getBatchDims(), gatherOp.getIndicesRankAttr());
    auto postMultiply =
            rewriter.create<IE::MultiplyOp>(multiplyOp->getLoc(), multiplyInput, multiplyGather.getOutput(),
                                            multiplyOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr, nullptr)
                    .getOutput();
    if (convertOpBeforeGather) {
        const auto gatherOutType = mlir::dyn_cast<vpux::NDTypeInterface>(gatherOp.getOutput().getType());
        postMultiply = rewriter.create<IE::ConvertOp>(convertOpBeforeGather->getLoc(), postMultiply,
                                                      gatherOutType.getElementType())
                               .getOutput();
    }
    rewriter.replaceOp(gatherOp, postMultiply);

    _log.trace("Successfully move multiply post gather");
    return mlir::success();
}

//
// MoveMultiplyPostOpPass
//

class MoveMultiplyPostOpPass final : public IE::MoveMultiplyPostOpBase<MoveMultiplyPostOpPass> {
public:
    explicit MoveMultiplyPostOpPass(Logger log, bool moveMultiplyPostFCForDynamicQuant)
            : _moveMultiplyPostFCForDynamicQuant(moveMultiplyPostFCForDynamicQuant) {
        Base::initLogger(log, Base::getArgumentName());
    }
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _moveMultiplyPostFCForDynamicQuant = false;
};

mlir::LogicalResult MoveMultiplyPostOpPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (moveMultiplyPostFCForDynamicQuant.hasValue()) {
        _log.trace("Overloading the default value {0} of the '_moveMultiplyPostFCForDynamicQuant' field to the "
                   "value {1} of the "
                   "pass option "
                   "'moveMultiplyPostFCForDynamicQuant' generated by MLIR",
                   _moveMultiplyPostFCForDynamicQuant, moveMultiplyPostFCForDynamicQuant);
        _moveMultiplyPostFCForDynamicQuant = moveMultiplyPostFCForDynamicQuant;
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void MoveMultiplyPostOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveMultiplyPostMatmul>(&ctx, _log);
    patterns.add<MoveMultiplySubtractPostGather>(&ctx, _log);
    if (_moveMultiplyPostFCForDynamicQuant) {
        patterns.add<MoveMultiplyPostFC>(&ctx, _log);
    }

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveMultiplyPostOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMoveMultiplyPostOpPass(Logger log, bool moveMultiplyPostFCForDynamicQuant) {
    return std::make_unique<MoveMultiplyPostOpPass>(log, moveMultiplyPostFCForDynamicQuant);
}

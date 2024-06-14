//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

mlir::LogicalResult verifyAndBroadcastInput(mlir::Location loc, mlir::Value& input, vpux::ShapeRef inputShape,
                                            vpux::ShapeRef outputShape, mlir::PatternRewriter& rewriter) {
    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (outputShape.size() != 4 || inputShape.size() != 4) {
        return mlir::failure();
    }
    if (inputShape[N] != 1 || inputShape[H] != 1 || inputShape[W] != 1) {
        return mlir::failure();
    }

    if (inputShape[C] != outputShape[C] && inputShape[C] != 1) {
        return mlir::failure();
    }

    // Broadcast scalar for all channels
    if (inputShape[C] != outputShape[C] && inputShape[C] == 1) {
        SmallVector<mlir::Operation*> opsVec;
        Const::DeclareOp input2Const = nullptr;
        // Convert [Const] -> [optional several Reshapes]-> [optional FQ] -> [optional several Reshapes] ->
        // [Multiply/Add] case to scaleShift
        mlir::Operation* operation = input.getDefiningOp();
        if (operation == nullptr) {
            return mlir::failure();
        }
        while (operation && mlir::isa<IE::ReshapeOp, IE::FakeQuantizeOp, Const::DeclareOp>(operation)) {
            if (mlir::isa<IE::ReshapeOp, IE::FakeQuantizeOp>(operation)) {
                if (!operation->hasOneUse()) {
                    return mlir::failure();
                }
                opsVec.insert(opsVec.begin(), operation);
                operation = operation->getOperand(0).getDefiningOp();
                continue;  // Continue searching for Const::DeclareOp
            }

            if (mlir::isa<Const::DeclareOp>(operation)) {
                input2Const = mlir::dyn_cast_or_null<Const::DeclareOp>(operation);
                break;
            }
        }

        // Const input can not be found
        if (input2Const == nullptr) {
            return mlir::failure();
        }

        Const::ContentAttr dataAttr = input2Const.getContentAttr().broadcast(C, outputShape[C]);

        if (dataAttr == nullptr) {
            return mlir::failure();
        }

        auto dataConstOp = rewriter.create<Const::DeclareOp>(loc, dataAttr.getType(), dataAttr);

        if (opsVec.size() == 0) {
            // [Const]->[Multiply/Add] case
            input = dataConstOp.getOutput();
        } else {
            // [Const] -> [several Reshapes]-> [FQ] -> [several Reshapes] -> [Multiply/Add] case
            opsVec.front()->getOpOperand(0).set(dataConstOp.getOutput());
            for (auto op : opsVec) {
                inferReturnTypes(op, InferShapedTypeMode::SHAPE);
            }
        }
    }

    return mlir::success();
}

//
// ConvertBiasToScaleShift
//

template <typename BiasTypeOp>
class ConvertBiasToScaleShift final : public mlir::OpRewritePattern<BiasTypeOp> {
public:
    ConvertBiasToScaleShift<BiasTypeOp>(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<BiasTypeOp>(ctx), _log(log) {
        this->setDebugName("ConvertBiasToScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(BiasTypeOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename BiasTypeOp>
mlir::LogicalResult ConvertBiasToScaleShift<BiasTypeOp>::matchAndRewrite(BiasTypeOp biasOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", biasOp->getName(), biasOp->getLoc());
    auto inElemType = biasOp.getInput2().getType().template cast<vpux::NDTypeInterface>().getElementType();
    auto outElemType = biasOp.getOutput().getType().template cast<vpux::NDTypeInterface>().getElementType();

    // from the ops defination, scale shift can only support F16
    if (!(inElemType.isF16())) {
        _log.trace("Could not convert to scale shift due to input date type is not FP16");
        return mlir::failure();
    }

    if (inElemType != outElemType) {
        _log.nest().trace("op {0} input and output types are not matching", biasOp->getName());
        return mlir::failure();
    }

    const auto lhsType = biasOp.getInput1().getType().template cast<vpux::NDTypeInterface>();
    const auto outShapeRes = biasOp.getOutput().getType().template cast<vpux::NDTypeInterface>();

    bool lhsIsActivation = (lhsType == outShapeRes);
    auto activationInput = lhsIsActivation ? biasOp.getInput1() : biasOp.getInput2();
    auto biasInput = lhsIsActivation ? biasOp.getInput2() : biasOp.getInput1();

    auto findBiasConst = IE::getConstParentOp(biasInput);
    if (mlir::failed(findBiasConst)) {
        _log.nest().trace("op {0} input is not constant", biasOp->getName());
        return mlir::failure();
    }

    auto mulOutShape = getShape(biasOp.getOutput());
    auto biasesShape = getShape(biasInput);

    if (verifyAndBroadcastInput(biasOp.getLoc(), biasInput, biasesShape, mulOutShape, rewriter).failed()) {
        _log.nest().trace("op {0} input cannot be broadcast", biasOp->getName());
        return mlir::failure();
    }

    auto biasConst = findBiasConst.value();

    // Convert:
    //
    // Tensor              Const
    //    |                  |
    //    |               Negative
    //    |                  |
    //     \______AddOp______/
    //              |
    //
    // To:
    //
    // Tensor             NewConst
    //    |                  |
    //    |                  |
    //    |                  |
    //     \___ScaleShift___/
    //              |

    if (mlir::isa<IE::NegativeOp>(biasInput.getDefiningOp())) {
        const auto negativeConstAttr = biasConst.getContentAttr().rescale(-1.0);
        auto newBiasInput =
                rewriter.create<Const::DeclareOp>(biasConst->getLoc(), biasConst.getType(), negativeConstAttr);
        _log.nest().trace("replacing op {0} with ScaleShift", biasOp->getName());
        rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), activationInput, nullptr, newBiasInput);

        return mlir::success();
    }

    if (mlir::isa<IE::SubtractOp>(biasOp)) {
        const auto negativeConstAttr = biasConst.getContentAttr().rescale(-1.0);
        rewriter.replaceOpWithNewOp<Const::DeclareOp>(biasConst, biasConst.getType(), negativeConstAttr)
                ->setLoc(biasConst->getLoc());
    }

    _log.nest().trace("replaced op {0} with ScaleShift", biasOp->getName());
    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), activationInput, nullptr, biasInput);

    return mlir::success();
}

//
// ConvertMultiplyToScaleShift
//

class ConvertMultiplyToScaleShift final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    ConvertMultiplyToScaleShift(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("ConvertMultiplyToScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertMultiplyToScaleShift::matchAndRewrite(IE::MultiplyOp mulOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", mulOp->getName(), mulOp->getLoc());
    const auto lhsType = mulOp.getInput1().getType().cast<mlir::ShapedType>();
    const auto outShapeRes = mulOp.getOutput().getType().cast<mlir::ShapedType>();

    // from the ops defination, scale shift can only support F16
    const auto lhsEltmentType = lhsType.getElementType();
    if (!(lhsEltmentType.isF16())) {
        _log.trace("Could not convert to scale shift due to input data type is not FP16");
        return mlir::failure();
    }

    bool lhsIsActivation = (lhsType == outShapeRes);
    auto activationInput = lhsIsActivation ? mulOp.getInput1() : mulOp.getInput2();
    auto weightsInput = lhsIsActivation ? mulOp.getInput2() : mulOp.getInput1();

    auto mulOutShape = getShape(mulOp.getOutput());
    auto weightsShape = getShape(weightsInput);

    // Activation shape and scaleShift output shape should be consistent
    if (getShape(activationInput) != mulOutShape) {
        return mlir::failure();
    }

    if (verifyAndBroadcastInput(mulOp.getLoc(), weightsInput, weightsShape, mulOutShape, rewriter).failed()) {
        return mlir::failure();
    }

    _log.nest().trace("replaced op {0} with ScaleShift", mulOp->getName());
    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(mulOp, mulOp.getType(), activationInput, weightsInput, nullptr);

    return mlir::success();
}

//
// ConvertToScaleShiftPass
//

class ConvertToScaleShiftPass final : public IE::ConvertToScaleShiftBase<ConvertToScaleShiftPass> {
public:
    explicit ConvertToScaleShiftPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertToScaleShiftPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertBiasToScaleShift<IE::AddOp>>(&ctx, _log);
    patterns.add<ConvertBiasToScaleShift<IE::SubtractOp>>(&ctx, _log);
    patterns.add<ConvertMultiplyToScaleShift>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToScaleShiftPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToScaleShiftPass(Logger log) {
    return std::make_unique<ConvertToScaleShiftPass>(log);
}

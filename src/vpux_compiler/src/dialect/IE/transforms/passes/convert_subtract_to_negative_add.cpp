//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

// experimental value
constexpr int64_t THRESHOLD_FOR_CONVERT_SUBSTRACT = 8192;

bool isGreaterShape(ShapeRef shape1, ShapeRef shape2) {
    if (shape1.size() < shape2.size()) {
        return false;
    }
    for (auto p : zip(shape1, shape2)) {
        if (std::get<0>(p) < std::get<1>(p)) {
            return false;
        }
    }
    return true;
}

static inline Const::DeclareOp createConstOpFromValue(mlir::PatternRewriter& rewriter, mlir::Location loc, float val,
                                                      mlir::RankedTensorType argType) {
    const auto denseElementVal = wrapData(argType, val);
    VPUX_THROW_UNLESS(denseElementVal != nullptr,
                      "Subtract pool has incompatible data type {0}, only float16 or float32 are supported",
                      argType.getElementType());

    return rewriter.create<Const::DeclareOp>(loc, argType, Const::ContentAttr::get(denseElementVal));
}

Const::DeclareOp getConstInput(IE::SubtractOp subtractOp) {
    if (auto input2Fq = subtractOp.getInput2().getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto fqConstInput = input2Fq.getInput().getDefiningOp<Const::DeclareOp>()) {
            return fqConstInput;
        }
    } else if (auto input2Const = subtractOp.getInput2().getDefiningOp<Const::DeclareOp>()) {
        return input2Const;
    }
    return nullptr;
}

mlir::Value createNegativeFqVal(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value fqVal,
                                mlir::RankedTensorType storageType) {
    auto valConst = fqVal.getDefiningOp<Const::DeclareOp>();
    auto valConstContent = valConst.getContent();

    VPUX_THROW_UNLESS(valConstContent.isSplat(), "Second input's FQ constant is not splat");
    auto inValue = valConstContent.getSplatValue<float>();
    auto negativeValue = (inValue == 0) ? 0 : -1 * inValue;
    return createConstOpFromValue(rewriter, loc, negativeValue, storageType);
}

IE::FakeQuantizeOp createNewFq(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value fqInput,
                               IE::FakeQuantizeOp initialFqOp) {
    auto fqValType = initialFqOp.getInputHigh().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto storageType = mlir::RankedTensorType::get({1, 1, 1, 1}, fqValType);

    mlir::Value inLow = createNegativeFqVal(rewriter, loc, initialFqOp.getInputHigh(), storageType);
    mlir::Value inHigh = createNegativeFqVal(rewriter, loc, initialFqOp.getInputLow(), storageType);
    mlir::Value outLow = createNegativeFqVal(rewriter, loc, initialFqOp.getOutputHigh(), storageType);
    mlir::Value outHigh = createNegativeFqVal(rewriter, loc, initialFqOp.getOutputLow(), storageType);

    return rewriter.create<IE::FakeQuantizeOp>(loc, fqInput, inLow, inHigh, outLow, outHigh,
                                               initialFqOp.getLevelsAttr(), initialFqOp.getLowFpTypeAttr(),
                                               initialFqOp.getAutoBroadcastAttr());
}

//
// ConvertSubtractToDWConvAdd
//

class ConvertSubtractToDWConvAdd final : public mlir::OpRewritePattern<IE::SubtractOp> {
public:
    ConvertSubtractToDWConvAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SubtractOp>(ctx), _log(log) {
        setDebugName("ConvertSubtractToDWConvAdd");
    }

    mlir::LogicalResult matchAndRewrite(IE::SubtractOp subOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSubtractToDWConvAdd::matchAndRewrite(IE::SubtractOp subOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto subOpLoc = subOp.getLoc();
    _log.trace("Found SubtractOp at location '{0}'", subOpLoc);

    auto input1 = subOp.getInput1();
    auto input2 = subOp.getInput2();

    const auto input1Shape = getShape(input1);
    const auto input2Shape = getShape(input2);
    if (input1Shape.size() != 4 || input2Shape.size() != 4) {
        return mlir::failure();
    }

    const auto elemType = subOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();

    mlir::Value negativeInput = nullptr;

    auto constInput2 = getConstInput(subOp);
    auto fqInput2 = input2.getDefiningOp<IE::FakeQuantizeOp>();
    if (constInput2 != nullptr) {
        auto constInput2Content = constInput2.getContentAttr();
        auto negativeContent = constInput2Content.rescale(-1.0);
        negativeInput = rewriter.create<Const::DeclareOp>(subOpLoc, constInput2.getType(), negativeContent);
    } else {
        const auto inputC = input2Shape[Dims4D::Act::C];
        const Shape filterShape = {inputC, 1, 1, 1};
        const auto filterStorageType = mlir::RankedTensorType::get(to_small_vector(filterShape), elemType);
        auto dwConvFilter = createConstOpFromValue(rewriter, subOpLoc, -1.0f, filterStorageType);
        auto filter = dwConvFilter.getOutput();

        if (fqInput2 != nullptr) {
            const auto fqArgType = mlir::RankedTensorType::get({1, 1, 1, 1}, elemType);
            auto fqVal = createConstOpFromValue(rewriter, subOpLoc, -1.0f, fqArgType);
            auto filterFQ = rewriter.create<IE::FakeQuantizeOp>(subOpLoc, filter, fqVal, fqVal, fqVal, fqVal,
                                                                fqInput2.getLevelsAttr(), /*lowFpType=*/nullptr,
                                                                fqInput2.getAutoBroadcastAttr());
            filter = filterFQ.getOutput();
        }

        auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
        auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
        auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
        auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
        auto groupAttr = getIntAttr(rewriter, inputC);

        auto dwConv = rewriter.create<IE::GroupConvolutionOp>(subOpLoc, input2, filter, /*bias=*/nullptr, stridesAttr,
                                                              padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                                                              /*post_opAttr=*/nullptr, /*clampAttr=*/nullptr);
        negativeInput = dwConv.getOutput();
    }

    if (fqInput2 != nullptr) {
        negativeInput = createNewFq(rewriter, subOpLoc, negativeInput, fqInput2).getOutput();
    }

    if (constInput2 == nullptr) {
        auto negativeInputShape = getShape(negativeInput);
        if (input1Shape != negativeInputShape) {
            auto ctx = rewriter.getContext();

            IE::BroadcastOp inputBroadcast;
            const auto broadcastType = IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY);
            if (isGreaterShape(input1Shape, negativeInputShape)) {
                inputBroadcast = rewriter.create<IE::BroadcastOp>(
                        subOpLoc, negativeInput,
                        vpux::IE::createShapeConstForBroadCast(rewriter, ctx, subOpLoc, input1Shape), nullptr,
                        broadcastType);
                negativeInput = inputBroadcast.getOutput();
            } else {
                inputBroadcast = rewriter.create<IE::BroadcastOp>(
                        subOpLoc, input1,
                        vpux::IE::createShapeConstForBroadCast(rewriter, ctx, subOpLoc, negativeInputShape), nullptr,
                        broadcastType);
                input1 = inputBroadcast.getOutput();
            }
        }
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(subOp, input1, negativeInput, subOp.getAutoBroadcastAttr(),
                                           /*post_op=*/nullptr, /*clamp=*/nullptr);
    return mlir::success();
}

//
// ConvertSubtractToNegativeAdd
//

class ConvertSubtractToNegativeAdd final : public mlir::OpRewritePattern<IE::SubtractOp> {
public:
    ConvertSubtractToNegativeAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SubtractOp>(ctx), _log(log) {
        setDebugName("ConvertSubtractToNegativeAdd");
    }

    mlir::LogicalResult matchAndRewrite(IE::SubtractOp subOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSubtractToNegativeAdd::matchAndRewrite(IE::SubtractOp subOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Found SubtractOp at location '{0}'", subOp.getLoc());

    auto input1 = subOp.getInput1();
    auto input2 = subOp.getInput2();

    auto negativeOp = rewriter.create<IE::NegativeOp>(subOp.getLoc(), input2.getType(), input2);

    rewriter.replaceOpWithNewOp<IE::AddOp>(subOp, input1, negativeOp, subOp.getAutoBroadcastAttr(),
                                           /*post_op=*/nullptr, /*clamp=*/nullptr);
    return mlir::success();
}

//
// ConvertSubtractToAddPass
//

class ConvertSubtractToAddPass final : public IE::ConvertSubtractToAddBase<ConvertSubtractToAddPass> {
public:
    explicit ConvertSubtractToAddPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertSubtractToAddPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto notEqualOne = [](auto dim) {
        return dim != 1;
    };

    const auto isLegalSubtractOp = [&](IE::SubtractOp op) {
        const auto input1ElemType = op.getInput1().getType().cast<vpux::NDTypeInterface>().getElementType();
        const auto input2ElemType = op.getInput2().getType().cast<vpux::NDTypeInterface>().getElementType();

        // SubTract with INTEGER input/ouput can not be executed as NCE op, and should be kept like Subtract and be
        // executed on shave.
        const auto hasIntInput = input1ElemType.isa<mlir::IntegerType>() || input2ElemType.isa<mlir::IntegerType>();

        if (hasIntInput) {
            return true;
        }

        auto constInput2 = getConstInput(op);
        const auto input1Shape = getShape(op.getInput1());
        const auto input2Shape = getShape(op.getInput2());

        // find the legal pattern that should not be changed by pass
        //    SubTract:<1x1x1xW>, <1x1x1x1>
        // Or
        //    SubTract:<1x1x1x1>, <1x1x1x1>

        const auto numOfNotOneDimForInput1 = llvm::count_if(input1Shape, notEqualOne);
        const auto isInput2Scalar = llvm::count_if(input2Shape, notEqualOne) == 0;
        const auto isBeneficialPattern = [&]() {
            if (constInput2) {
                return false;
            }
            if (!isInput2Scalar) {
                return false;
            }
            if (numOfNotOneDimForInput1 == 0) {
                return true;
            }
            if (numOfNotOneDimForInput1 == 1 && input1Shape[Dims4D::Act::W] >= THRESHOLD_FOR_CONVERT_SUBSTRACT) {
                return true;
            }
            return false;
        };
        if (isBeneficialPattern()) {
            return true;
        }
        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::GroupConvolutionOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<IE::BroadcastOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::NegativeOp>();
    target.addDynamicallyLegalOp<IE::SubtractOp>(isLegalSubtractOp);

    const auto arch = VPU::getArch(func);
    mlir::RewritePatternSet patterns(&ctx);
    if (arch == VPU::ArchKind::NPU30XX) {
        patterns.add<ConvertSubtractToNegativeAdd>(&ctx, _log);
    } else {
        patterns.add<ConvertSubtractToDWConvAdd>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSubtractToAddPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSubtractToAddPass(Logger log) {
    return std::make_unique<ConvertSubtractToAddPass>(log);
}

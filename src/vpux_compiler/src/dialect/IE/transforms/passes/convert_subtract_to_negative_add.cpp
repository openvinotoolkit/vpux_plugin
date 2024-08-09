//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

static inline Const::DeclareOp createConstOpFromValue(mlir::PatternRewriter& rewriter, mlir::Location loc, float val,
                                                      mlir::RankedTensorType argType) {
    const auto denseElementVal = wrapData(argType, val);
    VPUX_THROW_UNLESS(denseElementVal != nullptr,
                      "Subtract pool has incompatible data type {0}, only float16 or float32 are supported",
                      argType.getElementType());

    return rewriter.create<Const::DeclareOp>(loc, argType, Const::ContentAttr::get(denseElementVal));
}

Const::DeclareOp getInputConstantOp(mlir::Value input) {
    if (input == nullptr) {
        return nullptr;
    }
    if (auto inputFq = input.getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto fqConstInput = inputFq.getInput().getDefiningOp<Const::DeclareOp>()) {
            return fqConstInput;
        }
    } else if (auto inputConst = input.getDefiningOp<Const::DeclareOp>()) {
        return inputConst;
    }
    return nullptr;
}

mlir::Value createNegativeFqVal(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value fqVal,
                                mlir::RankedTensorType storageType) {
    auto valConst = fqVal.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(valConst.getContentAttr().isSplat(), "Second input's FQ constant is not splat");
    auto valConstContent = valConst.getContent();
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

    mlir::Value firstInput = input1;
    const auto outputType = subOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (auto constInput1Op = getInputConstantOp(input1)) {
        auto constInput1Type = constInput1Op.getType().cast<NDTypeInterface>();
        auto broadcastContent = constInput1Op.getContentAttr();
        auto newInput1Shape = to_small_vector(input1Shape);

        for (const auto& [index, inOutSizes] : enumerate(zip(constInput1Type.getShape(), outputType.getShape()))) {
            auto [inSize, outSize] = inOutSizes;
            if (inSize != outSize) {
                broadcastContent = broadcastContent.broadcast(Dim(index), outSize);
                newInput1Shape[index] = outSize;
            }
        }

        auto newInput1Type = constInput1Type.changeShape(Shape(newInput1Shape));
        firstInput = rewriter.create<Const::DeclareOp>(subOpLoc, newInput1Type, broadcastContent);
    }

    auto fqInput2 = input2.getDefiningOp<IE::FakeQuantizeOp>();

    mlir::Value negativeInput = input2;
    if (auto constInput2 = getInputConstantOp(input2)) {
        auto constInput2Content = constInput2.getContentAttr();
        auto negativeContent = constInput2Content.rescale(-1.0);
        negativeInput = rewriter.create<Const::DeclareOp>(subOpLoc, constInput2.getType(), negativeContent);
    } else {
        const auto elemType = outputType.getElementType();
        const auto inputC = input2Shape[Dims4D::Act::C];
        const auto filterStorageType = mlir::RankedTensorType::get(SmallVector<int64_t>{inputC, 1, 1, 1}, elemType);
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

    rewriter.replaceOpWithNewOp<IE::AddOp>(subOp, firstInput, negativeInput, subOp.getAutoBroadcastAttr(),
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

    const auto isLegalSubtractOp = [&](IE::SubtractOp op) {
        const auto input1Type = op.getInput1().getType().cast<vpux::NDTypeInterface>();
        const auto input2Type = op.getInput2().getType().cast<vpux::NDTypeInterface>();
        const auto outputType = op.getResult().getType().cast<vpux::NDTypeInterface>();

        // SubTract with INTEGER input/ouput can not be executed as NCE op, and should be kept like Subtract and be
        // executed on shave.
        const auto hasIntInput = input1Type.getElementType().isa<mlir::IntegerType>() ||
                                 input2Type.getElementType().isa<mlir::IntegerType>();

        // Check if a broadcast operation is needed. If the input and output types do not match, and it cannot be
        // optimized through constant folding, then broadcasting is required
        const auto needsBroadcast = (input1Type != input2Type) &&
                                    ((input1Type != outputType && getInputConstantOp(op.getInput1()) == nullptr) ||
                                     (input2Type != outputType && getInputConstantOp(op.getInput2()) == nullptr));

        // Check if an expansion operation is needed. If the size of the output shape is not a multiple of the channel
        // alignment, then expansion is required
        auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(op.getOperation());
        const auto alignment = iface.getOutputChannelAlignment();
        const auto needsExpansion = (outputType.getShape().totalSize() % alignment) != 0;

        return hasIntInput || needsBroadcast || needsExpansion;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::GroupConvolutionOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::NegativeOp>();
    target.addDynamicallyLegalOp<IE::SubtractOp>(isLegalSubtractOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertSubtractToDWConvAdd>(&ctx, _log);

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

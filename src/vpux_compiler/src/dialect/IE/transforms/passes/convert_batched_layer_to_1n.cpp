//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <mlir/IR/IRMapping.h>

using namespace vpux;

namespace {

bool isEqualToOne(mlir::Value value, const Dim& dim) {
    const auto shape = getShape(value);
    VPUX_THROW_UNLESS(shape.size() > checked_cast<size_t>(dim.ind()), "Invalid Dim {0} for shape {1}", dim, shape);
    return shape[dim] == 1;
}

std::optional<Dim> getDimEqualsToOne(ArrayRef<mlir::Value> values) {
    const SmallVector<Dim> candidates = {Dims4D::Act::H, Dims4D::Act::W};
    for (const auto& dim : candidates) {
        auto dimEqualsToOne = llvm::all_of(values, [&](const auto& value) {
            return isEqualToOne(value, dim);
        });
        if (dimEqualsToOne) {
            return dim;
        }
    }
    return std::nullopt;
}

bool isShapeRankEq4(mlir::Value val) {
    const auto inputShape = getShape(val);
    return inputShape.size() == 4;
}

bool isPerAxisQuant(mlir::Value val) {
    auto elemType = val.getType().dyn_cast<vpux::NDTypeInterface>().getElementType();
    return elemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
}

IE::TransposeOp createTransposeForLayerInput(mlir::PatternRewriter& rewriter, mlir::Value input, const Dim& dim,
                                             mlir::Location loc) {
    auto originShape = getShape(input);
    auto dimOrder = DimsOrder::fromValue(input);
    SmallVector<unsigned> transPerm(originShape.size());
    std::iota(transPerm.begin(), transPerm.end(), 0);

    transPerm[dimOrder.dimPos(Dims4D::Act::N)] = checked_cast<unsigned>(dimOrder.dimPos(dim));
    transPerm[dimOrder.dimPos(dim)] = checked_cast<unsigned>(dimOrder.dimPos(Dims4D::Act::N));

    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transPerm, rewriter.getContext()));
    auto newLoc = appendLoc(loc, "_ConvertBatchedLayer_inTranspose");
    return rewriter.create<IE::TransposeOp>(newLoc, input, nullptr, orderAttr);
}

template <class ConcreteEltwiseOp>
void reshapeForEltwiseOp(ConcreteEltwiseOp eltwiseOp, mlir::PatternRewriter& rewriter) {
    const auto ctx = eltwiseOp->getContext();
    auto inputShape1 = getShape(eltwiseOp.getInput1());
    auto inputShape2 = getShape(eltwiseOp.getInput2());
    auto outputShape = getShape(eltwiseOp.getOutput());

    Shape newInShape1 = Shape(inputShape1);
    newInShape1[Dims4D::Act::N] = 1;
    newInShape1[Dims4D::Act::C] = inputShape1[Dims4D::Act::C] * inputShape1[Dims4D::Act::N];
    Shape newInShape2 = Shape(inputShape2);
    newInShape2[Dims4D::Act::N] = 1;
    newInShape2[Dims4D::Act::C] = inputShape2[Dims4D::Act::C] * inputShape2[Dims4D::Act::N];
    Shape newOutputShape = Shape(outputShape);
    newOutputShape[Dims4D::Act::N] = 1;
    newOutputShape[Dims4D::Act::C] = outputShape[Dims4D::Act::C] * outputShape[Dims4D::Act::N];
    const auto resultType =
            mlir::RankedTensorType::get(newOutputShape.raw(), eltwiseOp.getOutput().getType().getElementType());

    auto inShapeCast1 = rewriter.create<IE::ShapeCastOp>(eltwiseOp->getLoc(), eltwiseOp.getInput1(),
                                                         getIntArrayAttr(ctx, newInShape1));
    auto inShapeCast2 = rewriter.create<IE::ShapeCastOp>(eltwiseOp->getLoc(), eltwiseOp.getInput2(),
                                                         getIntArrayAttr(ctx, newInShape2));

    auto newEltwiseOp = rewriter.create<ConcreteEltwiseOp>(
            eltwiseOp->getLoc(), resultType, inShapeCast1.getResult(), inShapeCast2.getResult(),
            eltwiseOp.getAutoBroadcastAttr(), eltwiseOp.getPostOpAttr(), eltwiseOp.getClampAttr(),
            eltwiseOp.getOutputChannelsAttr(), eltwiseOp.getInputChannelsAttr());

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(eltwiseOp, newEltwiseOp.getOutput(),
                                                 getIntArrayAttr(ctx, outputShape));
}

void reshapeForGroupConv(IE::GroupConvolutionOp groupConvOp, mlir::PatternRewriter& rewriter) {
    const auto ctx = groupConvOp->getContext();

    Shape newInputShape = Shape(getShape(groupConvOp.getInput()));
    newInputShape[Dims4D::Act::C] = newInputShape[Dims4D::Act::N];
    newInputShape[Dims4D::Act::N] = 1;
    Shape newOutputShape = Shape(getShape(groupConvOp.getOutput()));
    newOutputShape[Dims4D::Act::C] = newOutputShape[Dims4D::Act::N];
    newOutputShape[Dims4D::Act::N] = 1;
    auto group = newInputShape[Dims4D::Act::C];

    auto inShapeCast = rewriter.create<IE::ShapeCastOp>(groupConvOp->getLoc(), groupConvOp.getInput(),
                                                        getIntArrayAttr(ctx, newInputShape));

    const auto filter = groupConvOp.getFilter();
    auto targetConstShape = Shape(getShape(filter));
    targetConstShape[Dims4D::Filter::OC] = group;
    auto targetShapeConst =
            vpux::IE::createShapeConstForBroadCast(rewriter, ctx, groupConvOp->getLoc(), targetConstShape);
    auto newFilter = rewriter.create<IE::BroadcastOp>(groupConvOp->getLoc(), filter, targetShapeConst,
                                                      /*axes_mapping*/ nullptr,
                                                      IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY));
    auto newGroupConvOp = rewriter.create<IE::GroupConvolutionOp>(
            groupConvOp->getLoc(), inShapeCast.getResult(), newFilter.getOutput(), groupConvOp.getBias(),
            groupConvOp.getStridesAttr(), groupConvOp.getPadsBeginAttr(), groupConvOp.getPadsEndAttr(),
            groupConvOp.getDilationsAttr(), getIntAttr(ctx, group), groupConvOp.getPostOpAttr(),
            groupConvOp.getClampAttr(), groupConvOp.getOutputChannelsAttr(), groupConvOp.getInputChannelsAttr());

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(groupConvOp, newGroupConvOp.getOutput(),
                                                 getIntArrayAttr(ctx, getShape(groupConvOp.getOutput())));
}

//
// LayerConverter
//

template <class ConcreteOp>
class LayerConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    LayerConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("LayerConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::IRMapping mapOperands(ConcreteOp origOp, mlir::AffineMapAttr& orderAttr,
                                mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult LayerConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got layer at '{0}'", origOp->getLoc());

    auto addOp = mlir::dyn_cast<IE::AddOp>(*origOp);
    if (addOp != nullptr && !getDimEqualsToOne({addOp.getInput1(), addOp.getInput2()}).has_value()) {
        reshapeForEltwiseOp<IE::AddOp>(addOp, rewriter);
        _log.trace("Reshape AddOp directly for not suitable transpose cases");
        return mlir::success();
    }
    auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*origOp);
    if (multiplyOp != nullptr && !getDimEqualsToOne({multiplyOp.getInput1(), multiplyOp.getInput2()}).has_value()) {
        reshapeForEltwiseOp<IE::MultiplyOp>(multiplyOp, rewriter);
        _log.trace("Reshape MultiplyOp directly for not suitable transpose cases");
        return mlir::success();
    }

    auto groupConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(*origOp);
    if (groupConvOp != nullptr && isEqualToOne(groupConvOp.getInput(), Dims4D::Act::C)) {
        reshapeForGroupConv(groupConvOp, rewriter);
        _log.trace("Reshape GroupConvOp directly for not suitable transpose cases");
        return mlir::success();
    }

    mlir::AffineMapAttr transPermAttr = nullptr;
    mlir::Value output = nullptr;
    auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(*origOp);
    if (convOp != nullptr && convOp.getInput() == convOp.getFilter()) {
        auto dim = getDimEqualsToOne({convOp.getInput()}).value();
        auto inTranspose = createTransposeForLayerInput(rewriter, convOp.getInput(), dim, convOp->getLoc());
        transPermAttr = inTranspose.getOrderValueAttr();
        VPUX_THROW_WHEN(transPermAttr == nullptr, "Can not get order value from input tranpose");
        output = rewriter.create<IE::ConvolutionOp>(convOp->getLoc(), inTranspose.getOutput(), convOp.getFilter(),
                                                    convOp.getBias(), convOp.getStridesAttr(),
                                                    convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(),
                                                    convOp.getDilationsAttr(), convOp.getPostOpAttr(),
                                                    convOp.getClampAttr(), convOp.getStaticScaleAttr(),
                                                    convOp.getOutputChannelsAttr(), convOp.getInputChannelsAttr())
                         .getOutput();
    } else if (groupConvOp != nullptr && groupConvOp.getInput() == groupConvOp.getFilter()) {
        auto dim = getDimEqualsToOne({groupConvOp.getInput()}).value();
        auto inTranspose = createTransposeForLayerInput(rewriter, groupConvOp.getInput(), dim, groupConvOp->getLoc());
        transPermAttr = inTranspose.getOrderValueAttr();
        VPUX_THROW_WHEN(transPermAttr == nullptr, "Can not get order value from input tranpose");
        output = rewriter.create<IE::GroupConvolutionOp>(
                                 groupConvOp->getLoc(), inTranspose.getOutput(), groupConvOp.getFilter(),
                                 groupConvOp.getBias(), groupConvOp.getStridesAttr(), groupConvOp.getPadsBeginAttr(),
                                 groupConvOp.getPadsEndAttr(), groupConvOp.getDilationsAttr(),
                                 groupConvOp.getGroupsAttr(), groupConvOp.getPostOpAttr(), groupConvOp.getClampAttr(),
                                 groupConvOp.getOutputChannelsAttr(), groupConvOp.getInputChannelsAttr())
                         .getOutput();
    } else {
        auto mapper = mapOperands(origOp, transPermAttr, rewriter);
        VPUX_THROW_WHEN(transPermAttr == nullptr, "Can not get order value from input tranpose");
        auto* newOp = rewriter.clone(*origOp.getOperation(), mapper);
        vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);
        output = newOp->getResult(0);
    }
    // Support mixed precision convolution i8 -> fp16
    // In this case, the inferred type has become i8, and we have to set it back to fp16
    auto elemType = origOp.getOutput().getType().template dyn_cast<vpux::NDTypeInterface>().getElementType();
    auto outType = output.getType().template dyn_cast<vpux::NDTypeInterface>();
    output.setType(mlir::cast<mlir::RankedTensorType>(outType.changeElemType(elemType)));

    _log.trace("Insert new layer without batch: {0}", output);
    auto outTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, output, nullptr, transPermAttr);
    _log.trace("Insert transpose {0} for output", outTranspose);

    return mlir::success();
}

template <class ConcreteOp>
mlir::IRMapping LayerConverter<ConcreteOp>::mapOperands(ConcreteOp origOp, mlir::AffineMapAttr& orderAttr,
                                                        mlir::PatternRewriter& rewriter) const {
    mlir::IRMapping mapper;
    SmallVector<mlir::Value> inputVals;
    mlir::Operation* op = origOp;
    for (auto input : origOp.getInputs() | indexed) {
        if (!op->hasTrait<IE::EltwiseOp>() && input.index() > 0) {
            continue;
        }
        inputVals.push_back(input.value());
    }
    auto dim = getDimEqualsToOne(inputVals).value();
    for (auto input : inputVals) {
        auto inTranspose = createTransposeForLayerInput(rewriter, input, dim, origOp->getLoc());
        orderAttr = orderAttr == nullptr ? inTranspose.getOrderValueAttr() : orderAttr;
        _log.trace("Insert transpose for input: {0}", inTranspose);
        mapper.map(input, inTranspose.getOutput());
    }
    return mapper;
}

template <class ConcreteOp>
bool isLegalGeneric(ConcreteOp op) {
    auto hasPerAxisQuantization = isPerAxisQuant(op.getInput()) || isPerAxisQuant(op.getOutput());
    if (!isShapeRankEq4(op.getInput()) || isEqualToOne(op.getInput(), Dims4D::Act::N) || hasPerAxisQuantization) {
        return true;
    }
    return false;
}

template <class ConcreteOp>
bool isLegalConvOp(ConcreteOp op) {
    if (isLegalGeneric(op)) {
        return true;
    }
    auto transposedDim = getDimEqualsToOne({op.getInput()});
    return !transposedDim.has_value() || !isEqualToOne(op.getFilter(), transposedDim.value());
}

template <class ConcreteOp>
bool isLegalPoolOp(ConcreteOp op) {
    if (isLegalGeneric(op)) {
        return true;
    }
    auto transposedDim = getDimEqualsToOne({op.getInput()});
    if (!transposedDim.has_value()) {
        return true;
    }
    const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSize());
    if (transposedDim.value() == Dims4D::Act::H) {
        return kernelSize[Dims4D::Kernel::Y.ind()] != 1;
    } else if (transposedDim.value() == Dims4D::Act::W) {
        return kernelSize[Dims4D::Kernel::X.ind()] != 1;
    }
    return true;
}

bool isLegalGroupConvOp(IE::GroupConvolutionOp op) {
    if (isLegalGeneric(op)) {
        return true;
    }
    auto transposedDim = getDimEqualsToOne({op.getInput()});
    return (!transposedDim.has_value() || !isEqualToOne(op.getFilter(), transposedDim.value())) &&
           !isEqualToOne(op.getInput(), Dims4D::Act::C);
}

//
// SigmoidConverter
//

class SigmoidConverter final : public mlir::OpRewritePattern<IE::SigmoidOp> {
public:
    SigmoidConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SigmoidOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SigmoidOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

IE::AffineReshapeOp buildAffineReshape(mlir::Location loc, mlir::Value input, ArrayRef<int64_t> targetShape,
                                       mlir::PatternRewriter& rewriter, IE::SigmoidOp replacedOp) {
    const auto ctx = rewriter.getContext();
    const auto srcType = input.getType().cast<vpux::NDTypeInterface>();
    const auto dstType = srcType.changeShape(ShapeRef(targetShape));

    SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind(), Dims4D::Act::C.ind()},
                                                   {Dims4D::Act::C.ind()},
                                                   {Dims4D::Act::H.ind()},
                                                   {Dims4D::Act::W.ind()}};
    SmallVector<SmallVector<int64_t>> outDimMapping{{Dims4D::Act::N.ind()},
                                                    {Dims4D::Act::N.ind(), Dims4D::Act::C.ind()},
                                                    {Dims4D::Act::H.ind()},
                                                    {Dims4D::Act::W.ind()}};
    auto dimMapping = replacedOp == nullptr ? inDimMapping : outDimMapping;

    auto reshapeOp = replacedOp == nullptr ? rewriter.create<IE::AffineReshapeOp>(loc, dstType, input,
                                                                                  getIntArrayOfArray(ctx, dimMapping),
                                                                                  getIntArrayAttr(ctx, targetShape))
                                           : rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
                                                     replacedOp, dstType, input, getIntArrayOfArray(ctx, dimMapping),
                                                     getIntArrayAttr(ctx, targetShape));

    return reshapeOp;
}

IE::AffineReshapeOp affineReshapeInput(mlir::Location loc, mlir::Value input, ShapeRef origShape,
                                       mlir::PatternRewriter& rewriter) {
    Shape targetShape(origShape.raw());

    targetShape[Dims4D::Act::C] *= targetShape[Dims4D::Act::N];
    targetShape[Dims4D::Act::N] = 1;

    const auto reshapedLoc = appendLoc(loc, "reshape input for Sigmoid");
    return buildAffineReshape(reshapedLoc, input, ArrayRef(targetShape.raw()), rewriter, nullptr);
}

IE::AffineReshapeOp affineReshapeOutput(IE::SigmoidOp origOp, mlir::Value sigmoidOutput,
                                        mlir::PatternRewriter& rewriter) {
    const Shape origShape = getShape(origOp.getOutput()).toValues();
    const SmallVector<int64_t> targetShape = origShape.raw();

    const auto reshapedLoc = appendLoc(origOp.getLoc(), "reshape output for Sigmoid");
    return buildAffineReshape(reshapedLoc, sigmoidOutput, ArrayRef(targetShape), rewriter, origOp);
}

mlir::LogicalResult SigmoidConverter::matchAndRewrite(IE::SigmoidOp origOp, mlir::PatternRewriter& rewriter) const {
    // Create affineReshapeOp for input
    const auto sigmoidInShape = getShape(origOp.getInput());
    auto reshapeIn = affineReshapeInput(origOp.getLoc(), origOp.getInput(), sigmoidInShape, rewriter);

    // Create new sigmoidOp
    auto newSigmoidOp = rewriter.create<IE::SigmoidOp>(origOp.getLoc(), reshapeIn);

    // Create affineReshapeOp for output
    affineReshapeOutput(origOp, newSigmoidOp.getOutput(), rewriter);

    return mlir::success();
}

//
// ConvertBatchedLayerTo1NPass
//

class ConvertBatchedLayerTo1NPass final : public IE::ConvertBatchedLayerTo1NBase<ConvertBatchedLayerTo1NPass> {
public:
    explicit ConvertBatchedLayerTo1NPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertBatchedLayerTo1NPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) -> bool {
        return isLegalConvOp(op);
    });

    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) -> bool {
        return isLegalGroupConvOp(op);
    });

    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) -> bool {
        return isLegalPoolOp(op);
    });

    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) -> bool {
        return isLegalPoolOp(op);
    });

    auto isLegalEltwiseOp = [&](mlir::Operation* op) -> bool {
        if (op->getNumOperands() < 2) {
            return true;
        }
        auto hasPerAxisQuantization = isPerAxisQuant(op->getOperand(0)) || isPerAxisQuant(op->getOperand(1)) ||
                                      isPerAxisQuant(op->getResult(0));
        auto inShape1 = getShape(op->getOperand(0));
        auto inShape2 = getShape(op->getOperand(1));
        if (!isShapeRankEq4(op->getOperand(1)) || isEqualToOne(op->getOperand(0), Dims4D::Act::N) ||
            !isShapeRankEq4(op->getOperand(0)) || inShape1[Dims4D::Act::N] != inShape2[Dims4D::Act::N] ||
            hasPerAxisQuantization) {
            return true;
        }
        return false;
    };
    target.addDynamicallyLegalOp<IE::AddOp>(isLegalEltwiseOp);
    target.addDynamicallyLegalOp<IE::MultiplyOp>(isLegalEltwiseOp);

    target.addDynamicallyLegalOp<IE::SigmoidOp>([&](IE::SigmoidOp op) -> bool {
        auto hasPerAxisQuantization = isPerAxisQuant(op.getInput()) || isPerAxisQuant(op.getOutput());
        if (!isShapeRankEq4(op.getInput()) || isEqualToOne(op.getInput(), Dims4D::Act::N) || hasPerAxisQuantization) {
            return true;
        }
        return false;
    });

    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ShapeCastOp>();
    target.addLegalOp<IE::BroadcastOp>();
    target.addLegalOp<IE::AffineReshapeOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerConverter<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::AddOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<SigmoidConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBatchedLayerTo1NPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBatchedLayerTo1NPass(Logger log) {
    return std::make_unique<ConvertBatchedLayerTo1NPass>(log);
}

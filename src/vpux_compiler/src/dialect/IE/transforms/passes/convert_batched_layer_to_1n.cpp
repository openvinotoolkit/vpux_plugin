//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
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

void reshapeForEltwiseAdd(IE::AddOp addOp, mlir::PatternRewriter& rewriter) {
    const auto ctx = addOp->getContext();
    auto inputShape1 = getShape(addOp.getInput1());
    auto inputShape2 = getShape(addOp.getInput2());
    auto outputShape = getShape(addOp.getOutput());

    Shape newInShape1 = Shape(inputShape1);
    newInShape1[Dims4D::Act::N] = 1;
    newInShape1[Dims4D::Act::C] = inputShape1[Dims4D::Act::C] * inputShape1[Dims4D::Act::N];
    Shape newInShape2 = Shape(inputShape2);
    newInShape2[Dims4D::Act::N] = 1;
    newInShape2[Dims4D::Act::C] = inputShape2[Dims4D::Act::C] * inputShape2[Dims4D::Act::N];

    auto inShapeCast1 =
            rewriter.create<IE::ShapeCastOp>(addOp->getLoc(), addOp.getInput1(), getIntArrayAttr(ctx, newInShape1));
    auto inShapeCast2 =
            rewriter.create<IE::ShapeCastOp>(addOp->getLoc(), addOp.getInput2(), getIntArrayAttr(ctx, newInShape2));

    auto newAddOp =
            rewriter.create<IE::AddOp>(addOp->getLoc(), inShapeCast1.getResult(), inShapeCast2.getResult(),
                                       addOp.getAutoBroadcastAttr(), addOp.getPostOpAttr(), addOp.getClampAttr());

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(addOp, newAddOp.getOutput(), getIntArrayAttr(ctx, outputShape));
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
        reshapeForEltwiseAdd(addOp, rewriter);
        _log.trace("Reshape directly for not suitable transpose cases");
        return mlir::success();
    }

    mlir::AffineMapAttr transPermAttr = nullptr;
    auto mapper = mapOperands(origOp, transPermAttr, rewriter);
    VPUX_THROW_WHEN(transPermAttr == nullptr, "Can not get order value from input tranpose");
    auto* newOp = rewriter.clone(*origOp.getOperation(), mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    // Support mixed precision convolution i8 -> fp16
    // In this case, the inferred type has become i8, and we have to set it back to fp16
    auto elemType = origOp.getOutput().getType().template dyn_cast<vpux::NDTypeInterface>().getElementType();
    auto output = newOp->getResult(0);
    auto outType = output.getType().template dyn_cast<vpux::NDTypeInterface>();
    output.setType(mlir::cast<mlir::RankedTensorType>(outType.changeElemType(elemType)));
    _log.trace("Insert new layer without batch: {0}", newOp);
    auto outTranspose =
            rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, newOp->getResult(0), nullptr, transPermAttr);
    _log.trace("Insert transpose {0} for output", outTranspose);

    return mlir::success();
}

template <class ConcreteOp>
mlir::IRMapping LayerConverter<ConcreteOp>::mapOperands(ConcreteOp origOp, mlir::AffineMapAttr& orderAttr,
                                                        mlir::PatternRewriter& rewriter) const {
    mlir::IRMapping mapper;
    auto dim = getDimEqualsToOne({origOp.getInput()}).value();
    auto inTranspose = createTransposeForLayerInput(rewriter, origOp.getInput(), dim, origOp->getLoc());
    orderAttr = inTranspose.getOrderValueAttr();
    _log.trace("Insert transpose for input: {0}", inTranspose);
    mapper.map(origOp.getInput(), inTranspose.getOutput());
    return mapper;
}

template <>
mlir::IRMapping LayerConverter<IE::AddOp>::mapOperands(IE::AddOp origOp, mlir::AffineMapAttr& orderAttr,
                                                       mlir::PatternRewriter& rewriter) const {
    mlir::IRMapping mapper;
    auto dim = getDimEqualsToOne({origOp.getInput1(), origOp.getInput2()}).value();

    auto inTranspose1 = createTransposeForLayerInput(rewriter, origOp.getInput1(), dim, origOp->getLoc());
    _log.trace("Insert transpose for input1: {0}", inTranspose1);
    mapper.map(origOp.getInput1(), inTranspose1.getOutput());
    orderAttr = inTranspose1.getOrderValueAttr();

    auto inTranspose2 = createTransposeForLayerInput(rewriter, origOp.getInput2(), dim, origOp->getLoc());
    _log.trace("Insert transpose for input2: {0}", inTranspose2);
    mapper.map(origOp.getInput2(), inTranspose2.getOutput());
    return mapper;
}

template <class ConcreteOp>
bool isLegalConvOp(ConcreteOp op) {
    auto hasPerAxisQuantization = isPerAxisQuant(op.getInput()) || isPerAxisQuant(op.getOutput());
    if (!isShapeRankEq4(op.getInput()) || isEqualToOne(op.getInput(), Dims4D::Act::N) || hasPerAxisQuantization) {
        return true;
    }
    auto transposedDim = getDimEqualsToOne({op.getInput()});
    return !transposedDim.has_value() || !isEqualToOne(op.getFilter(), transposedDim.value());
}

template <class ConcreteOp>
bool isLegalPoolOp(ConcreteOp op) {
    auto hasPerAxisQuantization = isPerAxisQuant(op.getInput()) || isPerAxisQuant(op.getOutput());
    if (!isShapeRankEq4(op.getInput()) || isEqualToOne(op.getInput(), Dims4D::Act::N) || hasPerAxisQuantization) {
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
        return isLegalConvOp(op);
    });

    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) -> bool {
        return isLegalPoolOp(op);
    });

    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) -> bool {
        return isLegalPoolOp(op);
    });

    target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) -> bool {
        auto hasPerAxisQuantization =
                isPerAxisQuant(op.getInput1()) || isPerAxisQuant(op.getInput2()) || isPerAxisQuant(op.getOutput());
        auto inShape1 = getShape(op.getInput1());
        auto inShape2 = getShape(op.getInput2());
        if (!isShapeRankEq4(op.getInput2()) || isEqualToOne(op.getInput1(), Dims4D::Act::N) ||
            !isShapeRankEq4(op.getInput1()) || inShape1[Dims4D::Act::N] != inShape2[Dims4D::Act::N] ||
            hasPerAxisQuantization) {
            return true;
        }
        return false;
    });

    target.addDynamicallyLegalOp<IE::SigmoidOp>([&](IE::SigmoidOp op) -> bool {
        auto hasPerAxisQuantization = isPerAxisQuant(op.getInput()) || isPerAxisQuant(op.getOutput());
        if (!isShapeRankEq4(op.getInput()) || isEqualToOne(op.getInput(), Dims4D::Act::N) || hasPerAxisQuantization) {
            return true;
        }
        return false;
    });

    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ShapeCastOp>();
    target.addLegalOp<IE::AffineReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerConverter<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<LayerConverter<IE::AddOp>>(&ctx, _log);
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

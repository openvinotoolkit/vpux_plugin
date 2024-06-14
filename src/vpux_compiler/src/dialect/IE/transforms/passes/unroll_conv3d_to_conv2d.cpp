//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

auto createFQ(mlir::PatternRewriter& rewriter, mlir::Value input, IE::FakeQuantizeOp fq, int64_t index) {
    const auto sliceFqConstInput = [&](mlir::Value fqInput) {
        auto fqInputType = fqInput.getType().cast<NDTypeInterface>();
        const auto fqInputShape = fqInputType.getShape();
        auto newFqInputShape = to_small_vector(fqInputShape);
        Shape inputOffsets(fqInputShape.size(), 0);
        newFqInputShape[index] = 1;
        const auto newFqInputShapeAttr = getIntArrayAttr(rewriter.getContext(), newFqInputShape);
        const auto inputOffsetsAttr = getIntArrayAttr(rewriter.getContext(), inputOffsets);
        const auto inputSlice =
                rewriter.createOrFold<IE::SliceOp>(fq->getLoc(), fqInput, inputOffsetsAttr, newFqInputShapeAttr);
        newFqInputShape.erase(newFqInputShape.begin() + index);
        const auto newFqInputShapeSqueezedAttr = getIntArrayAttr(rewriter.getContext(), newFqInputShape);

        return rewriter.createOrFold<IE::ReshapeOp>(fq->getLoc(), inputSlice, nullptr, false,
                                                    newFqInputShapeSqueezedAttr);
    };
    auto inputLow = sliceFqConstInput(fq.getInputLow());
    auto inputHigh = sliceFqConstInput(fq.getInputHigh());
    auto outputLow = sliceFqConstInput(fq.getOutputLow());
    auto outputHigh = sliceFqConstInput(fq.getOutputHigh());
    return rewriter.create<IE::FakeQuantizeOp>(fq.getLoc(), input, inputLow, inputHigh, outputLow, outputHigh,
                                               fq.getLevelsAttr(), fq.getLowFpTypeAttr(), fq.getAutoBroadcast());
}

SmallVector<mlir::Value> getSlicedFilters(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, mlir::Value input,
                                          ShapeRef filterShape, Logger log) {
    SmallVector<mlir::Value> slicedFilters;
    const auto IC = filterShape[Dims5D::Filter::IC];
    const auto OC = filterShape[Dims5D::Filter::OC];
    const auto kernelZ = filterShape[Dims5D::Filter::KZ];
    const auto kernelY = filterShape[Dims5D::Filter::KY];
    const auto kernelX = filterShape[Dims5D::Filter::KX];
    const auto subFilterSize = IC * OC * kernelY * kernelX;
    const Shape outputWeightShape = {OC, IC, kernelY, kernelX};

    for (int64_t kz = 0; kz < kernelZ; kz++) {
        Shape offsets(filterShape.size());
        offsets[Dims5D::Filter::IC] = IC;
        offsets[Dims5D::Filter::OC] = OC;
        offsets[Dims5D::Filter::KZ] = kz;
        offsets[Dims5D::Filter::KX] = kernelX;
        offsets[Dims5D::Filter::KY] = kernelY;

        auto weightsCst = input.getDefiningOp<Const::DeclareOp>();
        auto weightsFQ = input.getDefiningOp<IE::FakeQuantizeOp>();
        if (weightsFQ != nullptr) {
            weightsCst = weightsFQ.getInput().getDefiningOp<Const::DeclareOp>();
        }
        auto weightsCstContent = weightsCst.getContent();
        auto contentValue = weightsCstContent.getValues<vpux::type::float16>();
        std::vector<vpux::type::float16> subWeights(subFilterSize, 0.0f);

        for (auto indexOC = 0; indexOC < OC; indexOC++) {
            for (auto indexIC = 0; indexIC < IC; indexIC++) {
                for (auto indexKY = 0; indexKY < kernelY; indexKY++) {
                    for (auto indexKX = 0; indexKX < kernelX; indexKX++) {
                        auto subIndex = indexKX + indexKY * kernelX + indexIC * kernelY * kernelX +
                                        indexOC * IC * kernelY * kernelX;
                        auto origIndex = indexKX + indexKY * kernelX + kz * kernelX * kernelY +
                                         indexIC * kernelZ * kernelY * kernelX +
                                         indexOC * IC * kernelZ * kernelY * kernelX;
                        subWeights[subIndex] = (vpux::type::float16)contentValue[origIndex];
                    }
                }
            }
        }

        const auto elemType = input.getType().cast<vpux::NDTypeInterface>().getElementType();
        const auto dataStorageType = mlir::RankedTensorType::get(outputWeightShape.raw(), elemType);
        const auto newContentAttr = mlir::DenseElementsAttr::get(dataStorageType, ArrayRef(subWeights));
        auto newConstInput = rewriter.create<Const::DeclareOp>(origOp->getLoc(), dataStorageType,
                                                               Const::ContentAttr::get(newContentAttr))
                                     .getOutput();
        if (weightsFQ != nullptr) {
            newConstInput = createFQ(rewriter, newConstInput, weightsFQ, Dims5D::Filter::KZ.ind());
        }
        slicedFilters.push_back(newConstInput);
    }
    log.trace("Sliced filters size: '{0}'.", slicedFilters.size());
    return slicedFilters;
}

//
// ConvGeneralAggregation
//
template <class ConcreteOp>
class ConvGeneralAggregation final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvGeneralAggregation(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ConvGeneralAggregation<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Convert NCE to 4D for '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto* ctx = origOp->getContext();

    auto spatialDimKernelIndex = 2;

    const auto input = origOp->getOperand(0);
    const auto filter = origOp.getFilter();
    // Reduce shape over spatial dims with kernel 1
    const auto inputShape = input.getType().template cast<vpux::NDTypeInterface>().getShape();
    const auto filterShape = filter.getType().template cast<vpux::NDTypeInterface>().getShape();
    if (inputShape.size() != 5) {
        return mlir::failure();
    }

    auto newInputShape = to_small_vector(inputShape);
    auto newFilterShape = to_small_vector(filterShape);

    auto newStrides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr());
    auto newPadsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBeginAttr());
    auto newPadsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEndAttr());
    auto newDilations = parseIntArrayAttr<int64_t>(origOp.getDilationsAttr());
    for (auto kernelIt = newFilterShape.begin() + spatialDimKernelIndex; kernelIt < newFilterShape.end() - 1;
         kernelIt++) {
        if (*kernelIt == 1 && *(kernelIt + 1) == 1) {
            auto kernelIndex = kernelIt - newFilterShape.begin();
            auto spatialIndex = kernelIndex - 2;
            auto arePadsZero = newPadsBegin[spatialIndex] == 0 && newPadsBegin[spatialIndex + 1] == 0 &&
                               newPadsEnd[spatialIndex] == 0 && newPadsEnd[spatialIndex + 1] == 0;
            auto isStrideSame = newStrides[spatialIndex] == newStrides[spatialIndex + 1];
            auto isDilationSame = newDilations[spatialIndex] == newDilations[spatialIndex + 1];
            if (arePadsZero && isStrideSame && isDilationSame) {
                newFilterShape.erase(kernelIt + 1);
                newInputShape[kernelIndex] *= newInputShape[kernelIndex + 1];
                newInputShape.erase(newInputShape.begin() + kernelIndex + 1);
                newStrides.erase(newStrides.begin() + spatialIndex + 1);
                newPadsBegin.erase(newPadsBegin.begin() + spatialIndex + 1);
                newPadsEnd.erase(newPadsEnd.begin() + spatialIndex + 1);
                newDilations.erase(newDilations.begin() + spatialIndex + 1);
            }
        }
    }

    if (newInputShape.size() != 4) {
        return mlir::failure();
    }

    const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), newInputShape);
    auto newInput = rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), input, nullptr, false, newInputShapeAttr);

    const auto newFilterShapeAttr = getIntArrayAttr(rewriter.getContext(), newFilterShape);
    auto newFilter = rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), filter, nullptr, false, newFilterShapeAttr);

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector<mlir::Value>{newInput, newFilter});
    auto* newConvOp = rewriter.clone(*origOp.getOperation(), mapper);

    VPUX_THROW_UNLESS(newConvOp->hasAttr("pads_begin") && newConvOp->hasAttr("pads_end") &&
                              newConvOp->hasAttr("strides") && newConvOp->hasAttr("dilations"),
                      "Cannot get all attributions");
    newConvOp->setAttr("pads_begin", getIntArrayAttr(rewriter.getContext(), newPadsBegin));
    newConvOp->setAttr("pads_end", getIntArrayAttr(rewriter.getContext(), newPadsEnd));
    newConvOp->setAttr("strides", getIntArrayAttr(rewriter.getContext(), newStrides));
    newConvOp->setAttr("dilations", getIntArrayAttr(rewriter.getContext(), newDilations));

    vpux::inferReturnTypes(newConvOp, vpux::InferShapedTypeMode::ALL);

    const auto outputType = origOp.getOutput().getType().template dyn_cast<vpux::NDTypeInterface>();
    const auto outputShapeAttr = getIntArrayAttr(ctx, outputType.getShape());
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConvOp->getResult(0), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 4D '{0}'", origOp->getName());
    return mlir::success();
}

// This pass unrolls 3D convolution to a combination of 2D convolutions.
// The detail steps :
// 1. slice the filter according to the Z value of 3D filter.
// 2. slice the activations by depth in output shape and Z value of 3D filter.
// 3. add the new convolution one by one
// 4. concat the add results in depths
//
//  [act]        [w]        [act]       [w]     [act]        [w]        ...       [act]       [w]     [act]        [w]
//    |           |    to     |          |        |           |         ...         |          |        |           |
//  -(convolution3D)-       (slice)    (slice)  (slice)     (slice)     ...      (slice)    (slice)  (slice)     (slice)
//                            |          |        |           |         ...         |          |        |           |
//                              -(conv)-            -(conv)-            ...           -(conv)-            -(conv)-
//                                  |                  |                ...              |                  |
//                                    -- (eltwise) --                   ...                 -- (eltwise) --
//                                          |                           ...                        |
//                                             -----------------------(concat)--------------------

//
// ConvGeneralRewriter
//

template <class ConcreteOp>
class ConvGeneralRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvGeneralRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ConvGeneralRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa_and_nonnull<IE::ConvolutionOp, IE::GroupConvolutionOp>(origOp.getOperation())) {
        return matchFailed(rewriter, origOp, "Unroll 3D supports only Convolution and GroupConvolution");
    }
    _log.trace("Got layer '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    const auto dilations = Shape(parseIntArrayAttr<int64_t>(origOp.getDilations()));
    const auto filterShape = getShape(origOp.getFilter());

    const auto kernelZ = filterShape[Dims5D::Filter::KZ];
    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));
    const auto strides = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
    const auto stridesZ = strides[Dims5D::Strides::Z];
    const auto inputShape = getShape(origOp->getOperand(0));
    const auto outputShape = getShape(origOp->getResult(0));

    const auto padFront = padStart[Dims5D::PadsBegin::Front];
    mlir::MLIRContext* ctx = origOp->getContext();

    // 1. slice Filters
    SmallVector<mlir::Value> slicedFilters = getSlicedFilters(rewriter, origOp, origOp.getFilter(), filterShape, _log);

    // 2. slice activation and create new convolution
    SmallVector<mlir::Value> newConvs;
    auto input = origOp.getInput();
    const auto inputFQ = origOp->getOperand(0).template getDefiningOp<IE::FakeQuantizeOp>();

    for (int64_t actIndex = 0; actIndex < outputShape[Dims5D::Act::D]; actIndex++) {
        SmallVector<mlir::Value> newSubConvs;
        for (int64_t depthIndex = 0; depthIndex < kernelZ; depthIndex++) {
            // Calculate the activation Depth index
            auto actDepthIndex = actIndex * stridesZ + depthIndex - padFront;
            if (actDepthIndex < 0 || actDepthIndex > inputShape[Dims5D::Act::D] - 1) {
                // For padding at begin and end, do not add subconvolution.
                continue;
            }

            Shape offsets(inputShape.size(), 0);
            offsets[Dims5D::Act::D] = actDepthIndex;
            SmallVector<int64_t> sliceShape{inputShape[Dims5D::Act::N], inputShape[Dims5D::Act::C], 1,
                                            inputShape[Dims5D::Act::H], inputShape[Dims5D::Act::W]};
            auto slicedActivation =
                    rewriter.create<IE::SliceOp>(origOp->getLoc(), input, getIntArrayAttr(ctx, offsets.raw()),
                                                 getIntArrayAttr(ctx, sliceShape))
                            .getResult();
            SmallVector<int64_t> reshapeShape{inputShape[Dims5D::Act::N], inputShape[Dims5D::Act::C],
                                              inputShape[Dims5D::Act::H], inputShape[Dims5D::Act::W]};
            auto reshapeSlicedActivation = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), slicedActivation, nullptr,
                                                                          false, getIntArrayAttr(ctx, reshapeShape))
                                                   .getOutput();

            mlir::Builder builder(origOp->getContext());
            auto stridesAttr = builder.getI64ArrayAttr({strides[Dims5D::Strides::Y], strides[Dims5D::Strides::X]});
            auto padBeginAttr =
                    builder.getI64ArrayAttr({padStart[Dims5D::PadsBegin::Top], padStart[Dims5D::PadsBegin::Left]});
            auto padEndAttr =
                    builder.getI64ArrayAttr({padEnd[Dims5D::PadsEnd::Bottom], padEnd[Dims5D::PadsEnd::Right]});
            auto dilationsAttr =
                    builder.getI64ArrayAttr({dilations[Dims5D::Strides::Y], dilations[Dims5D::Strides::X]});
            mlir::Operation* newConvOp;
            if (auto groupConv = mlir::dyn_cast_or_null<IE::GroupConvolutionOp>(origOp.getOperation())) {
                if (inputFQ != nullptr) {
                    auto newFakeQuantizeOp = createFQ(rewriter, reshapeSlicedActivation, inputFQ, Dims5D::Act::D.ind());
                    newConvOp = rewriter.create<IE::GroupConvolutionOp>(
                            origOp.getLoc(), newFakeQuantizeOp->getResult(0), slicedFilters[depthIndex],
                            /*bias=*/nullptr, stridesAttr, padBeginAttr, padEndAttr, dilationsAttr,
                            groupConv.getGroupsAttr(),
                            /*post_opAttr=*/nullptr, /*clamp=*/nullptr);
                } else {
                    newConvOp = rewriter.create<IE::GroupConvolutionOp>(
                            origOp.getLoc(), reshapeSlicedActivation, slicedFilters[depthIndex], /*bias=*/nullptr,
                            stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupConv.getGroupsAttr(),
                            /*post_opAttr=*/nullptr, /*clamp=*/nullptr);
                }
            } else {
                auto conv = mlir::cast<IE::ConvolutionOp>(origOp);
                if (inputFQ != nullptr) {
                    auto newFakeQuantizeOp = createFQ(rewriter, reshapeSlicedActivation, inputFQ, Dims5D::Act::D.ind());
                    newConvOp = rewriter.create<IE::ConvolutionOp>(
                            origOp.getLoc(), newFakeQuantizeOp->getResult(0), slicedFilters[depthIndex],
                            /*bias=*/nullptr, stridesAttr, padBeginAttr, padEndAttr, dilationsAttr,
                            /*post_opAttr=*/nullptr, /*clamp=*/nullptr, conv.getStaticScaleAttr());

                } else {
                    newConvOp = rewriter.create<IE::ConvolutionOp>(
                            origOp.getLoc(), reshapeSlicedActivation, slicedFilters[depthIndex], /*bias=*/nullptr,
                            stridesAttr, padBeginAttr, padEndAttr, dilationsAttr,
                            /*post_opAttr=*/nullptr, /*clamp=*/nullptr, conv.getStaticScaleAttr());
                }
            }

            newSubConvs.push_back(newConvOp->getResult(0));
        }
        if (newSubConvs.empty()) {
            _log.trace("No sub convolution generated.");
            continue;
        }
        if (newSubConvs.size() > 1) {
            const auto broadcastType =
                    vpux::IE::AutoBroadcastTypeAttr::get(origOp->getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
            mlir::Value add = newSubConvs.front();
            for (size_t i = 1; i < newSubConvs.size(); i++) {
                add = rewriter.create<IE::AddOp>(origOp->getLoc(), add, newSubConvs[i], broadcastType,
                                                 (i == newSubConvs.size() - 1) ? origOp.getClampAttr(),
                                                 origOp.getPostOpAttr() : nullptr, nullptr)
                              ->getResult(0);
            }

            newConvs.push_back(add);

        } else {
            newConvs.push_back(newSubConvs.front());
        }
    }

    // 3. add the new convolution one by one
    if (newConvs.empty()) {
        return matchFailed(rewriter, origOp, "no any new conv created.");
    }

    SmallVector<mlir::Value> concatInputs;
    SmallVector<int64_t> subOutputShape{outputShape[Dims5D::Act::N], outputShape[Dims5D::Act::C], 1,
                                        outputShape[Dims5D::Act::H] * outputShape[Dims5D::Act::W]};
    for (auto subConv : newConvs) {
        auto subOutputReshape = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), subConv, nullptr, false,
                                                               getIntArrayAttr(ctx, subOutputShape))
                                        .getOutput();
        concatInputs.push_back(subOutputReshape);
    }
    auto concatOutput = rewriter.create<IE::ConcatOp>(origOp->getLoc(), concatInputs, Dims4D::Act::H).getOutput();
    auto outputReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), concatOutput, nullptr, false,
                                                              getIntArrayAttr(ctx, outputShape));
    rewriter.replaceOp(origOp, outputReshape);

    return mlir::success();
}

class UnrollConv3dToConv2dPass final : public IE::UnrollConv3dToConv2dBase<UnrollConv3dToConv2dPass> {
public:
    explicit UnrollConv3dToConv2dPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollConv3dToConv2dPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalNceOp = [&](mlir::Operation* op) {
        const auto inputShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape();
        return inputShape.size() != 5;
    };

    const auto isLegalGroupConvOp = [&](IE::GroupConvolutionOp groupConv) {
        const auto inputShape = groupConv.getFilter().getType().cast<vpux::NDTypeInterface>().getShape();
        const auto hasGroups = groupConv.getGroups().has_value() ? 1 : 0;
        return (inputShape.size() + hasGroups) != 6;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalNceOp);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalGroupConvOp);

    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::ExpandDilatedOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvGeneralRewriter<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralRewriter<IE::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralAggregation<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralAggregation<IE::GroupConvolutionOp>>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollConv3dToConv2dPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollConv3dToConv2dPass(Logger log) {
    return std::make_unique<UnrollConv3dToConv2dPass>(log);
}

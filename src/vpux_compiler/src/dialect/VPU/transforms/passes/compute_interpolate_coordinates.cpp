//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/utils/core/error.hpp>
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <utility>

using namespace vpux;

namespace {

const float zeroFive = 0.5;

class ComputeInterpolateCoordinates final : public mlir::OpRewritePattern<VPU::InterpolateOp> {
public:
    ComputeInterpolateCoordinates(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::InterpolateOp>(ctx), _log(std::move(log)) {
        setDebugName("ComputeInterpolateCoordinates");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::InterpolateOp interpolateOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

float mapCoord(int x, float scale, int64_t lenghtResized, int64_t lenghtOriginal, IE::InterpolateCoordMode coordMode) {
    const auto xF = static_cast<float>(x);
    switch (coordMode) {
    case IE::InterpolateCoordMode::HALF_PIXEL: {
        return ((xF + zeroFive) / scale) - zeroFive;
    }
    case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL: {
        return (lenghtResized > 1) ? (xF + zeroFive) / scale - zeroFive : 0.0f;
    }
    case IE::InterpolateCoordMode::ASYMMETRIC: {
        return xF / scale;
    }
    case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN: {
        return (xF + zeroFive) / scale;
    }
    case IE::InterpolateCoordMode::ALIGN_CORNERS: {
        return (lenghtResized == 1)
                       ? 0
                       : xF * static_cast<float>(lenghtOriginal - 1) / static_cast<float>(lenghtResized - 1);
    }
    }
    return 0;
}

mlir::LogicalResult ComputeInterpolateCoordinates::matchAndRewrite(VPU::InterpolateOp interpolateOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    const auto coordinates = interpolateOp.getCoordinates();
    const auto lambdas = interpolateOp.getLambdas();

    if (coordinates != nullptr && lambdas != nullptr) {
        return mlir::failure();
    }

    const auto interpolateAttr = interpolateOp.getAttr();
    const auto interpolateMode = interpolateAttr.getMode().getValue();

    // Computing coordinates at compile time is a feature supported only for linear interpolate modes.
    // The ticket for adding support for all interpolate modes is E#129853.
    if (interpolateMode != IE::InterpolateMode::LINEAR && interpolateMode != IE::InterpolateMode::LINEAR_ONNX) {
        return mlir::failure();
    }

    const auto loc = interpolateOp.getLoc();
    const auto ctx = interpolateOp.getContext();

    const auto inType = interpolateOp.getInput().getType().cast<NDTypeInterface>();
    const auto inOrder = inType.getDimsOrder();
    const auto inShape = inType.getShape().raw();
    const auto outShape = interpolateOp.getOutput().getType().cast<NDTypeInterface>().getShape().raw();
    const auto coordMode = interpolateAttr.getCoordMode().getValue();

    const auto axesResult = IE::extractIntVector(loc, interpolateOp.getAxes(), interpolateOp.getAxesAttrAttr());
    VPUX_THROW_WHEN(mlir::failed(axesResult), "Failed to extract axes");
    const auto innermostAxisResult = IE::getInnermostAxis(loc, inOrder, axesResult.value());
    VPUX_THROW_WHEN(mlir::failed(innermostAxisResult), "Failed to get the innermost axis");
    const auto innermostAxis = innermostAxisResult.value();

    const auto parseIntArrayAttrOr = [](std::optional<mlir::ArrayAttr> attribute,
                                        const SmallVector<int64_t>& defaultValue) {
        return attribute.has_value() ? parseIntArrayAttr<int64_t>(attribute.value()) : defaultValue;
    };

    const auto initialInputShape =
            parseIntArrayAttrOr(interpolateOp.getInitialInputDimsAttr(), to_small_vector(inShape));
    const auto initialOutputShape =
            parseIntArrayAttrOr(interpolateOp.getInitialOutputDimsAttr(), to_small_vector(outShape));
    const auto initialInputOffset =
            parseIntArrayAttrOr(interpolateOp.getInitialInputOffsetAttr(), SmallVector<int64_t>(inShape.size(), 0));
    const auto initialOutputOffset =
            parseIntArrayAttrOr(interpolateOp.getInitialOutputOffsetAttr(), SmallVector<int64_t>(outShape.size(), 0));

    const auto innermostAxisInputDim = outShape[innermostAxis];
    const auto innermostAxisOuputDim = inShape[innermostAxis];
    const auto innermostAxisInitialInputDim = initialInputShape[innermostAxis];
    const auto innermostAxisInitialOutputDim = initialOutputShape[innermostAxis];
    const auto innermostAxisInputOffset = checked_cast<float>(initialInputOffset[innermostAxis]);
    const auto innermostAxisOutputOffset = checked_cast<int>(initialOutputOffset[innermostAxis]);
    const auto scale =
            static_cast<float>(innermostAxisInitialOutputDim) / static_cast<float>(innermostAxisInitialInputDim);

    const auto coordinatesSize = IE::getInterpCoordinatesSize(interpolateOp.getOutput(), innermostAxis);
    const auto lambdasSize = IE::getInterpLambdasSize(interpolateOp.getOutput(), innermostAxis);
    std::vector<int> coordinatesVec(coordinatesSize);
    std::vector<type::float16> lambdasVec(lambdasSize);

    const auto inMemStrides = inOrder.toLogicalOrder(inType.getMemStrides()).raw();
    const auto innermostAxisStride = inMemStrides[innermostAxis].to<Byte>().count();

    for (int i = 0; i < innermostAxisInputDim; i++) {
        float inCoord = mapCoord(i + innermostAxisOutputOffset, scale, innermostAxisInitialOutputDim,
                                 innermostAxisInitialInputDim, coordMode) -
                        innermostAxisInputOffset;
        inCoord = std::clamp(inCoord, 0.0f, static_cast<float>(innermostAxisOuputDim - 1));
        const auto inCoord1 = static_cast<int64_t>(inCoord);
        const auto inCoord2 = std::clamp(inCoord1 + 1, int64_t(0), innermostAxisOuputDim - 1);

        auto lambda1 = inCoord - static_cast<float>(inCoord1);
        auto lambda2 = 1.f - lambda1;

        if (interpolateMode == IE::InterpolateMode::LINEAR_ONNX && inCoord1 == inCoord2) {
            lambda1 = lambda2 = zeroFive;
        }

        coordinatesVec[i] = checked_cast<int>(inCoord1 * innermostAxisStride);
        lambdasVec[i * 2] = lambda1;
        lambdasVec[i * 2 + 1] = lambda2;
    }

    const auto coordinatesConstType = mlir::RankedTensorType::get({1, 1, 1, coordinatesSize}, getSInt32Type(ctx));
    const auto coordinatesConst = Const::createConst(rewriter, loc, coordinatesConstType, ArrayRef(coordinatesVec));

    const auto lambdasConstType = mlir::RankedTensorType::get({1, 1, 1, lambdasSize}, mlir::Float16Type::get(ctx));
    const auto lambdasConst = Const::createConst(rewriter, loc, lambdasConstType, ArrayRef(lambdasVec));

    rewriter.replaceOpWithNewOp<VPU::InterpolateOp>(
            interpolateOp, interpolateOp.getInput(), interpolateOp.getSizes(), interpolateOp.getScales(),
            interpolateOp.getAxes(), coordinatesConst, lambdasConst, interpolateOp.getSizesAttrAttr(),
            interpolateOp.getScalesAttrAttr(), interpolateOp.getAxesAttrAttr(), interpolateOp.getTileOffsetAttrAttr(),
            interpolateOp.getInitialInputDimsAttrAttr(), interpolateOp.getInitialOutputDimsAttrAttr(),
            interpolateOp.getInitialInputOffsetAttrAttr(), interpolateOp.getInitialOutputOffsetAttrAttr(),
            interpolateOp.getMultiClusterStrategyAttr(), interpolateOp.getAttrAttr());

    return mlir::success();
}

class ComputeInterpolateCoordinatesPass final :
        public VPU::ComputeInterpolateCoordinatesBase<ComputeInterpolateCoordinatesPass> {
public:
    explicit ComputeInterpolateCoordinatesPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ComputeInterpolateCoordinatesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet greedyPatterns(&ctx);
    greedyPatterns.add<ComputeInterpolateCoordinates>(&ctx, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(greedyPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createComputeInterpolateCoordinatesPass(Logger log) {
    return std::make_unique<ComputeInterpolateCoordinatesPass>(log);
}

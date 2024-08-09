//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Support/LogicalResult.h>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace {

//
// OptimizeGroupConvConcat
//

/*
    Convert below pattern:

        Root(1x16x144x144)             Weights(16x1x3x3)
       /               \                  /
      |                GroupConv(1x16x144x144)
       \                    /
        Concat(1x32x144x144)

    to:

        Root(1x16x144x144)             Weights(32x1x3x3)
                    \                     /
                    Convolution(1x32x144x144)

    When root tensor is ReLu or layer with fused ReLu post op
    below conversion is still appliable because applying ReLu for serveral times does not change the results

       ReLu(1x16x144x144)             Weights(16x1x3x3)
       /               \                  /
      |               GroupConv + ReLu(1x16x144x144)
      |                         |
       \                       /
          Concat(1x32x144x144)

    to:

       ReLu(1x16x144x144)             Weights(32x1x3x3)
                       \                  /
                      Conv + ReLu(1x32x144x144)
*/

constexpr size_t SUPPORTED_RANK = 4;

struct InputPattern {
    size_t index;
    IE::GroupConvolutionOp groupConv = nullptr;

    InputPattern(size_t id, IE::GroupConvolutionOp groupConvOp): index(id), groupConv(groupConvOp) {
    }
};

struct GroupConvParameters {
    bool isPopulated = false;
    int64_t kx{};
    int64_t ky{};
    SmallVector<int64_t> stride;
    Shape padBegin;
    Shape padEnd;
    SmallVector<int64_t> dilations;
    bool withBias{};
    bool withPostOp{};
    bool withClamp{};

    bool operator==(const GroupConvParameters& other) const {
        return std::tie(isPopulated, kx, ky, stride, padBegin, padEnd, dilations, withBias, withPostOp, withClamp) ==
               std::tie(other.isPopulated, other.kx, other.ky, other.stride, other.padBegin, other.padEnd,
                        other.dilations, other.withBias, other.withPostOp, other.withClamp);
    }
};

GroupConvParameters extractGroupConvParameters(IE::GroupConvolutionOp op) {
    GroupConvParameters parameters;

    auto filterShape = getShape(op.getFilter());
    parameters.kx = filterShape[Dims4D::Filter::KX];
    parameters.ky = filterShape[Dims4D::Filter::KY];

    parameters.stride = parseIntArrayAttr<int64_t>(op.getStrides());
    parameters.padBegin = Shape(parseIntArrayAttr<int64_t>(op.getPadsBegin()));
    parameters.padEnd = Shape(parseIntArrayAttr<int64_t>(op.getPadsEnd()));
    parameters.dilations = parseIntArrayAttr<int64_t>(op.getDilations());

    parameters.withBias = op.getBias() != nullptr;
    parameters.withPostOp = op.getPostOpAttr() != nullptr;
    parameters.withClamp = op.getClampAttr() != nullptr;

    parameters.isPopulated = true;

    return parameters;
}

class OptimizeGroupConvConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    OptimizeGroupConvConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(std::move(log)) {
        setDebugName("OptimizeGroupConvConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::FailureOr<SmallVector<InputPattern>> getValidConcatInputs(IE::ConcatOp concatOp,
                                                                    GroupConvParameters& parameters) const;

    bool hasReLUPostOp(mlir::Operation* op) const;
    bool isSupportedParameters(const GroupConvParameters& parameters, mlir::Operation* rootOp) const;

    mlir::Value convertGroupConvWeights(IE::GroupConvolutionOp groupConv, mlir::PatternRewriter& rewriter) const;

    mlir::Value createWeights(mlir::Value activation, const GroupConvParameters& parameters,
                              mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

bool OptimizeGroupConvConcat::hasReLUPostOp(mlir::Operation* op) const {
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op)) {
        const auto postOpName = layerWithPostOp.getPostOp();
        return postOpName.has_value() && postOpName.value().getStringRef() == IE::ReLUOp::getOperationName();
    }
    return false;
}

bool OptimizeGroupConvConcat::isSupportedParameters(const GroupConvParameters& parameters,
                                                    mlir::Operation* rootOp) const {
    auto isGreaterThanOne = [](int64_t size) {
        return size > 1;
    };
    auto hasNonOneDilation = llvm::any_of(parameters.dilations, isGreaterThanOne);
    if (hasNonOneDilation) {
        return false;
    }

    auto hasNonOneStride = llvm::any_of(parameters.stride, isGreaterThanOne);
    if (hasNonOneStride) {
        return false;
    }

    // Odd & even kernels have different weights structures, right now only odd kernel is supported
    if (parameters.kx != parameters.ky || (parameters.kx - 1) % 2 != 0) {
        return false;
    }

    if (parameters.padBegin[Dims4D::PadsBegin::Left] != parameters.padEnd[Dims4D::PadsEnd::Right] ||
        parameters.padBegin[Dims4D::PadsBegin::Top] != parameters.padEnd[Dims4D::PadsEnd::Bottom] ||
        parameters.kx / 2 != parameters.padBegin[Dims4D::PadsBegin::Left]) {
        return false;
    }

    if (parameters.withBias || parameters.withClamp) {
        return false;
    }

    // It can be supported when root tensor is a ReLU layer or layer with ReLU postOp.
    // The new Convolution with ReLU postOp can produce the identical result, because
    // multiple cascading ReLu ops can produce the same result with single ReLu.
    if (parameters.withPostOp && !hasReLUPostOp(rootOp) && !mlir::isa<IE::ReLUOp>(rootOp)) {
        return false;
    }

    return true;
}

bool isChannelAligned(ShapeRef shape, int64_t alignment) {
    if (shape.size() != SUPPORTED_RANK) {
        return false;
    }

    return shape[Dims4D::Act::C] % alignment == 0;
}

SmallVector<Dim> getConcatDims(ShapeRef inShape, ShapeRef outShape) {
    VPUX_THROW_UNLESS(inShape.size() == outShape.size(), "Got unexpected input and output shape");
    SmallVector<Dim> concatDims;
    auto ioShapes = zip(inShape, outShape);
    for (const auto& ioShape : ioShapes | indexed) {
        const auto inSize = std::get<0>(ioShape.value());
        const auto outSize = std::get<1>(ioShape.value());
        if (inSize != outSize) {
            concatDims.push_back(Dim(ioShape.index()));
        }
    }
    return concatDims;
}

mlir::FailureOr<SmallVector<InputPattern>> OptimizeGroupConvConcat::getValidConcatInputs(
        IE::ConcatOp concatOp, GroupConvParameters& parameters) const {
    auto inputNum = concatOp.getInputs().size();
    SmallVector<InputPattern> concatInputs;
    concatInputs.reserve(inputNum);

    const auto concatOutType = concatOp.getOutput().getType().cast<NDTypeInterface>();
    const auto concatOutShape = concatOutType.getShape();

    const auto alignment = VPU::NCEInvariant::getAlignment(concatOutType.getElementType());
    if (!isChannelAligned(concatOutShape, alignment)) {
        return mlir::failure();
    }

    // Inputs of Concat should have the same root
    mlir::Operation* maybeRootOp = nullptr;
    auto hasTheSameParentOp = [&maybeRootOp](mlir::Operation* parentOp) {
        if (maybeRootOp == nullptr) {
            maybeRootOp = parentOp;
            return true;
        }

        return maybeRootOp == parentOp;
    };

    // Inputs of Concat with GroupConvolution should have the same parameters
    GroupConvParameters prevParameters;
    auto hasTheSameParameters = [&prevParameters](const GroupConvParameters& currParameters) {
        if (!prevParameters.isPopulated) {
            prevParameters = currParameters;
            return true;
        }

        return prevParameters == currParameters;
    };

    for (const auto& input : concatOp.getInputs() | indexed) {
        auto inputIndex = input.index();
        auto inputValue = input.value();

        auto concatDims = getConcatDims(getShape(inputValue), concatOutShape);
        if (concatDims.size() != 1 || concatDims.front() != Dims4D::Act::C) {
            _log.trace("Unsupported concat dimension");
            return mlir::failure();
        }

        if (inputValue.hasOneUse()) {
            auto groupConv = inputValue.getDefiningOp<IE::GroupConvolutionOp>();
            if (groupConv == nullptr) {
                return mlir::failure();
            }

            if (groupConv.getFilter().getDefiningOp<Const::DeclareOp>() == nullptr) {
                return mlir::failure();
            }

            auto ic = getShape(groupConv.getInput())[Dims4D::Act::C];
            auto groupNum = groupConv.getGroups().value();
            if (groupNum != ic) {
                _log.trace("Only support DWConv for now");
                return mlir::failure();
            }

            auto currParameters = extractGroupConvParameters(groupConv);
            auto parentOp = groupConv.getInput().getDefiningOp();
            if (!isSupportedParameters(currParameters, parentOp)) {
                return mlir::failure();
            }

            if (!hasTheSameParameters(currParameters)) {
                return mlir::failure();
            }

            if (!hasTheSameParentOp(parentOp)) {
                return mlir::failure();
            }

            concatInputs.push_back(InputPattern(inputIndex, groupConv));
        } else if (std::distance(inputValue.getUsers().begin(), inputValue.getUsers().end()) >=
                   checked_cast<int64_t>(inputNum)) {
            auto parentOp = inputValue.getDefiningOp();
            if (!hasTheSameParentOp(parentOp)) {
                return mlir::failure();
            }

            concatInputs.push_back(InputPattern(inputIndex, nullptr));
        } else {
            _log.trace("[{0}] unknown input at '{1}'", concatOp->getName(), concatOp->getLoc());
            return mlir::failure();
        }
    }

    if (!prevParameters.isPopulated) {
        _log.trace("[{0}] No GroupConv is found at concat inputs '{1}'", concatOp->getName(), concatOp->getLoc());
        return mlir::failure();
    }

    if (maybeRootOp == nullptr || !isChannelAligned(getShape(maybeRootOp->getResult(0)), alignment)) {
        return mlir::failure();
    }

    parameters = std::move(prevParameters);
    return concatInputs;
}

// Create weights to convert GroupConvolution to Convolution
mlir::Value OptimizeGroupConvConcat::convertGroupConvWeights(IE::GroupConvolutionOp groupConv,
                                                             mlir::PatternRewriter& rewriter) const {
    auto origFilter = groupConv.getFilter();
    auto origFilterOp = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(origFilterOp == nullptr, "Unable to find filter constant operation");
    auto origContentAttr = origFilterOp.getContentAttr();
    auto origContent = origContentAttr.fold();
    auto origContentType = origContent.getType();

    const auto origFiliterShape = getShape(origFilter);
    const auto kx = origFiliterShape[Dims4D::Filter::KX];
    const auto ky = origFiliterShape[Dims4D::Filter::KY];

    const auto groupNum = groupConv.getGroups().value();
    const auto groupInSize = origFiliterShape[Dims4D::Filter::IC];
    const auto groupOutSize = origFiliterShape[Dims4D::Filter::OC] / groupNum;
    const auto groupInChannel = getShape(groupConv.getInput())[Dims4D::Act::C] / groupNum;
    const auto groupOutChannel = getShape(groupConv.getOutput())[Dims4D::Act::C] / groupNum;
    VPUX_THROW_UNLESS(groupInSize == groupInChannel && groupOutSize == groupOutChannel,
                      "groupInSize '{0}' not equal with input channel '{1}' or groupOutSize '{2}' not equal with "
                      "output channel '{3}' ",
                      groupInSize, groupInChannel, groupOutSize, groupOutChannel);

    const auto oc = getShape(groupConv.getOutput())[Dims4D::Act::C];
    const auto ic = getShape(groupConv.getInput())[Dims4D::Act::C];
    auto newFilterShape = Shape{oc, ic, ky, kx};
    std::vector<float> weights(newFilterShape.totalSize(), .0f);

    const auto inValues = origContent.getValues<float>();

    const auto weightsSizePerOC = ic * ky * kx;
    const auto weightsSizePerGroup = kx * ky * groupInSize * groupOutSize;
    for (int64_t groupIndex = 0; groupIndex < groupNum; groupIndex++) {
        auto innerOffset = weightsSizePerGroup * groupIndex;
        for (auto idx = 0; idx < weightsSizePerGroup; idx++) {
            weights[weightsSizePerOC * groupIndex + innerOffset + idx] = inValues[innerOffset + idx];
        }
    }

    const DimsOrder weightsOrder = DimsOrder::OIYX;
    const auto weightsType =
            mlir::RankedTensorType::get(newFilterShape.raw(), origContentType.getElementType(),
                                        getTensorAttr(rewriter.getContext(), weightsOrder, nullptr, nullptr));

    return Const::buildWeightsConst(rewriter, origFilter.getLoc(), weightsType, ArrayRef(weights));
}

// Create weights for Convolution that can generate the same result with input
mlir::Value OptimizeGroupConvConcat::createWeights(mlir::Value activation, const GroupConvParameters& parameters,
                                                   mlir::PatternRewriter& rewriter) const {
    const auto kx = parameters.kx;
    const auto ky = parameters.ky;
    const auto actShape = getShape(activation);
    const auto ic = actShape[Dims4D::Act::C];
    const auto oc = actShape[Dims4D::Act::C];

    const Shape weightsShape = {oc, ic, ky, kx};
    std::vector<float> weights(weightsShape.totalSize(), .0f);

    // assign values
    const auto weightsSizePerOC = ic * ky * kx;
    for (int64_t ocIndex = 0; ocIndex < oc; ocIndex++) {
        auto currOffsetPerOC = ocIndex * weightsSizePerOC;
        auto innerOffset = kx * ky * ocIndex + kx * ky / 2;
        weights[currOffsetPerOC + innerOffset] = 1.0f;
    }

    const DimsOrder weightsOrder = DimsOrder::OIYX;
    const auto weightsType = mlir::RankedTensorType::get(
            weightsShape.raw(), mlir::cast<NDTypeInterface>(activation.getType()).getElementType(),
            getTensorAttr(rewriter.getContext(), weightsOrder, nullptr, nullptr));
    return Const::buildWeightsConst(rewriter, activation.getLoc(), weightsType, ArrayRef(weights));
}

mlir::LogicalResult OptimizeGroupConvConcat::matchAndRewrite(IE::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got concat layer at '{1}'", origOp->getName(), origOp->getLoc());

    auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    if (outputShape.size() != SUPPORTED_RANK) {
        return mlir::failure();
    }

    // Skip conversion when channel number is large for below reasons:
    // 1.Performance of DMA is extremely poor only when the number of concat channels is small and the concat dimension
    // is on the innermost dimension.
    // 2.Too large number of channels would cause large constant size of new filter.
    // 64 is an experimental value to ensure the conversion is beneficial.
    const int64_t CHANNEL_NUM_TO_PREVENT_LARGE_FILTER = 64;
    if (outputShape[Dims4D::Act::C] > CHANNEL_NUM_TO_PREVENT_LARGE_FILTER) {
        return mlir::failure();
    }

    GroupConvParameters parameters;
    auto getInputs = getValidConcatInputs(origOp, parameters);
    if (mlir::failed(getInputs)) {
        _log.trace("[{0}] Failed to get valid input pattern at '{1}'", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    _log.trace("[{0}] Rewriting '{1}'", origOp->getName(), origOp->getLoc());

    auto inputs = getInputs.value();
    mlir::Value root = nullptr;
    IE::GroupConvolutionOp origGroupConv = nullptr;
    // Create filter
    SmallVector<mlir::Value> newWeights;
    for (const auto& input : inputs) {
        auto groupConv = input.groupConv;
        if (groupConv != nullptr) {
            origGroupConv = groupConv;
            // Find root by groupConv's input
            if (root == nullptr) {
                root = groupConv.getInput();
            }
            newWeights.push_back(convertGroupConvWeights(groupConv, rewriter));
        } else {
            // Find root by concat's input
            if (root == nullptr) {
                auto inputIndex = input.index;
                root = origOp.getInputs()[inputIndex];
            }
            newWeights.push_back(createWeights(root, parameters, rewriter));
        }
    }
    VPUX_THROW_WHEN(origGroupConv == nullptr, "Can't find GroupConv");

    auto concatWeights = rewriter.createOrFold<IE::ConcatOp>(origOp.getLoc(), newWeights, Dims4D::Filter::OC);

    auto newConv = rewriter.create<IE::ConvolutionOp>(
            origOp.getLoc(), root, concatWeights, nullptr, origGroupConv.getStrides(), origGroupConv.getPadsBegin(),
            origGroupConv.getPadsEnd(), origGroupConv.getDilations(), origGroupConv.getPostOpAttr(),
            origGroupConv.getClampAttr(), nullptr);

    rewriter.replaceOp(origOp, newConv.getOutput());

    return mlir::success();
}

//
// OptimizeGroupConvConcatPass
//

class OptimizeGroupConvConcatPass final : public IE::OptimizeGroupConvConcatBase<OptimizeGroupConvConcatPass> {
public:
    explicit OptimizeGroupConvConcatPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeGroupConvConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeGroupConvConcat>(&ctx, _log);
    IE::ConcatOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeGroupConvConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeGroupConvConcatPass(Logger log) {
    return std::make_unique<OptimizeGroupConvConcatPass>(log);
}

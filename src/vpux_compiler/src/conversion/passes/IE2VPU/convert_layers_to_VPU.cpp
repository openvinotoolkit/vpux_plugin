//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_layers_to_VPU.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/factories/convert_layers_to_vpu_strategy_getter.hpp"
#include "vpux/compiler/core/attributes/tensor_attr.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

// Generated
#include <vpux/compiler/conversion/convert_layers_to_VPU.hpp.inc>

using namespace vpux;

//
// IfRewrite
//

mlir::LogicalResult IfRewrite::matchAndRewrite(IE::IfOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.debug("Found If Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::IRMapping mapper;
    auto thenBlock = &origOp.getThenBranch().getBlocks().front();
    auto elseBlock = &origOp.getElseBranch().getBlocks().front();

    for (auto valueIt : llvm::enumerate(thenBlock->getArguments())) {
        auto blockArg = origOp.getInputs()[valueIt.index()];
        mapper.map(valueIt.value(), blockArg);
    }
    for (auto valueIt : llvm::enumerate(elseBlock->getArguments())) {
        auto blockArg = origOp.getInputs()[valueIt.index()];
        mapper.map(valueIt.value(), blockArg);
    }

    // Then branch construct
    SmallVector<mlir::Value> thenBranchResults;
    SmallVector<mlir::Type> outTypes;
    for (auto& op : origOp.getThenBranch().getOps()) {
        mlir::Operation* newOp = rewriter.clone(op, mapper);
        if (mlir::isa<IE::YieldOp>(op)) {
            for (mlir::Value operand : newOp->getOperands()) {
                thenBranchResults.push_back(operand);
                outTypes.push_back(operand.getType());
            }
            rewriter.eraseOp(newOp);
            continue;
        }
        for (const auto& [result, newResult] : zip(op.getResults(), newOp->getResults())) {
            mapper.map(result, newResult);
        }
    }

    // Else branch construct
    SmallVector<mlir::Value> elseBranchResults;
    for (auto& op : origOp.getElseBranch().getOps()) {
        mlir::Operation* newOp = rewriter.clone(op, mapper);
        if (mlir::isa<IE::YieldOp>(op)) {
            for (mlir::Value operand : newOp->getOperands()) {
                elseBranchResults.push_back(operand);
            }
            rewriter.eraseOp(newOp);
            continue;
        }
        for (const auto& [result, newResult] : zip(op.getResults(), newOp->getResults())) {
            mapper.map(result, newResult);
        }
    }

    auto cond = origOp.getCond();
    SmallVector<mlir::Value> branchResults;
    int64_t numInputs = thenBranchResults.size();
    for (auto i = 0; i < numInputs; i++) {
        auto result = rewriter.create<VPU::ConditionalCopyOp>(origOp.getLoc(), outTypes[i], cond, thenBranchResults[i],
                                                              elseBranchResults[i]);
        branchResults.push_back(result);
    }
    rewriter.replaceOp(origOp, branchResults);

    return mlir::success();
}

//
// CTCGreedyDecoderSeqLenRewrite
//

mlir::LogicalResult CTCGreedyDecoderSeqLenRewrite::matchAndRewrite(IE::CTCGreedyDecoderSeqLenOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CTCGreedyDecoderSeqLen Operation '{0}'", origOp->getLoc());

    mlir::Value blankIndexValue = origOp.getBlankIndex();
    if (blankIndexValue == nullptr) {
        // Default value is C-1
        auto* ctx = origOp->getContext();
        const auto inShape = getShape(origOp.getInput()).raw();

        if (inShape.size() != 3) {
            return errorAt(origOp.getLoc(), "ConvertLayers2VPU::CTCGreedyDecoderSeqLenRewrite: First input tensor "
                                            "should have 3 dimensions: [N, T, C]");
        }
        auto blankIndxDefValue = checked_cast<int32_t>(inShape.back() - 1);
        auto blankIndxShape = mlir::RankedTensorType::get(
                {1}, mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signed));
        blankIndexValue = Const::createConst(rewriter, origOp.getLoc(), blankIndxShape, ArrayRef(blankIndxDefValue));
    }
    rewriter.replaceOpWithNewOp<VPU::CTCGreedyDecoderSeqLenOp>(origOp, origOp.getInput(), origOp.getSequenceLength(),
                                                               blankIndexValue, origOp.getMergeRepeatedAttr());
    return mlir::success();
}

//
// ProposalRewrite
//

mlir::LogicalResult ProposalRewrite::matchAndRewrite(IE::ProposalOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Proposal Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::ProposalOp>(origOp, origOp.getClassProbs(), origOp.getBboxDeltas(),
                                                 origOp.getImageShape(), nullptr, origOp.getProposalAttrsAttr());
    _log.trace("Replaced with 'VPU.ProposalOp'");

    return mlir::success();
}

//
// SplitRewrite
//

mlir::LogicalResult SplitRewrite::matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Split Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::SplitOp>(origOp, origOp.getInput(), origOp.getAxis(), origOp.getNumSplitsAttr(),
                                              origOp.getAxisValueAttr());

    return mlir::success();
}

mlir::LogicalResult StubRewrite::matchAndRewrite(IE::StubOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.debug("Found Stub Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::StubOp>(origOp, origOp.getOutputs().getTypes(), origOp.getInputs());

    return mlir::success();
}

//
// NonMaxSuppressionRewrite
//

mlir::LogicalResult NonMaxSuppressionRewrite::matchAndRewrite(IE::NonMaxSuppressionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found NonMaxSuppression Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::NonMaxSuppressionOp>(
            origOp, origOp.getInBoxCoords(), origOp.getInBoxScores(), origOp.getBoxEncoding(),
            origOp.getSortResultDescending(), origOp.getMaxOutputBoxesPerClassValueAttr(),
            origOp.getIouThresholdValueAttr(), origOp.getScoreThresholdValueAttr(), origOp.getSoftNmsSigmaValueAttr());

    _log.trace("Replaced with 'VPU.NonMaxSuppressionOp'");

    return mlir::success();
}

//
// GRUCellRewrite
//

mlir::LogicalResult GRUCellRewrite::matchAndRewrite(IE::GRUCellOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found GRUCell Operation '{0}'", origOp->getLoc());

    auto* ctx = origOp->getContext();
    const auto inputShape = getShape(origOp.getInputData()).raw();
    const auto batchSize = inputShape[0];
    const auto inputSize = inputShape[1];
    SmallVector<int64_t> newInputShape = {batchSize, 1, inputSize};
    const auto newInputShapeAttr = getIntArrayAttr(ctx, newInputShape);
    auto newInput =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getInputData(), nullptr, false, newInputShapeAttr);

    const auto initialStateShape = getShape(origOp.getInitialHiddenState()).raw();
    const auto hiddenSize = initialStateShape[1];
    SmallVector<int64_t> newInitialStateShape = {batchSize, 1, hiddenSize};
    const auto newInitialStateShapeAttr = getIntArrayAttr(ctx, newInitialStateShape);
    auto newInitialState =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getInitialHiddenState(), /*shape=*/nullptr,
                                            /*special_zero=*/false, newInitialStateShapeAttr);

    SmallVector<int64_t> newWeightsShape = {1, 3 * hiddenSize, inputSize};
    const auto newWeightsShapeAttr = getIntArrayAttr(ctx, newWeightsShape);
    auto newWeights =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getWeights(), nullptr, false, newWeightsShapeAttr);

    SmallVector<int64_t> newReWeightsShape = {1, 3 * hiddenSize, hiddenSize};
    const auto newReWeightsShapeAttr = getIntArrayAttr(ctx, newReWeightsShape);
    auto newReWeights =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getRecurrenceWeights(), /*shape=*/nullptr,
                                            /*special_zero=*/false, newReWeightsShapeAttr);

    const auto biasesShape = getShape(origOp.getBiases()).raw();
    SmallVector<int64_t> newBiasesShape = {1, biasesShape[0]};
    const auto newBiasesShapeAttr = getIntArrayAttr(ctx, newBiasesShape);
    auto newBiases = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.getBiases(), /*shape=*/nullptr,
                                                     /*special_zero=*/false, newBiasesShapeAttr);

    const auto seqLenAttr = getIntAttr(ctx, 1);
    const auto directionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::FORWARD);

    auto gruSeq =
            rewriter.create<VPU::GRUSequenceOp>(origOp->getLoc(), newInput, newInitialState, newWeights, newReWeights,
                                                newBiases, origOp.getHiddenSizeAttr(), seqLenAttr, directionAttr,
                                                origOp.getShouldLinearBeforeResetAttr(), origOp.getClipAttr());
    SmallVector<int64_t> newOutputShape = {batchSize, hiddenSize};
    const auto newOutputShapeAttr = getIntArrayAttr(ctx, newOutputShape);
    rewriter.replaceOpWithNewOp<VPU::ReshapeOp>(origOp, gruSeq.getOutputHiddenState(), /*shape=*/nullptr,
                                                /*special_zero=*/false, newOutputShapeAttr);

    return mlir::success();
}

//
// InterpolateRewrite
//

mlir::LogicalResult InterpolateRewrite::matchAndRewrite(IE::InterpolateOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    rewriter.replaceOpWithNewOp<VPU::InterpolateOp>(
            origOp, origOp.getType(), origOp.getInput(), origOp.getSizes(), origOp.getScales(), origOp.getAxes(),
            /*coordinates*/ nullptr, /* lambdas */ nullptr, origOp.getSizesAttrAttr(), origOp.getScalesAttrAttr(),
            origOp.getAxesAttrAttr(), origOp.getTileOffsetAttrAttr(), origOp.getInitialInputDimsAttrAttr(),
            origOp.getInitialOutputDimsAttrAttr(),
            /*initial_input_offset_attr=*/nullptr, /*initial_output_offset_attr=*/nullptr,
            /*multiClusterStrategy=*/nullptr, origOp.getAttrAttr(), origOp.getOutputChannelsAttr());
    return mlir::success();
}

//
// TopKRewrite
//

mlir::LogicalResult TopKRewrite::matchAndRewrite(IE::TopKOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found TopK Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::TopKOp>(origOp, origOp.getInput(), origOp.getK(), origOp.getKValueAttr(),
                                             origOp.getAxis(), origOp.getMode(), origOp.getSort(),
                                             origOp.getElementType(), /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// TransposedConvRewrite
//

mlir::LogicalResult TransposedConvRewrite::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found TransposedConvolution Operation '{0}'", origOp->getLoc());

    auto outType = origOp.getOutput().getType();

    rewriter.replaceOpWithNewOp<VPU::TransposedConvolutionOp>(
            origOp, outType, origOp.getInput(), origOp.getFilter(), origOp.getOutputShape(), origOp.getBias(),
            origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getOutputChannelsAttr());

    return mlir::success();
}

//
// NormalizeL2Rewrite
//

mlir::LogicalResult NormalizeL2Rewrite::matchAndRewrite(IE::NormalizeL2Op origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found NormalizeL2 Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::NormalizeL2Op>(origOp, origOp.getData(), origOp.getAxesValueAttr(),
                                                    origOp.getEpsAttr(), origOp.getEpsModeAttr(),
                                                    /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// LSTMCellRewrite
//

mlir::LogicalResult LSTMCellRewrite::matchAndRewrite(IE::LSTMCellOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto weights = origOp.getWeights();
    const auto biases = origOp.getBiases();
    if (!weights || !biases) {
        return matchFailed(rewriter, origOp,
                           "VPU::LSTMCell does not support missing weights or biases; it should have been decomposed "
                           "by the DecomposeLSTMCellPass.");
    }

    rewriter.replaceOpWithNewOp<VPU::LSTMCellOp>(origOp, origOp.getInputData(), origOp.getInitialHiddenState(),
                                                 origOp.getInitialCellState(), weights, origOp.getRecurrenceWeights(),
                                                 biases, origOp.getHiddenSizeAttr());
    return mlir::success();
}

//
// LSTMSequenceRewrite
//

mlir::LogicalResult LSTMSequenceRewrite::matchAndRewrite(IE::LSTMSequenceOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    const auto weights = origOp.getWeights();
    const auto biases = origOp.getBiases();
    if (weights || biases) {
        return matchFailed(rewriter, origOp,
                           "VPU::LSTMSequence does not support weights and biases; it should have been decomposed by "
                           "the DecomposeLSTMSequencePass.");
    }

    _log.trace("Found LSTMSequence Operation '{0}'", origOp->getLoc());
    rewriter.replaceOpWithNewOp<VPU::LSTMSequenceOp>(
            origOp, origOp.getInputData(), origOp.getInitialHiddenState(), origOp.getInitialCellState(),
            origOp.getReccurenceWeights(), origOp.getSequenceLengthAttr(), origOp.getDirectionAttr(), nullptr);

    return mlir::success();
}

//
// LSTMGatesRewrite
//

mlir::LogicalResult LSTMGatesRewrite::matchAndRewrite(IE::LSTMGatesOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found LSTMGates Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::LSTMGatesOp>(origOp, origOp.getGatesInput(), origOp.getInitialCellState(),
                                                  /*multiClusterStrategy=*/nullptr);

    return mlir::success();
}

//
// GroupConvolutionRewrite
//

mlir::LogicalResult GroupConvolutionRewrite::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found GroupConvolutionRewrite Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::GroupConvolutionOp>(
            origOp, origOp.getOutput().getType(), origOp.getInput(), origOp.getFilter(), origOp.getBias(),
            origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
            origOp.getGroupsAttr(), origOp.getPostOpAttr(), origOp.getOutputChannelsAttr());

    return mlir::success();
}

//
// DynamicReshapeRewrite
//

mlir::LogicalResult DynamicReshapeRewrite::matchAndRewrite(IE::DynamicReshapeOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DynamicReshape Operation '{0}'", origOp->getLoc());

    const auto outputType = origOp.getOutput().getType();
    rewriter.replaceOpWithNewOp<VPU::DynamicReshapeOp>(origOp, outputType, origOp.getInput(), origOp.getShape(),
                                                       origOp.getOutputShapeAttr(), origOp.getOutputBoundsAttr(),
                                                       origOp.getOnlySetShapeAttr());

    return mlir::success();
}

//
// DynamicTileRewrite
//

mlir::LogicalResult DynamicTileRewrite::matchAndRewrite(IE::DynamicTileOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DynamicTileOp Operation '{0}'", origOp->getLoc());

    const auto outputType = origOp.getOutput().getType();
    rewriter.replaceOpWithNewOp<VPU::DynamicTileOp>(origOp, outputType, origOp.getInput(), origOp.getTargetShape(),
                                                    origOp.getRepeats(), origOp.getRepeatsValuesAttr(),
                                                    origOp.getOutputShapeAttr(), origOp.getOutputBoundsAttr());

    return mlir::success();
}

namespace {

//
// ConvertLayers2VPUPass
//

class ConvertLayers2VPUPass final : public ConvertLayers2VPUBase<ConvertLayers2VPUPass> {
public:
    explicit ConvertLayers2VPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2VPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto arch = VPU::getArch(func);

    auto archSpecificStrategy = createConvertLayers2VPUStrategy(arch);

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();

    mlir::RewritePatternSet patterns(&ctx);
    archSpecificStrategy->addPatterns(patterns, _log);

    patterns.add<IfRewrite>(&ctx, _log);
    patterns.add<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.add<ProposalRewrite>(&ctx, _log);
    patterns.add<SplitRewrite>(&ctx, _log);
    patterns.add<StubRewrite>(&ctx, _log);
    patterns.add<NonMaxSuppressionRewrite>(&ctx, _log);
    patterns.add<InterpolateRewrite>(&ctx, _log);

    patterns.add<GRUCellRewrite>(&ctx, _log);
    patterns.add<TopKRewrite>(&ctx, _log);
    patterns.add<TransposedConvRewrite>(&ctx, _log);
    patterns.add<NormalizeL2Rewrite>(&ctx, _log);
    patterns.add<LSTMCellRewrite>(&ctx, _log);
    patterns.add<LSTMSequenceRewrite>(&ctx, _log);
    patterns.add<LSTMGatesRewrite>(&ctx, _log);
    patterns.add<GroupConvolutionRewrite>(&ctx, _log);
    patterns.add<DynamicReshapeRewrite>(&ctx, _log);
    patterns.add<DynamicTileRewrite>(&ctx, _log);
    populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2VPUPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUPass>(log);
}

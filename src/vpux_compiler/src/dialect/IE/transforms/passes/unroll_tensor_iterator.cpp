//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

//
// TensorIteratorRewriter
//

class TensorIteratorRewriter final : public mlir::OpRewritePattern<IE::TensorIteratorOp> {
public:
    TensorIteratorRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TensorIteratorOp>(ctx), _log(log) {
        setDebugName("TensorIteratorRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::TensorIteratorOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// TensorIteratorRewriter
//

mlir::LogicalResult TensorIteratorRewriter::matchAndRewrite(IE::TensorIteratorOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.debug("Found TensorIterator Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::IRMapping mapper;
    auto* ctx = origOp->getContext();
    auto bodyBlock = &origOp.getBodyModule().getBlocks().front();
    auto sliceInputDescAttrVector = parseCustomAttrArray<IE::SliceInputPortMapAttr>(origOp.getSliceInputDescsAttr());
    auto invariantInputDescAttrVector =
            parseCustomAttrArray<IE::InvariantInputPortMapAttr>(origOp.getInvariantInputDescsAttr());
    auto feedbackInputDescAttrVector =
            parseCustomAttrArray<IE::MergedInputPortMapAttr>(origOp.getFeedbackInputDescsAttr());
    auto concatOutputDescAttrVector =
            parseCustomAttrArray<IE::ConcatOutputPortMapAttr>(origOp.getConcatOutputDescsAttr());
    auto invariantOutputDescAttrVector =
            parseCustomAttrArray<IE::InvariantOutputPortMapAttr>(origOp.getInvariantOutputDescsAttr());

    auto opInputs = origOp.getInputs();
    auto bodyInputs = bodyBlock->getArguments();
    auto numIterations = origOp.getNumIterations();
    SmallVector<mlir::Value> bodyResultsBuffer;

    // Prepare mapper
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> slicedInputValueMap, toConcatOutputValueMap;

    for (auto sipm : sliceInputDescAttrVector) {
        auto exId = sipm.getExternalPortId().getValue().getSExtValue();
        auto inId = sipm.getInternalLayerId().getValue().getSExtValue();
        auto toSplitInput = opInputs[exId];

        auto partSize = sipm.getPartSize().getValue().getSExtValue();
        auto start = sipm.getStart().getValue().getSExtValue();
        auto stride = sipm.getStride().getValue().getSExtValue();
        auto axis = sipm.getAxis().getValue().getSExtValue();
        auto shape = getShape(toSplitInput);
        mlir::SmallVector<int64_t> staticSizes = to_small_vector(shape.raw());
        auto staticOffsets = mlir::SmallVector<int64_t>(shape.size(), 0);
        staticSizes[axis] = 1;

        // split iterating
        SmallVector<mlir::Value> splitResults;
        for (auto idx : irange(numIterations)) {
            staticOffsets[axis] = stride > 0 ? start + idx * partSize : start - idx * partSize;
            auto slicedOp =
                    rewriter.create<IE::SliceOp>(origOp->getLoc(), toSplitInput, getIntArrayAttr(ctx, staticOffsets),
                                                 getIntArrayAttr(ctx, staticSizes));
            splitResults.push_back(slicedOp.getResult());
        }
        slicedInputValueMap.insert({inId, splitResults});
        _log.debug("Collect SliceInputDesc exId: {0} inId: {1} pair,\n corresponding sliced value: {2}.", exId, inId,
                   splitResults);
    }

    for (auto copm : concatOutputDescAttrVector) {
        auto exId = copm.getExternalPortId().getValue().getSExtValue();
        SmallVector<mlir::Value> toConcatResults(numIterations);
        toConcatOutputValueMap.insert({exId, toConcatResults});
    }
    for (auto iipm : invariantInputDescAttrVector) {
        auto exId = iipm.getExternalPortId().getValue().getSExtValue();
        auto inId = iipm.getInternalLayerId().getValue().getSExtValue();
        mapper.map(bodyInputs[inId], opInputs[exId]);
    }

    // Begin iterations
    for (int current_iter = 0; current_iter < numIterations; current_iter++) {
        _log.debug("current_iter:  {0}", current_iter);
        // Update mapper for feedback inputs i.e. back edge connections
        for (auto fipm : feedbackInputDescAttrVector) {
            auto exId = fipm.getExternalPortId().getValue().getSExtValue();
            auto bodyInputId = fipm.getBodyInputIndex().getValue().getSExtValue();
            auto inId = fipm.getInternalLayerId().getValue().getSExtValue();
            mapper.map(bodyInputs[inId], current_iter == 0 ? opInputs[exId] : bodyResultsBuffer[bodyInputId]);
        }

        for (auto sipm : sliceInputDescAttrVector) {
            auto inId = sipm.getInternalLayerId().getValue().getSExtValue();
            mapper.map(bodyInputs[inId], slicedInputValueMap[inId][current_iter]);
        }

        bodyResultsBuffer.clear();
        for (auto& op : origOp.getBodyModule().getOps()) {
            mlir::Operation* newOp = rewriter.clone(op, mapper);
            if (mlir::isa<IE::LoopTerminatorOp>(op)) {
                for (mlir::Value operand : newOp->getOperands()) {
                    bodyResultsBuffer.push_back(operand);
                }
                rewriter.eraseOp(newOp);
                continue;
            }
            for (const auto& [result, newResult] : zip(op.getResults(), newOp->getResults())) {
                mapper.map(result, newResult);
            }
        }

        for (auto copm : concatOutputDescAttrVector) {
            auto exId = copm.getExternalPortId().getValue().getSExtValue();
            auto inId = copm.getInternalLayerId().getValue().getSExtValue();
            toConcatOutputValueMap[exId][current_iter] = bodyResultsBuffer[inId];
        }
        _log.debug("current_iter {0} result: {1}", current_iter, bodyResultsBuffer);
    }

    mlir::DenseMap<int64_t, mlir::Value> concatedOutputValueMap;
    for (auto copm : concatOutputDescAttrVector) {
        auto exId = copm.getExternalPortId().getValue().getSExtValue();
        auto axis = copm.getAxis();
        auto stride = copm.getStride().getValue().getSExtValue();
        SmallVector<mlir::Value> inputList;
        for (auto idx : irange(numIterations)) {
            inputList.push_back(stride > 0 ? toConcatOutputValueMap[exId][idx]
                                           : toConcatOutputValueMap[exId][numIterations - 1 - idx]);
        }
        auto newConcat = rewriter.create<IE::ConcatOp>(origOp->getLoc(), inputList, axis, nullptr, nullptr);

        mlir::Value concatedValue = newConcat.getOutput();
        concatedOutputValueMap.insert({exId, concatedValue});
    }

    mlir::DenseMap<int64_t, mlir::Value> invariantOutputValueMap;
    for (auto iopm : invariantOutputDescAttrVector) {
        auto exId = iopm.getExternalPortId().getValue().getSExtValue();
        auto inId = iopm.getInternalLayerId().getValue().getSExtValue();
        invariantOutputValueMap.insert({exId, bodyResultsBuffer[inId]});
    }

    SmallVector<mlir::Value> finalResults;
    for (auto idx : irange(bodyResultsBuffer.size())) {
        mlir::Value result =
                toConcatOutputValueMap.count(idx) ? concatedOutputValueMap[idx] : invariantOutputValueMap[idx];
        finalResults.push_back(result);
    }
    rewriter.replaceOp(origOp, finalResults);

    return mlir::success();
}

//
// UnrollTensorIterator
//

class UnrollTensorIterator final : public IE::UnrollTensorIteratorBase<UnrollTensorIterator> {
public:
    explicit UnrollTensorIterator(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void UnrollTensorIterator::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TensorIteratorOp>();
    target.addIllegalOp<IE::LoopTerminatorOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TensorIteratorRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollTensorIteratorPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollTensorIteratorPass(Logger log) {
    return std::make_unique<UnrollTensorIterator>(log);
}

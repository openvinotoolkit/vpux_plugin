//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/unroll_batch.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// UnrollBatchPass
//

class UnrollBatchPass final : public IE::arch37xx::UnrollBatchBase<UnrollBatchPass> {
public:
    explicit UnrollBatchPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <class ConcreteOp>
bool isLegalOp(ConcreteOp op) {
    return vpux::IE::isShapeRankEqualToZero(op.getInput()) || vpux::IE::isBatchEqualToOne(op.getInput());
}

//
// safeRunOnFunc
//

void UnrollBatchPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>(&isLegalOp<IE::MaxPoolOp>);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(&isLegalOp<IE::AvgPoolOp>);
    target.addDynamicallyLegalOp<IE::FullyConnectedOp>(&isLegalOp<IE::FullyConnectedOp>);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(&isLegalOp<IE::ConvolutionOp>);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(&isLegalOp<IE::GroupConvolutionOp>);
    target.addDynamicallyLegalOp<IE::ExpOp>(&isLegalOp<IE::ExpOp>);
    target.addDynamicallyLegalOp<IE::SigmoidOp>(&isLegalOp<IE::SigmoidOp>);
    target.addDynamicallyLegalOp<IE::AndOp>([&](IE::AndOp op) -> bool {
        return (vpux::IE::isShapeRankEqualToZero(op.getInput1()) || vpux::IE::isShapeRankEqualToZero(op.getInput2())) ||
               !vpux::IE::areShapeRanksEqual(op.getInput1(), op.getInput2()) ||
               (vpux::IE::isBatchEqualToOne(op.getInput1()) || vpux::IE::isBatchEqualToOne(op.getInput2()));
    });
    target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) -> bool {
        return (vpux::IE::isShapeRankEqualToZero(op.getInput1()) || vpux::IE::isShapeRankEqualToZero(op.getInput2())) ||
               !vpux::IE::areShapeRanksEqual(op.getInput1(), op.getInput2()) ||
               (vpux::IE::isBatchEqualToOne(op.getInput1()) || vpux::IE::isBatchEqualToOne(op.getInput2()));
    });
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::ConvolutionOp>>(&ctx, _log, 1);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::FullyConnectedOp>>(&ctx, _log, 1);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::GroupConvolutionOp>>(&ctx, _log, 1);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::ExpOp>>(&ctx, _log, 1);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::SigmoidOp>>(&ctx, _log, 1);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::AndOp>>(&ctx, _log, 2);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::AddOp>>(&ctx, _log, 2);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::AvgPoolOp>>(&ctx, _log, 1);
    patterns.add<vpux::IE::BatchUnrollConverter<IE::MaxPoolOp>>(&ctx, _log, 1);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollBatchPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createUnrollBatchPass(Logger log) {
    return std::make_unique<UnrollBatchPass>(log);
}

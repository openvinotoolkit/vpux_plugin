//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// PerAxisFQConcatPass
//

class PerAxisFQConcatPass final : public IE::PerAxisFQConcatBase<PerAxisFQConcatPass> {
public:
    explicit PerAxisFQConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ConcatOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ConcatOpConverter
//

class PerAxisFQConcatPass::ConcatOpConverter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isPerAxisFqValue(mlir::Value input) {
    auto maybeFqOp = input.getDefiningOp<IE::FakeQuantizeOp>();
    if (maybeFqOp == nullptr) {
        return false;
    }

    return !IE::isPerTensorFQ({maybeFqOp});
}

bool isPerAxisFqOp(mlir::Operation* op) {
    auto maybeFqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op);
    if (maybeFqOp == nullptr) {
        return false;
    }

    return !IE::isPerTensorFQ({maybeFqOp});
}

bool isLegalConcat(IE::ConcatOp origConcatOp) {
    const auto concatUsers = origConcatOp.getOutput().getUsers();
    if (std::all_of(concatUsers.begin(), concatUsers.end(), isPerAxisFqOp)) {
        return true;
    }
    const auto concatInputList = origConcatOp.getInputs();
    return !std::all_of(concatInputList.begin(), concatInputList.end(), isPerAxisFqValue);
}

void appendFqValues(mlir::Value fqInput, std::vector<float>& totalValues) {
    // Fetch values from given FQ input and concatenate them with destination vector
    auto inConst = fqInput.getDefiningOp<Const::DeclareOp>();
    auto inConstAttr = inConst.getContentAttr().fold();
    auto inValues = inConstAttr.getValues<float>();
    std::copy(inValues.begin(), inValues.end(), std::back_inserter(totalValues));
}

Const::DeclareOp createFqTensor(mlir::Location loc, const std::vector<float>& totalFqValues,
                                mlir::PatternRewriter& rewriter) {
    // Build FQ input using concatenated values
    const auto tensorType = mlir::RankedTensorType::get({1, checked_cast<int64_t>(totalFqValues.size()), 1, 1},
                                                        mlir::Float32Type::get(rewriter.getContext()));
    const auto tensorAttr = mlir::DenseElementsAttr::get(tensorType, ArrayRef(totalFqValues));
    const auto tensorContentAttr = Const::ContentAttr::get(tensorAttr);
    return rewriter.create<Const::DeclareOp>(loc, tensorType, tensorContentAttr);
}

mlir::LogicalResult PerAxisFQConcatPass::ConcatOpConverter::matchAndRewrite(IE::ConcatOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    if (isLegalConcat(origOp)) {
        return mlir::failure();
    }

    const auto concatInputList = origOp.getInputs();
    if (concatInputList.empty()) {
        return mlir::failure();
    }

    std::vector<float> totalInLo;
    std::vector<float> totalInHi;
    std::vector<float> totalOutLo;
    std::vector<float> totalOutHi;

    SmallVector<IE::AutoBroadcastType> autoBroadcastVec;
    SmallVector<int64_t> levelsVec;

    for (const auto& concatInput : concatInputList) {
        auto fqOp = concatInput.getDefiningOp<IE::FakeQuantizeOp>();
        appendFqValues(fqOp.getInputLow(), totalInLo);
        appendFqValues(fqOp.getInputHigh(), totalInHi);
        appendFqValues(fqOp.getOutputLow(), totalOutLo);
        appendFqValues(fqOp.getOutputHigh(), totalOutHi);

        const auto autob = fqOp.getAutoBroadcast();
        if (!fqOp.getLevels().has_value()) {
            return mlir::failure();
        }
        const auto levels = fqOp.getLevels().value();
        autoBroadcastVec.push_back(autob);
        levelsVec.push_back(levels);
    }

    // Check that all levels are the same.
    const auto levels = levelsVec[0];
    const auto isEqualLvl = [levels](const int64_t lvl) -> bool {
        return levels == lvl;
    };
    if (!std::all_of(levelsVec.begin(), levelsVec.end(), isEqualLvl)) {
        return mlir::failure();
    }

    // Check that all broadcast types are the same.
    const auto autob = autoBroadcastVec[0];
    const auto isEqualBroadcast = [autob](const IE::AutoBroadcastType& broadcast) -> bool {
        return autob == broadcast;
    };

    if (!std::all_of(autoBroadcastVec.begin(), autoBroadcastVec.end(), isEqualBroadcast)) {
        return mlir::failure();
    }

    auto concatOp = rewriter.create<IE::ConcatOp>(origOp->getLoc(), concatInputList, origOp.getPerAxisAttr(),
                                                  origOp.getStaticOffsetsAttr());

    auto inLowOp = createFqTensor(origOp->getLoc(), totalInLo, rewriter);
    auto inHighOp = createFqTensor(origOp->getLoc(), totalInHi, rewriter);
    auto outLowOp = createFqTensor(origOp->getLoc(), totalOutLo, rewriter);
    auto outHighOp = createFqTensor(origOp->getLoc(), totalOutHi, rewriter);

    // lowFpType in not needed (nullptr), only levels are given
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origOp, concatOp.getOutput(), inLowOp, inHighOp, outLowOp,
                                                    outHighOp, getIntAttr(origOp.getContext(), levels),
                                                    /*lowFpType=*/nullptr, autob);

    return mlir::success();
}

void PerAxisFQConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PerAxisFQConcatPass::ConcatOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createPerAxisFQConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPerAxisFQConcatPass(Logger log) {
    return std::make_unique<PerAxisFQConcatPass>(log);
}

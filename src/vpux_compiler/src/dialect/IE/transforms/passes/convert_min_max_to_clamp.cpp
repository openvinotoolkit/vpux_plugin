//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

using namespace vpux;

namespace {

//
// ConvertMinMaxToClampPass
//

class ConvertMinMaxToClampPass final : public IE::ConvertMinMaxToClampBase<ConvertMinMaxToClampPass> {
public:
    explicit ConvertMinMaxToClampPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// MinMaxConverter
//

template <class ConcreteOp>
class MinMaxConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MinMaxConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("MinMaxConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult MinMaxConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto nonScalarOperand = origOp.getInput2();
    auto scalarOperand = origOp.getInput1();
    if (origOp.getInput1().getType().template cast<vpux::NDTypeInterface>().getNumElements() != 1) {
        nonScalarOperand = scalarOperand;
        scalarOperand = origOp.getInput2();
    }
    if (scalarOperand.getType().template cast<vpux::NDTypeInterface>().getNumElements() != 1) {
        return mlir::failure();
    }

    auto scalarInputConst = scalarOperand.template getDefiningOp<Const::DeclareOp>();
    if (scalarInputConst == nullptr) {
        _log.trace("Only constant input is supported for scalar_input'{0}'", origOp->getLoc());
        return mlir::failure();
    }
    if (const auto attr = scalarInputConst.getContentAttr(); !attr.isSplat()) {
        _log.trace("Only splat input is supported for 'scalar_input' '{0}'", origOp->getLoc());
        return mlir::failure();
    }
    auto scalarInputContent = scalarInputConst.getContent();
    auto scalarValue = scalarInputContent.template getSplatValue<float>();

    mlir::FloatAttr clampMax;
    mlir::FloatAttr clampMin;
    if (mlir::isa<IE::MaximumOp>(origOp)) {
        clampMax =
                getFPAttr(origOp->getContext(), checked_cast<double>(std::numeric_limits<vpux::type::float16>::max()));
        clampMin = getFPAttr(origOp->getContext(), scalarValue);
    } else if (mlir::isa<IE::MinimumOp>(origOp)) {
        clampMax = getFPAttr(origOp->getContext(), scalarValue);
        clampMin = getFPAttr(origOp->getContext(),
                             checked_cast<double>(std::numeric_limits<vpux::type::float16>::lowest()));
    }

    auto newClampOp = rewriter.create<IE::ClampOp>(origOp->getLoc(), nonScalarOperand, clampMin, clampMax);
    rewriter.replaceOp(origOp, newClampOp.getOutput());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertMinMaxToClampPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);

    target.addLegalOp<IE::ClampOp>();

    target.addDynamicallyLegalOp<IE::MaximumOp, IE::MinimumOp>([&](mlir::Operation* op) {
        for (auto operand : op->getOperands()) {
            if (operand.getType().cast<vpux::NDTypeInterface>().getNumElements() == 1)
                return false;
        }
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MinMaxConverter<IE::MinimumOp>>(&ctx, _log);
    patterns.add<MinMaxConverter<IE::MaximumOp>>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.debug("Failed to replace Min or Max operation with Clamp");
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertMinMaxToClampPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertMinMaxToClampPass(Logger log) {
    return std::make_unique<ConvertMinMaxToClampPass>(log);
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

namespace {

//
// PrecisionConverter
//

class PrecisionConverter final : public mlir::OpInterfaceConversionPattern<IE::LayerOpInterface> {
public:
    PrecisionConverter(mlir::MLIRContext* ctx, vpux::Logger log)
            : mlir::OpInterfaceConversionPattern<IE::LayerOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, vpux::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PrecisionConverter::matchAndRewrite(IE::LayerOpInterface origOp, vpux::ArrayRef<mlir::Value>,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());

    const auto inputElemType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto elemTypeFP16 = mlir::Float16Type::get(inputElemType.getContext());
    const auto inputCvtToFP16 = rewriter.createOrFold<IE::ConvertOp>(origOp->getLoc(), origOp->getOperand(0),
                                                                     mlir::TypeAttr::get(elemTypeFP16));

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperand(0), inputCvtToFP16);
    auto newOp = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ELEM_TYPE);

    const auto outputCvtToOrig = rewriter.createOrFold<IE::ConvertOp>(origOp->getLoc(), newOp->getResult(0),
                                                                      mlir::TypeAttr::get(inputElemType));
    origOp->getResult(0).replaceAllUsesWith(outputCvtToOrig);

    const auto resultNum = origOp->getNumResults();
    if (resultNum > 1) {
        for (auto index : irange<unsigned>(1, resultNum)) {
            origOp->getResult(index).replaceAllUsesWith(newOp->getResult(index));
        }
    }

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// AdjustSoftwareOpsPrecisionPass
//

class AdjustSoftwareOpsPrecisionPass final : public IE::AdjustSoftwareOpsPrecisionBase<AdjustSoftwareOpsPrecisionPass> {
public:
    explicit AdjustSoftwareOpsPrecisionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void AdjustSoftwareOpsPrecisionPass::safeRunOnModule() {
    auto& ctx = getContext();

    const auto isLegalTopKOp = [](IE::TopKOp op) {
        const auto inputElemType = op.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
        return inputElemType.isF16();
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::ConvertOp>();
    target.addDynamicallyLegalOp<IE::TopKOp>(isLegalTopKOp);
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PrecisionConverter>(&ctx, _log);

    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertOpsPrecisionToFP16Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustSoftwareOpsPrecisionPass(Logger log) {
    return std::make_unique<AdjustSoftwareOpsPrecisionPass>(log);
}

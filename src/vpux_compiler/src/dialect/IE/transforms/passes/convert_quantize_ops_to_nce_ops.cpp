//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/convert_quantize_ops_to_nce_ops_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/passes.hpp"

using namespace vpux;

namespace {

//
// ConvertQuantizeOpsToNceOpsPass
//

class ConvertQuantizeOpsToNceOpsPass final : public IE::ConvertQuantizeOpsToNceOpsBase<ConvertQuantizeOpsToNceOpsPass> {
public:
    explicit ConvertQuantizeOpsToNceOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

bool isPerAxisQuant(mlir::Value val) {
    auto elemType = val.getType().cast<vpux::NDTypeInterface>().getElementType();
    return elemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
}

void ConvertQuantizeOpsToNceOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    auto archSpecificStrategy = IE::createConvertQuantizeOpsToNceOpsStrategy(arch);

    mlir::ConversionTarget toAvgPoolTarget(ctx);
    mlir::RewritePatternSet toAvgPoolPatterns(&ctx);
    archSpecificStrategy->prepareAvgPool(toAvgPoolTarget, toAvgPoolPatterns, ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, toAvgPoolTarget, std::move(toAvgPoolPatterns)))) {
        signalPassFailure();
    }

    // perTensor quantize/dequantize convert to add or and
    mlir::ConversionTarget toEltwiseTarget(ctx);
    mlir::RewritePatternSet toEltwisePatterns(&ctx);
    archSpecificStrategy->prepareEltwise(toEltwiseTarget, toEltwisePatterns, ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, toEltwiseTarget, std::move(toEltwisePatterns)))) {
        signalPassFailure();
    }

    // per-axis scales and per-tensor zero points quantize/dequantize convert to DW conv
    mlir::ConversionTarget quantToDwTarget(ctx);
    mlir::RewritePatternSet quantToDwPatterns(&ctx);
    archSpecificStrategy->prepareQuantToDw(quantToDwTarget, quantToDwPatterns, ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, quantToDwTarget, std::move(quantToDwPatterns)))) {
        signalPassFailure();
    }
}

}  // namespace

namespace vpux::IE {

bool isLegalQuantizeOp(IE::QuantizeOp quantizeOp, bool canUseCMajor) {
    const auto isPerAxisQuantized = isPerAxisQuant(quantizeOp.getOutput());
    auto outputLayerUsers = quantizeOp.getOutput().getUsers();
    auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
        return mlir::isa<IE::ConvolutionOp>(user);
    });

    return (anyUserIsConv && canUseCMajor) || isPerAxisQuantized;
};

bool isLegalDequantizeOp(IE::DequantizeOp dequantizeOp) {
    return isPerAxisQuant(dequantizeOp.getInput());
};

}  // namespace vpux::IE

//
// createConvertQuantizeOpsToNceOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertQuantizeOpsToNceOpsPass(Logger log) {
    return std::make_unique<ConvertQuantizeOpsToNceOpsPass>(log);
}

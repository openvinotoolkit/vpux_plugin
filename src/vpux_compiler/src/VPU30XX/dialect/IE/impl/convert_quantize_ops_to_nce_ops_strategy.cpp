//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"

using namespace vpux;

namespace vpux::IE::arch30xx {

//
// GenericConverter
//

template <class ConcreteOp>
class GenericConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericConverter<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(this->getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    rewriter.replaceOpWithNewOp<IE::AndOp>(originOp, originOp.getType(), originOp.getInput(), originOp.getInput(),
                                           broadcastType, nullptr, nullptr);

    return mlir::success();
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareAvgPool(mlir::ConversionTarget&, mlir::RewritePatternSet&,
                                                        mlir::MLIRContext&, Logger&) {
    // not supported on the arch
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareEltwise(mlir::ConversionTarget& toEltwiseTarget,
                                                        mlir::RewritePatternSet& toEltwisePatterns,
                                                        mlir::MLIRContext& ctx, Logger& log) {
    toEltwiseTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        auto outType = quantizeOp.getOutput().getType().cast<vpux::NDTypeInterface>();
        const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(VPU::ArchKind::NPU30XX, outType);
        return IE::isLegalQuantizeOp(quantizeOp, canUseCMajor);
    });
    toEltwiseTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        return IE::isLegalDequantizeOp(dequantizeOp);
    });
    toEltwiseTarget.addLegalOp<IE::AndOp>();
    toEltwiseTarget.addLegalOp<IE::AddOp>();
    toEltwiseTarget.addLegalOp<IE::QuantizeCastOp>();

    toEltwisePatterns.add<GenericConverter<IE::QuantizeOp>>(&ctx, log);
    toEltwisePatterns.add<GenericConverter<IE::DequantizeOp>>(&ctx, log);
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareQuantToDw(mlir::ConversionTarget&, mlir::RewritePatternSet&,
                                                          mlir::MLIRContext&, Logger&) {
}

}  // namespace vpux::IE::arch30xx

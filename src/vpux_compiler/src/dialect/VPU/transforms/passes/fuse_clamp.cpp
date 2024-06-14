//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

using namespace vpux;

namespace {

//
// ClampConverter
//

class ClampConverter final : public mlir::OpRewritePattern<VPU::ClampOp> {
public:
    ClampConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ClampOp>(ctx), _log(log) {
        this->setDebugName("ClampConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::ClampOp clampOp, mlir::PatternRewriter& rewriter) const final;
    std::pair<int32_t, int32_t> getTargetClamps(mlir::Type outElemType, const int32_t ppeClampLow,
                                                const int32_t ppeClampHigh, const double clampMin,
                                                const double clampMax) const;
    VPU::PPEModeAttr getPPEMode(const VPU::PPEModeAttr mode, mlir::Type outElemType) const;

private:
    Logger _log;
};

std::pair<int32_t, int32_t> ClampConverter::getTargetClamps(mlir::Type outElemType, const int32_t ppeClampLow,
                                                            const int32_t ppeClampHigh, const double clampMin,
                                                            const double clampMax) const {
    // For quantized types min/max must be divided by scale.
    // For float types max must be converted to float16/bfloat16. Clamp min should be set to 0x80000000.
    // Target clamp high must be the minimal value between existing PPE high and clamp max.
    // For example:
    // NCE has range [0, 255], clamp max = 128. Target clamp high must be set to 128.
    // NCE has range [0, 100], clamp max = 128. Target clamp high must be set to 100.
    // Same applies for target clamp low, except that the maximal value must be chosen.
    if (outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        const auto perTensor = outElemType.cast<mlir::quant::UniformQuantizedType>();
        const auto scale = perTensor.getScale();
        const auto zp = perTensor.getZeroPoint();

        const auto quantizedLow = std::round(clampMin / scale) + zp;
        const auto quantizedHigh = std::round(clampMax / scale) + zp;

        const auto targetClampLow = std::max(ppeClampLow, checked_cast<int32_t>(quantizedLow));
        const auto targetClampHigh = std::min(ppeClampHigh, checked_cast<int32_t>(quantizedHigh));
        return std::pair<int32_t, int32_t>{targetClampLow, targetClampHigh};
    } else if (outElemType.isF16()) {
        const vpux::type::float16 clampMaxF16 = static_cast<float>(clampMax);
        const int16_t* clampMaxI16 = reinterpret_cast<const int16_t*>(&clampMaxF16);
        const int32_t clampMaxI32 = *clampMaxI16;

        const auto targetClampLow = std::numeric_limits<int32_t>::min();
        const auto targetClampHigh = std::min(ppeClampHigh, clampMaxI32);
        return std::pair<int32_t, int32_t>{targetClampLow, targetClampHigh};
    } else if (outElemType.isBF16()) {
        const vpux::type::bfloat16 clampMaxF16 = static_cast<float>(clampMax);
        const int16_t* clampMaxI16 = reinterpret_cast<const int16_t*>(&clampMaxF16);
        const int32_t clampMaxI32 = *clampMaxI16;

        const auto targetClampLow = std::numeric_limits<int32_t>::min();
        const auto targetClampHigh = std::min(ppeClampHigh, clampMaxI32);
        return std::pair<int32_t, int32_t>{targetClampLow, targetClampHigh};
    } else {
        VPUX_THROW("ClampConverter::getTargetClamps: Unsupported type: {0}", outElemType);
    }
}

VPU::PPEModeAttr ClampConverter::getPPEMode(const VPU::PPEModeAttr mode, mlir::Type outElemType) const {
    // For quantized types lower bound of the clamp can be less than zero point.
    // In that case the original mode must be set.
    if (outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        return mode;
    }

    // For float16 and bi-float16 types lower bound of the clamp must be equal to zero.
    // The mode effectively becomes RELUX.
    return VPU::PPEModeAttr::get(outElemType.getContext(), VPU::PPEMode::LRELUX);
}

mlir::LogicalResult ClampConverter::matchAndRewrite(VPU::ClampOp clampOp, mlir::PatternRewriter& rewriter) const {
    auto nceOp = mlir::cast<VPU::NCEOpInterface>(clampOp.getInput().getDefiningOp());
    const auto ppeTask = nceOp.getPPE().value();

    const auto clampMin = clampOp.getMin().convertToDouble();
    const auto clampMax = clampOp.getMax().convertToDouble();

    const auto ppeClampLowAttr = ppeTask.getClampLow();
    const auto ppeClampHighAttr = ppeTask.getClampHigh();
    const auto ppeClampLow = checked_cast<int32_t>(ppeClampLowAttr.getValue().getSExtValue());
    const auto ppeClampHigh = checked_cast<int32_t>(ppeClampHighAttr.getValue().getSExtValue());

    const auto outElemType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto targetClamp = getTargetClamps(outElemType, ppeClampLow, ppeClampHigh, clampMin, clampMax);
    const auto targetClampLow = targetClamp.first;
    const auto targetClampHigh = targetClamp.second;

    const auto ppeMode = getPPEMode(ppeTask.getMode(), outElemType);
    auto ctx = clampOp.getContext();
    auto ppeTaskAttr =
            VPU::PPETaskAttr::get(ctx, ppeMode, getIntAttr(ctx, targetClampLow), getIntAttr(ctx, targetClampHigh),
                                  ppeTask.getLreluMult(), ppeTask.getLreluShift(), ppeTask.getQuantScale(),
                                  ppeTask.getQuantMult(), ppeTask.getQuantShift(), ppeTask.getQuantPostShift(),
                                  ppeTask.getIn1QuantMult(), ppeTask.getIn2QuantMult(), ppeTask.getFpPreluAlpha());

    auto newOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(rewriter.clone(*nceOp));
    newOp.setPPE(ppeTaskAttr);
    rewriter.replaceOp(clampOp, newOp->getResult(0));
    if (nceOp.use_empty()) {
        rewriter.eraseOp(nceOp);
    }

    return mlir::success();
}

bool isLegalClamp(VPU::ClampOp clampOp) {
    // Clamp operations without NCE producer are legal.
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(clampOp.getInput().getDefiningOp());
    if (nceOp == nullptr) {
        return true;
    }

    // NCE producers without PPE attributes are covered in FusePostOps pass.
    if (!nceOp.getPPE().has_value()) {
        return true;
    }

    // If the operation is quantized, only per-tensor quantization is supported.
    // If the operation is has float16 or bfloat16 precision, the lower bound of the clamp must be zero.
    // Other data types are not supported.
    const auto outElemType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (outElemType.isa<mlir::quant::QuantizedType>()) {
        return outElemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
    } else if (outElemType.isF16() || outElemType.isBF16()) {
        const auto clampMin = clampOp.getMin().convertToDouble();
        return !isDoubleEqual(clampMin, 0.0);
    } else {
        return true;
    }
}

//
// FuseClampPass
//

class FuseClampPass final : public VPU::FuseClampPassBase<FuseClampPass> {
public:
    explicit FuseClampPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FuseClampPass::safeRunOnFunc() {
    // TODO: #70644

    auto func = getOperation();
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    // Adding the entire dialect here since the VPU.Clamp will be replaced with one of VPU.NCE operations.
    target.addLegalDialect<VPU::VPUDialect>();
    target.addDynamicallyLegalOp<VPU::ClampOp>(isLegalClamp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ClampConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseClampPass(Logger log) {
    return std::make_unique<FuseClampPass>(log);
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

#include <numeric>

#include "vpux/compiler/utils/attributes_properties_conversion.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {
namespace VPU {

// This operation based on input and output of Eltwise op will prepare final quantization scale value

double calculateQuantScaleVectorForEltwise(vpux::NDTypeInterface input1ShapedType,
                                           vpux::NDTypeInterface input2ShapedType,
                                           vpux::NDTypeInterface outputShapedType, VPU::ArchKind arch,
                                           bool isMultiplyOp) {
    const auto input1ElementType = input1ShapedType.getElementType();
    const auto input2ElementType = input2ShapedType.getElementType();
    const auto outputElementType = outputShapedType.getElementType();

    // In case of fully not quantized operation return
    if (!input1ElementType.isa<mlir::quant::QuantizedType>() && !input2ElementType.isa<mlir::quant::QuantizedType>() &&
        !outputElementType.isa<mlir::quant::QuantizedType>()) {
        return 1.0;
    }

    VPUX_THROW_WHEN(input1ElementType.isa<mlir::quant::UniformQuantizedPerAxisType>() ||
                            input2ElementType.isa<mlir::quant::UniformQuantizedPerAxisType>() ||
                            outputElementType.isa<mlir::quant::UniformQuantizedPerAxisType>(),
                    "Only per-tensor quantization is supported");

    double scaleInput1 = 0;
    double scaleOutput = 0;

    // floats in the compute pipeline are represented as S16.16 values
    // In order to convert from I32 to S16.16 and back, we need to multiply/divide by 1<<16
    // Depends on target hardware
    const double fp16_scale =
            (VPU::ArchKind::NPU37XX == arch || VPU::ArchKind::NPU40XX == arch) ? (1.0) : (1.0 / 65536);

    if (!input1ElementType.isa<mlir::quant::QuantizedType>() && !input2ElementType.isa<mlir::quant::QuantizedType>()) {
        scaleOutput = extractScalesAndZeroPoints(outputElementType).first.front();
        scaleInput1 = fp16_scale;
    } else if (!outputElementType.isa<mlir::quant::QuantizedType>()) {
        scaleInput1 = input1ElementType.isa<mlir::quant::QuantizedType>()
                              ? extractScalesAndZeroPoints(input1ElementType).first.front()
                              : 1.0;
        scaleOutput = fp16_scale;
    } else {
        scaleInput1 = input1ElementType.isa<mlir::quant::QuantizedType>()
                              ? extractScalesAndZeroPoints(input1ElementType).first.front()
                              : 1.0;
        scaleOutput = outputElementType.isa<mlir::quant::QuantizedType>()
                              ? extractScalesAndZeroPoints(outputElementType).first.front()
                              : 1.0;
    }

    VPUX_THROW_UNLESS(scaleInput1 != 0, "Invalid input scale value '0'");
    VPUX_THROW_UNLESS(scaleOutput != 0, "Invalid output scale value '0'");

    double ppeScale = 1.0;

    if (isMultiplyOp) {
        const auto scaleInput2 = input2ElementType.isa<mlir::quant::QuantizedType>()
                                         ? extractScalesAndZeroPoints(input2ElementType).first.front()
                                         : 1.0;
        VPUX_THROW_UNLESS(scaleInput2 != 0, "Invalid input scale value '0'");
        ppeScale = scaleInput1 * scaleInput2 / scaleOutput;
    } else {  // Add, Subtract, And
        ppeScale = scaleInput1 / scaleOutput;
    }

    return ppeScale;
}

double calculateQuantScaleVectorForAvgPool(vpux::NDTypeInterface inputShapedType,
                                           vpux::NDTypeInterface outputShapedType, ArrayRef<int64_t> filter_size,
                                           VPU::ArchKind arch) {
    const auto inputElementType = inputShapedType.getElementType();
    const auto outputElementType = outputShapedType.getElementType();
    const auto divisor = static_cast<double>(
            std::accumulate(filter_size.begin(), filter_size.end(), 1ll, std::multiplies<int64_t>()));

    // In case of fully not quantized operation return
    if (!inputElementType.isa<mlir::quant::QuantizedType>() && !outputElementType.isa<mlir::quant::QuantizedType>()) {
        return 1.0 / divisor;
    }

    VPUX_THROW_WHEN(inputElementType.isa<mlir::quant::UniformQuantizedPerAxisType>() ||
                            outputElementType.isa<mlir::quant::UniformQuantizedPerAxisType>(),
                    "Only per-tensor quantization is supported");

    // floats in the compute pipeline are represented as S16.16 values
    // In order to convert from I32 to S16.16 and back, we need to multiply/divide by 1<<16
    // Depends on target hardware
    const double fp16_scale =
            (VPU::ArchKind::NPU37XX == arch || VPU::ArchKind::NPU40XX == arch) ? (1.0) : (1.0 / 65536);

    auto scaleInput = fp16_scale;
    auto scaleOutput = fp16_scale;

    if (inputElementType.isa<mlir::quant::QuantizedType>())
        scaleInput = extractScalesAndZeroPoints(inputElementType).first.front();

    if (outputElementType.isa<mlir::quant::QuantizedType>())
        scaleOutput = extractScalesAndZeroPoints(outputElementType).first.front();

    VPUX_THROW_UNLESS(scaleInput != 0, "Invalid input scale value '0'");
    VPUX_THROW_UNLESS(scaleOutput != 0, "Invalid output scale value '0'");
    return scaleInput / scaleOutput / divisor;
}

VPU::PPETaskAttr getPPEAttr(const VPU::PostOpParams& postOpParams, mlir::MLIRContext* ctx) {
    if (postOpParams.quantParams.has_value()) {
        const auto quantParams = postOpParams.quantParams.value();
        return getPPETaskAttr(ctx, postOpParams.layerType, postOpParams.clampLow, postOpParams.clampHigh,
                              postOpParams.LreluMult, postOpParams.LreluShift, quantParams.quantMult,
                              quantParams.quantShift, quantParams.postShift);
    } else {
        return getPPETaskAttr(ctx, postOpParams.layerType, postOpParams.clampLow, postOpParams.clampHigh,
                              postOpParams.LreluMult, postOpParams.LreluShift);
    }
}

VPU::PPETaskAttr getPPETaskAttrFromPostOpsParams(mlir::Value opInput, mlir::Value opOutput,
                                                 vpux::IE::PostOpAttr postOpAttr, mlir::Location loc,
                                                 mlir::MLIRContext* ctx, VPU::ArchKind arch) {
    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    const auto postOpParams = VPU::parsePostOp(postOpAttr, inElemType, outElemType, arch, loc);

    VPU::PPETaskAttr ppeTaskAttr;
    if (postOpParams.has_value()) {
        ppeTaskAttr = VPU::getPPEAttr(postOpParams.value(), ctx);
    }

    return ppeTaskAttr;
}

VPU::PPETaskAttr getNCEAveragePoolPPETaskAttr(vpux::NDTypeInterface inputType, mlir::ArrayAttr kernelSizeAttr,
                                              vpux::NDTypeInterface outputType, vpux::IE::PostOpAttr postOpAttr,
                                              mlir::Location loc, mlir::MLIRContext* ctx, VPU::ArchKind arch) {
    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t LreluMult = 1;
    int64_t LreluShift = 0;
    VPU::PPEMode ppeType = VPU::PPEMode::NOOP;
    auto kernelSize = parseIntArrayAttr<int64_t>(kernelSizeAttr);

    const auto inputElemType = inputType.getElementType();
    const auto outputElemType = outputType.getElementType();

    if (auto outElemQType = outputElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputElemType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    const auto postOpParams = parsePostOp(postOpAttr, inputElemType, outputElemType, arch, loc);
    if (postOpParams.has_value()) {
        clampLow = postOpParams->clampLow;
        clampHigh = postOpParams->clampHigh;
        LreluMult = postOpParams->LreluMult;
        LreluShift = postOpParams->LreluShift;
        ppeType = postOpParams->layerType;
    }

    VPU::PPETaskAttr ppeAttr;
    // Since AvgPool operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto quantScale = VPU::calculateQuantScaleVectorForAvgPool(inputType, outputType, kernelSize, arch);
    // Input datatype entirely decides the precision of the compute pipeline.
    // Since NPU3720 we have a separation for float and integer compute pipelines
    // so it's best to make use of the correct pipeline.
    if ((VPU::ArchKind::NPU37XX == arch || VPU::ArchKind::NPU40XX == arch) &&
        !inputElemType.isa<mlir::quant::QuantizedType>()) {
        ppeAttr =
                getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift, ArrayRef<double>{quantScale});
    } else {
        const auto scaleApproximation = QuantizationApproximation(arch, quantScale);
        ppeAttr = getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift,
                                 ArrayRef<int64_t>{scaleApproximation.mult()},
                                 ArrayRef<int64_t>{scaleApproximation.shift()}, scaleApproximation.postShift());
    }

    return ppeAttr;
}

VPU::PPETaskAttr getNCEEltwisePPETaskAttr(vpux::NDTypeInterface input1Type, vpux::NDTypeInterface input2Type,
                                          vpux::NDTypeInterface outputType, vpux::IE::PostOpAttr postOpAttr,
                                          mlir::Location loc, VPU::EltwiseType opType, mlir::MLIRContext* ctx,
                                          VPU::ArchKind arch) {
    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t LreluMult = 1;
    int64_t LreluShift = 0;

    const auto input1ElemType = input1Type.getElementType();
    const auto input2ElemType = input2Type.getElementType();
    const auto outputElemType = outputType.getElementType();

    VPUX_THROW_UNLESS(
            input1ElemType.isa<mlir::quant::QuantizedType>() == input2ElemType.isa<mlir::quant::QuantizedType>(),
            "Not supporting mixed precision on the inputs of eltwise!");

    if (auto outElemQType = outputElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputElemType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    const auto postOpParams = parsePostOp(postOpAttr, input1ElemType, outputElemType, arch, loc);
    if (postOpParams.has_value()) {
        clampLow = postOpParams->clampLow;
        clampHigh = postOpParams->clampHigh;
        LreluMult = postOpParams->LreluMult;
        LreluShift = postOpParams->LreluShift;
    }

    VPU::PPEMode ppeType = VPU::getPPEMode(opType);
    // Since Eltwise operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto quantScale = VPU::calculateQuantScaleVectorForEltwise(input1Type, input2Type, outputType, arch,
                                                               opType == VPU::EltwiseType::MULTIPLY);
    if (supportsPerInputEltwiseScale(arch)) {
        // Input datatype entirely decides the precision of the compute pipeline.
        // Since NPU3720 we have a separation for float and integer compute pipelines
        // so it's best to make use of the correct pipeline.
        if (!input1ElemType.isa<mlir::quant::QuantizedType>() && !input2ElemType.isa<mlir::quant::QuantizedType>()) {
            return getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift,
                                  ArrayRef<double>{quantScale});
        } else if (input1ElemType.isa<mlir::quant::QuantizedType>() ||
                   input2ElemType.isa<mlir::quant::QuantizedType>()) {
            auto input1QuantScale = input1ElemType.isa<mlir::quant::QuantizedType>()
                                            ? extractScalesAndZeroPoints(input1ElemType).first.front()
                                            : 1.0;
            auto input2QuantScale = input2ElemType.isa<mlir::quant::QuantizedType>()
                                            ? extractScalesAndZeroPoints(input2ElemType).first.front()
                                            : 1.0;
            auto outputQuantScale = outputElemType.isa<mlir::quant::QuantizedType>()
                                            ? extractScalesAndZeroPoints(outputElemType).first.front()
                                            : 1.0;

            const auto allScaleApproximation =
                    EltwiseQuantizationApproximation(arch, input1QuantScale, input2QuantScale, outputQuantScale);
            return getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift,
                                  ArrayRef<int64_t>{allScaleApproximation.output().mult()},
                                  ArrayRef<int64_t>{allScaleApproximation.output().shift()},
                                  allScaleApproximation.output().postShift(),
                                  ArrayRef<int64_t>{allScaleApproximation.input1().mult()},
                                  ArrayRef<int64_t>{allScaleApproximation.input2().mult()});
        }
    }

    const auto scaleApproximation = QuantizationApproximation(arch, quantScale);
    return getPPETaskAttr(ctx, ppeType, clampLow, clampHigh, LreluMult, LreluShift,
                          ArrayRef<int64_t>{scaleApproximation.mult()}, ArrayRef<int64_t>{scaleApproximation.shift()},
                          scaleApproximation.postShift());
}

VPU::PostOpParams getPwlPostOpParams(const mlir::Type inElemType, const mlir::Type outElemType, VPU::PPEMode ppeType) {
    const int64_t clampLow = getPwlClamp(inElemType, outElemType, ppeType, true);
    const int64_t clampHigh = getPwlClamp(inElemType, outElemType, ppeType, false);
    const int64_t LreluMult = 1;
    const int64_t LreluShift = 0;
    const int64_t postShift = getPwlPostShift(ppeType);

    // Dummy values for mult & shift, as the actual values will be computed in the weights table
    return PostOpParams{ppeType,   clampLow,   clampHigh,
                        LreluMult, LreluShift, QuantizationParams{/*quantMult*/ {1}, /*quantShift*/ {0}, postShift}};
}

PostOpParams getCustomPwlPostOpParams(IE::PostOpAttr postOp, mlir::Type outElemType) {
    auto pwlTable = findCustomPWLTable(postOp, outElemType);

    VPUX_THROW_UNLESS(pwlTable.has_value(), "Custom PWL Table was not found for {0} {1}", postOp.getName().getValue(),
                      outElemType);

    auto pwlTableRange = pwlTable.value().range;

    VPUX_THROW_UNLESS(!pwlTableRange.empty(), "Custom PWL Table range is empty for {0} {1}",
                      postOp.getName().getValue(), outElemType);

    const int64_t clampLow = pwlTableRange[0];
    const int64_t clampHigh = pwlTableRange[pwlTableRange.size() - 1];
    const int64_t LreluMult = 1;
    const int64_t LreluShift = 0;
    const int64_t postShift = pwlTable.value().postShift;

    // Dummy values for mult & shift, as the actual values will be computed in the weights table
    return PostOpParams{VPU::PPEMode::FLEXARB,
                        clampLow,
                        clampHigh,
                        LreluMult,
                        LreluShift,
                        QuantizationParams{/*quantMult=*/{1}, /*quantShift=*/{0}, postShift}};
}

std::optional<VPU::PostOpParams> parsePostOp(IE::PostOpAttr postOp, const mlir::Type inElemType,
                                             const mlir::Type outElemType, VPU::ArchKind arch, mlir::Location loc) {
    if (postOp == nullptr) {
        return std::nullopt;
    }

    auto outElemQType = outElemType.dyn_cast<mlir::quant::QuantizedType>();
    int64_t clampLowQuantized = 0;
    int64_t clampHighQuantized = 0;
    if (outElemQType != nullptr) {
        const auto zps = extractScalesAndZeroPoints(outElemType).second;
        clampLowQuantized = outElemQType.getStorageTypeMin() - zps.front();
        clampHighQuantized = outElemQType.getStorageTypeMax() - zps.front();
    }

    if (postOp.getName().getValue() == IE::ReLUOp::getOperationName()) {
        VPUX_THROW_UNLESS(postOp.getAttrs().empty(), "'{0}' PostOp should not have any attributes", postOp.getName());

        int64_t clampLow = 0;
        int64_t clampHigh = (outElemQType != nullptr) ? clampHighQuantized : std::numeric_limits<int32_t>::max();
        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;

        return PostOpParams{VPU::PPEMode::LRELU, clampLow, clampHigh, LreluMult, LreluShift};
    } else if (postOp.getName().getValue() == IE::ClampOp::getOperationName()) {
        IE::ClampOp::Adaptor clamp(std::nullopt, nullptr, toProperties<IE::ClampOp>(postOp.getAttrs()));
        VPUX_THROW_UNLESS(clamp.verify(loc).succeeded(), "Wrong attributes '{0}' for '{1}' PostOp", postOp.getAttrs(),
                          postOp.getName());

        int64_t clampLow = 0;
        int64_t clampHigh = 0;
        if (outElemQType != nullptr) {
            auto clampLowHigh = getClampValuesForQuantizedOps(
                    {clamp.getMin().convertToDouble(), clamp.getMax().convertToDouble()}, outElemQType, outElemType);
            clampLow = clampLowHigh.first;
            clampHigh = clampLowHigh.second;
        } else {
            clampLow = vpux::VPU::NCESparsity::toFixedPoint(clamp.getMin().convertToDouble());
            clampHigh = vpux::VPU::NCESparsity::toFixedPoint(clamp.getMax().convertToDouble());
        }

        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;

        return PostOpParams{VPU::PPEMode::NOOP, clampLow, clampHigh, LreluMult, LreluShift};
    } else if (postOp.getName().getValue() == IE::LeakyReluOp::getOperationName()) {
        // PWL case
        if (arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX && outElemQType != nullptr) {
            return getCustomPwlPostOpParams(postOp, outElemType);
        }

        IE::LeakyReluOp::Adaptor leakyRelu(std::nullopt, nullptr, toProperties<IE::LeakyReluOp>(postOp.getAttrs()));
        VPUX_THROW_UNLESS(leakyRelu.verify(loc).succeeded(), "Wrong attributes '{0}' for '{1}' PostOp",
                          postOp.getAttrs(), postOp.getName());

        // On some architectures negative slope is applied before the clamping, there's no need to adjust bounds.
        const bool skipAlpha = (arch == VPU::ArchKind::NPU37XX || arch == VPU::ArchKind::NPU40XX);
        const auto alpha = leakyRelu.getNegativeSlope().convertToDouble();
        int32_t clampLow = skipAlpha ? std::numeric_limits<int32_t>::min()
                                     : static_cast<int32_t>(std::numeric_limits<int32_t>::min() / alpha);
        if (outElemQType != nullptr) {
            clampLow = skipAlpha ? static_cast<int32_t>(clampLowQuantized)
                                 : static_cast<int32_t>(clampLowQuantized / alpha);
        }

        int64_t clampHigh = (outElemQType != nullptr) ? clampHighQuantized : std::numeric_limits<int32_t>::max();
        int64_t preluMult = 1;
        int64_t preluShift = 0;
        if (isFloatEqual(static_cast<float>(alpha), 0.0)) {
            preluMult = 0;
        } else if (!isFloatEqual(static_cast<float>(alpha), 1.0)) {
            const auto alphaApproximation = PReLUApproximation(arch, alpha);
            preluMult = alphaApproximation.mult();
            preluShift = alphaApproximation.shift();
        }
        return PostOpParams{VPU::PPEMode::LPRELU, static_cast<int64_t>(clampLow), clampHigh, preluMult, preluShift};
    } else if (postOp.getName().getValue() == IE::SigmoidOp::getOperationName()) {
        return VPU::getPwlPostOpParams(inElemType, outElemType, VPU::PPEMode::SIGMOID);
    } else if (postOp.getName().getValue() == IE::TanhOp::getOperationName()) {
        return VPU::getPwlPostOpParams(inElemType, outElemType, VPU::PPEMode::TANH);
    }

    VPUX_THROW("Unsupported PostOp '{0}'", postOp.getName());
}

bool supportsPerInputEltwiseScale(const VPU::ArchKind arch) {
    return arch == VPU::ArchKind::NPU37XX || arch == VPU::ArchKind::NPU40XX;
}

}  // namespace VPU
}  // namespace vpux

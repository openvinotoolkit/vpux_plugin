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

bool supportsPerInputEltwiseScale(const VPU::ArchKind arch) {
    return arch == VPU::ArchKind::NPU37XX || arch == VPU::ArchKind::NPU40XX;
}

}  // namespace VPU
}  // namespace vpux

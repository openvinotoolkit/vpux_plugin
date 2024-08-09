//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isNCEEltwiseSupported(VPU::ArchKind arch, vpux::NDTypeInterface input1Type,
                                      vpux::NDTypeInterface input2Type, vpux::NDTypeInterface outputType,
                                      bool allowDifferentScales, bool allowDifferentZp, bool checkLayout,
                                      bool checkChannelAlignment, LogCb logCb) {
    if (input1Type.getRank() != 4 || input2Type.getRank() != 4 || outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (input1Type.getShape() != input2Type.getShape()) {
        logCb(formatv("Broadcasting is not supported"));
        return false;
    }

    // Output type can differ from input type. In case of quantization this can be different quant scale value.
    // Input types can also differ when both of them are quantized. E.g. scale value for Eltwise Multiply
    const auto input1ElemType = input1Type.getElementType();
    const auto input2ElemType = input2Type.getElementType();

    if (!input1ElemType.isa<mlir::quant::QuantizedType>() && !input2ElemType.isa<mlir::quant::QuantizedType>()) {
        if (!input1ElemType.isa<mlir::Float16Type>() || !input2ElemType.isa<mlir::Float16Type>()) {
            return false;
        }
    } else if (input1ElemType.isa<mlir::quant::UniformQuantizedType>() &&
               input2ElemType.isa<mlir::quant::UniformQuantizedType>()) {
        auto qInput1 = input1ElemType.cast<mlir::quant::UniformQuantizedType>();
        auto qInput2 = input2ElemType.cast<mlir::quant::UniformQuantizedType>();

        if (qInput1.getExpressedType() != qInput2.getExpressedType() ||
            qInput1.getStorageType() != qInput2.getStorageType() || qInput1.isSigned() != qInput2.isSigned()) {
            logCb(formatv("Mismatch in inputs quantization parameters"));
            return false;
        }

        if (!allowDifferentZp && qInput1.getZeroPoint() != qInput2.getZeroPoint()) {
            logCb(formatv("Mismatch in inputs zero points"));
            return false;
        }

        if (!allowDifferentScales &&
            !isFloatEqual(static_cast<float>(qInput1.getScale()), static_cast<float>(qInput2.getScale()))) {
            logCb(formatv("Mismatch in inputs quantization scales"));
            return false;
        }
    } else {
        logCb(formatv("Unsupported inputs element types"));
        return false;
    }

    if (checkChannelAlignment) {
        if (!NCEInvariant::isInputActTypeSupported(
                    arch, input1Type, vpux::VPU::NCEInvariant::getAlignment(input1Type.getElementType()), false) ||
            !NCEInvariant::isInputActTypeSupported(
                    arch, input2Type, vpux::VPU::NCEInvariant::getAlignment(input2Type.getElementType()), false) ||
            !NCEInvariant::isOutputActTypeSupported(
                    outputType, vpux::VPU::NCEInvariant::getAlignment(outputType.getElementType()))) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        if (!NCEInvariant::checkLayouts({input1Type, input2Type}, {outputType}, arch, 2, logCb)) {
            return false;
        }
    }

    return true;
}

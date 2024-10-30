//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/utils/convert_utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

mlir::LogicalResult vpux::Const::ConvertElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                             mlir::Type elemType) {
    if (elemType == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'ConvertElemTypeAttr'");
    }

    // TODO: Support quantization transformation
    if (mlir::isa<mlir::quant::QuantizedType>(elemType)) {
        return printTo(emitError(), "Unsupported conversion: 'outElemType = {1}'", elemType);
    }

    return mlir::success();
}

vpux::NDTypeInterface vpux::Const::ConvertElemTypeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input.changeElemType(getElemType());
}

bool vpux::Const::ConvertElemTypeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

Const::Content vpux::Const::ConvertElemTypeAttr::transform(vpux::Const::Content& input) const {
    auto inType = input.getType();
    auto outNDType = inferOutputType(inType);
    auto outputIsSplat = inferOutputSplat(input.isSplat(), inType);

    if (auto qElemType = inType.getElementType().dyn_cast<mlir::quant::QuantizedType>()) {
        // TODO: Support dequantization transformation
        VPUX_THROW("Unsupported conversion: {0} -> {1}", qElemType, outNDType.getElementType());
    }

    // For subbyte type, we unpack the data
    auto bitWidth = inType.getElemTypeSize().count();
    if (bitWidth < CHAR_BIT && outNDType.getElementType().isInteger(8)) {
        return subByteConversion(input, outNDType, outputIsSplat, bitWidth);
    }

    // TODO: Support generic transformation
    VPUX_THROW("Unsupported conversion: {0} -> {1}", inType.getElementType(), outNDType.getElementType());
}

Const::ContentSetup vpux::Const::ContentSetup::convertElemType(mlir::Type newElemType) {
    return addTransformation(Const::ConvertElemTypeAttr::get(newElemType));
}

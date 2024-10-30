//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// QuantCastAttr::verify
//

mlir::LogicalResult vpux::Const::QuantCastAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError, mlir::Type type) {
    if (type == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'QuantCastAttr'");
    }

    return mlir::success();
}

//
// QuantCastAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::QuantCastAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto outElemType = getElemType();

    if (auto qElemType = outElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto quantStorageType = normalizeQuantStorageType(qElemType);
        VPUX_THROW_UNLESS(input.getElementType() == quantStorageType, "Can't cast '{0}' element type to '{1}'",
                          input.getElementType(), outElemType);

        return input.changeElemType(outElemType);
    }

    const auto inQuantType = input.getElementType().dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(inQuantType != nullptr, "Unable to restore storage type from non-quantized type");

    auto quantStorageType = normalizeQuantStorageType(inQuantType);
    VPUX_THROW_UNLESS(quantStorageType == outElemType, "Can't cast '{0}' element type to '{1}'", inQuantType,
                      outElemType);

    return input.changeElemType(quantStorageType);
}

bool vpux::Const::QuantCastAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

//
// QuantCastAttr::transform
//

Const::Content vpux::Const::QuantCastAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

Const::ContentSetup vpux::Const::ContentSetup::quantCast(mlir::Type newElemType) {
    return addTransformation(Const::QuantCastAttr::get(newElemType));
}

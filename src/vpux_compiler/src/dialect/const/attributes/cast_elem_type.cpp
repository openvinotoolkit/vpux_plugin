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
// CastElemTypeAttr::verify
//

mlir::LogicalResult vpux::Const::CastElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                          mlir::Type type) {
    if (type == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'CastElemTypeAttr'");
    }

    return mlir::success();
}

//
// CastElemTypeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::CastElemTypeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto outElemType = getElemType();

    if (auto qElemType = outElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto quantStorageType = normalizeQuantStorageType(qElemType);
        VPUX_THROW_UNLESS(input.getElementType() == quantStorageType, "Can't cast '{0}' element type to '{1}'",
                          input.getElementType(), outElemType);

        return input.changeElemType(outElemType);
    }

    if (auto inQuantType = input.getElementType().dyn_cast<mlir::quant::QuantizedType>()) {
        auto quantStorageType = normalizeQuantStorageType(inQuantType);
        VPUX_THROW_UNLESS(quantStorageType == outElemType, "Can't cast '{0}' element type to '{1}'", inQuantType,
                          outElemType);

        return input.changeElemType(quantStorageType);
    }

    VPUX_THROW_UNLESS(input.getElementType().isIntOrFloat(), "Can't cast '{0}' element type to '{1}'",
                      input.getElementType(), outElemType);
    return input.changeElemType(outElemType);
}

bool vpux::Const::CastElemTypeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

//
// CastElemTypeAttr::transform
//

Const::Content vpux::Const::CastElemTypeAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

Const::ContentSetup vpux::Const::ContentSetup::castElemType(mlir::Type newElemType) {
    return addTransformation(Const::CastElemTypeAttr::get(newElemType));
}

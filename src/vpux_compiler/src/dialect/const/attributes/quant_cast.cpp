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

mlir::LogicalResult vpux::Const::QuantCastAttr::verify(FuncRef<mlir::InFlightDiagnostic()>,
                                                       mlir::quant::QuantizedType) {
    return mlir::success();
}

//
// QuantCastAttr::print
//

void vpux::Const::QuantCastAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    if (const auto elemTypeVal = getElemType()) {
        printer.printType(elemTypeVal);
    }
    printer << ">";
}

//
// QuantCastAttr::parse
//

mlir::Attribute vpux::Const::QuantCastAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::quant::QuantizedType elemType;
    parser.parseOptionalType(elemType);

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::QuantCastAttr>(parser.getContext(), elemType);
}

//
// QuantCastAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::QuantCastAttr::inferOutputType(vpux::NDTypeInterface input) const {
    if (const auto elemTypeVal = getElemType()) {
        const auto quantStorateType = normalizeQuantStorageType(elemTypeVal);

        VPUX_THROW_UNLESS(input.getElementType() == quantStorateType, "Can't cast '{0}' element type to '{1}'",
                          input.getElementType(), getElemType());

        return input.changeElemType(elemTypeVal);
    }

    const auto quantType = input.getElementType().dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(quantType != nullptr, "Unable to restore storage type from non-quantized type");

    return input.changeElemType(normalizeQuantStorageType(quantType));
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

//
// ContentAttr::quantCast
//

Const::ContentAttr vpux::Const::ContentAttr::quantCast(mlir::quant::QuantizedType newElemType) const {
    return ContentAttr::addTransformation(
            *this, Const::QuantCastAttr::get(getContext(), newElemType).cast<Const::TransformAttrInterface>());
}

//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/DialectImplementation.h>
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

//
// ChangeShapeAndElemTypeAttr::verify
//

mlir::LogicalResult vpux::Const::ChangeShapeAndElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                                    mlir::ArrayAttr shape, mlir::Type) {
    if (shape == nullptr) {
        return printTo(emitError(), "Got NULL 'shape' in 'ChangeShapeAndElemTypeAttr'");
    }

    const auto shapeValues = shape.getValue();
    for (const auto& dimAttr : shapeValues) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value '{0}' in 'shape' for 'ChangeShapeAndElemTypeAttr'",
                           dimAttr);
        }
        if (dimAttr.cast<mlir::IntegerAttr>().getInt() <= 0) {
            return printTo(emitError(),
                           "Got unsupported dimension value '{0}' in 'shape' for 'ChangeShapeAndElemTypeAttr'",
                           dimAttr);
        }
    }

    return mlir::success();
}

//
// ChangeShapeAndElemTypeAttr::print
//

void vpux::Const::ChangeShapeAndElemTypeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getShape());
    printer << ", ";
    printer.printType(getElemType());
    printer << ">";
}

//
// ChangeShapeAndElemTypeAttr::parse
//

mlir::Attribute vpux::Const::ChangeShapeAndElemTypeAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr shape;
    if (mlir::failed(parser.parseAttribute(shape))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::Type elemType;
    if (mlir::failed(parser.parseType(elemType))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::ChangeShapeAndElemTypeAttr::get(shape, elemType);
}

//
// ChangeShapeAndElemTypeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ChangeShapeAndElemTypeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto newElemType = getElemType();
    const auto newShape = parseIntArrayAttr<int64_t>(getShape());
    return input.changeShapeElemType(ShapeRef(newShape), newElemType);
}

//
// ChangeShapeAndElemTypeAttr::transform
//

Const::Content vpux::Const::ChangeShapeAndElemTypeAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

//
// ContentAttr::changeShapeAndElemType
//

Const::ContentAttr vpux::Const::ContentAttr::changeShapeAndElemType(ShapeRef newShape, mlir::Type newElemType) const {
    return ContentAttr::addTransformation(
            *this, Const::ChangeShapeAndElemTypeAttr::get(getIntArrayAttr(getContext(), newShape), newElemType)
                           .cast<Const::TransformAttrInterface>());
}

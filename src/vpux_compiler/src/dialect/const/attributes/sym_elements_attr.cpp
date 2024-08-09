//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

mlir::Attribute Const::SymElementsAttr::parse(::mlir::AsmParser& odsParser, ::mlir::Type) {
    if (mlir::failed(odsParser.parseLess())) {
        return {};
    }

    auto symRefAttr = mlir::FieldParser<mlir::SymbolRefAttr>::parse(odsParser);

    if (mlir::failed(symRefAttr)) {
        return {};
    }

    if (mlir::failed(odsParser.parseGreater())) {
        return {};
    }

    if (mlir::failed(odsParser.parseColon())) {
        return {};
    }

    auto type = mlir::FieldParser<mlir::ShapedType>::parse(odsParser);

    if (mlir::failed(type)) {
        return {};
    }

    return SymElementsAttr::get(odsParser.getContext(), symRefAttr.value(), type.value());
}

void Const::SymElementsAttr::print(::mlir::AsmPrinter& odsPrinter) const {
    odsPrinter << "<" << getSymName() << ">"
               << " : " << getShapedType();
}

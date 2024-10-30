//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include <mlir/IR/DialectImplementation.h>

mlir::Attribute vpux::Const::MultiContentSymbolAttr::parse(mlir::AsmParser& odsParser, mlir::Type) {
    // What we are trying to parse:
    // @some::@symbol : type [, list_of_transformations]

    auto symRefAttr = mlir::FieldParser<mlir::SymbolRefAttr>::parse(odsParser);

    if (mlir::failed(symRefAttr)) {
        return {};
    }

    if (mlir::failed(odsParser.parseColon())) {
        return {};
    }

    auto bundleSymbolType = mlir::FieldParser<mlir::ShapedType>::parse(odsParser);

    if (mlir::failed(bundleSymbolType)) {
        return {};
    }

    if (mlir::succeeded(odsParser.parseOptionalComma())) {
        auto transformations = mlir::FieldParser<TransformAttrInterfaceArrayAttr>::parse(odsParser);

        if (mlir::failed(transformations)) {
            return {};
        }

        return MultiContentSymbolAttr::get(odsParser.getContext(), symRefAttr.value(), bundleSymbolType.value(),
                                           transformations.value());
    }

    return MultiContentSymbolAttr::get(odsParser.getContext(), symRefAttr.value(), bundleSymbolType.value(),
                                       TransformAttrInterfaceArrayAttr::get(odsParser.getContext(), {}));
}

void vpux::Const::MultiContentSymbolAttr::print(::mlir::AsmPrinter& odsPrinter) const {
    odsPrinter << getBundleSymbol() << " : " << getBundleSymbolType();

    if (const auto transformations = getTransformations().getValue(); !transformations.empty()) {
        odsPrinter << ", "
                   << "[" << transformations << "]";
    }
}

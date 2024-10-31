//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectResourceBlobManager.h>

mlir::LogicalResult vpux::Const::MultiContentAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                          vpux::Const::ElementsAttrArrayAttr baseContent,
                                                          vpux::Const::TransformAttrInterfaceArrayAttr transformations,
                                                          vpux::NDTypeInterface finalType) {
    std::ignore = finalType;  // inferred

    if (baseContent.empty()) {
        return printTo(emitError(), "'baseContent' cannot be empty");
    }

    // verification is mostly identical to ContentAttr
    for (auto content : baseContent) {
        if (auto result = ContentAttr::verify(emitError, content, transformations); mlir::failed(result)) {
            return result;
        }

        // SymElementsAttr for example is not allowed - in contrast to ContentAttr
        if (!llvm::isa<mlir::DenseElementsAttr, mlir::DenseResourceElementsAttr>(content)) {
            return printTo(emitError(), "Got unsupported type '{0}' for 'baseContent' in 'MultiContentAttr'",
                           content.getType());
        }
    }

    // Having the same base type in all 'baseContent' elements implies that all applications of 'transformations' yields
    // the same final type.
    bool allBaseTypesEqual = llvm::all_equal(llvm::map_range(baseContent, [transformations](auto content) {
        return Const::inferFinalType(content.getType(), transformations);
    }));
    if (!allBaseTypesEqual) {
        return printTo(emitError(), "All base types in 'baseContent' must be equal");
    }

    return mlir::success();
}

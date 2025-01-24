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

    SmallVector<mlir::Type> contentTypes;
    contentTypes.reserve(baseContent.size());

    // verification is mostly identical to ContentAttr
    for (auto content : baseContent) {
        const auto [type, isSplat] = vpux::Const::inferFinalTypeAndSplat(content, transformations.getValue());
        const mlir::UnitAttr splatAttr = isSplat ? mlir::UnitAttr::get(baseContent.getContext()) : nullptr;
        if (auto result = ContentAttr::verify(emitError, content, transformations, type, splatAttr);
            mlir::failed(result)) {
            return result;
        }

        // SymElementsAttr for example is not allowed - in contrast to ContentAttr
        if (!llvm::isa<mlir::DenseElementsAttr, mlir::DenseResourceElementsAttr>(content)) {
            return printTo(emitError(), "Got unsupported type '{0}' for 'baseContent' in 'MultiContentAttr'",
                           content.getType());
        }

        contentTypes.push_back(type);
    }

    // Having the same base type in all 'baseContent' elements implies that all applications of 'transformations' yields
    // the same final type.
    bool allBaseTypesEqual = llvm::all_equal(contentTypes);
    if (!allBaseTypesEqual) {
        return printTo(emitError(), "All base types in 'baseContent' must be equal");
    }

    return mlir::success();
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpImplementation.h>

namespace vpux {

//
// OptionalTypes
//

void printOptionalTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::TypeRange types);
mlir::ParseResult parseOptionalTypes(mlir::OpAsmParser& parser, SmallVectorImpl<mlir::Type>& types);

template <typename... Args>
void printOptionalTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::Type type, Args... types) {
    printOptionalTypes(printer, op, mlir::TypeRange({type, types...}));
}

namespace details {

mlir::ParseResult parseOptionalTypes(mlir::OpAsmParser& parser, ArrayRef<mlir::Type*> types);

}  // namespace details

template <typename... Args>
mlir::ParseResult parseOptionalTypes(mlir::OpAsmParser& parser, mlir::Type& type, Args&... types) {
    return details::parseOptionalTypes(parser, ArrayRef({&type, &types...}));
}

//
// OptionalResultTypes
//

void printOptionalResultTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::TypeRange types);
mlir::ParseResult parseOptionalResultTypes(mlir::OpAsmParser& parser, SmallVectorImpl<mlir::Type>& types);

template <typename... Args>
void printOptionalResultTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::Type type, Args... types) {
    SmallVector<mlir::Type> nonNullTypes;
    if (type != nullptr) {
        nonNullTypes.push_back(type);
    }
    for (auto extraType : {types...}) {
        if (extraType != nullptr) {
            nonNullTypes.push_back(extraType);
        }
    }

    if (!nonNullTypes.empty()) {
        printOptionalResultTypes(printer, op, mlir::TypeRange(ArrayRef(nonNullTypes)));
    }
}

namespace details {

mlir::ParseResult parseOptionalResultTypes(mlir::OpAsmParser& parser, ArrayRef<mlir::Type*> types);

}  // namespace details

template <typename... Args>
mlir::ParseResult parseOptionalResultTypes(mlir::OpAsmParser& parser, mlir::Type& type, Args&... types) {
    return details::parseOptionalResultTypes(parser, ArrayRef({&type, &types...}));
}

//
// OptionalRegion
//

void printOptionalRegion(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::Region& region);
mlir::ParseResult parseOptionalRegion(mlir::OpAsmParser& parser, mlir::Region& region);

//
// OptionalBlockRegion
//

void printOptionalBlockRegion(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::Region& region);
mlir::ParseResult parseOptionalBlockRegion(mlir::OpAsmParser& parser, mlir::Region& region);

//
// Assembly print/parse utils
//

void printGroupOfOperands(mlir::OpAsmPrinter& p, mlir::Block* entry, mlir::StringRef groupName,
                          mlir::ValueRange operands, unsigned& opIdx);
mlir::ParseResult parseGroupOfOperands(mlir::OpAsmParser& parser, mlir::OperationState& result,
                                       SmallVector<mlir::OpAsmParser::Argument>& blockArgs,
                                       SmallVector<mlir::Type>& blockTypes, mlir::StringRef groupName, int32_t& count);

//
// Dynamic tensor dims
//

mlir::ParseResult parseTensorDims(mlir::OpAsmParser& parser, mlir::OperationState& result, mlir::StringRef groupName,
                                  int32_t& dimsCount);

}  // namespace vpux

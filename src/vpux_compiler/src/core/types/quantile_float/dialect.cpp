// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/dialect.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"

namespace vpux {
namespace type {

QuantileFloatDialect::~QuantileFloatDialect() = default;

void QuantileFloatDialect::initialize() {
    registerTypes();
}

void QuantileFloatDialect::registerTypes() {
    addTypes<vpux::type::NF4Type>();
}

mlir::OptionalParseResult QuantileFloatDialect::generatedTypeParser(mlir::AsmParser& parser, llvm::StringRef* mnemonic,
                                                                    mlir::Type& value) {
    return mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
            .Case(vpux::type::QuantileFloatType::getMnemonic(),
                  [&](llvm::StringRef, llvm::SMLoc) {
                      value = vpux::type::QuantileFloatType::parse(parser);
                      return mlir::success(!!value);
                  })
            .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
                *mnemonic = keyword;
                return std::nullopt;
            });
}

mlir::LogicalResult QuantileFloatDialect::generatedTypePrinter(mlir::Type def, mlir::AsmPrinter& printer) {
    return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(def)
            .Case<vpux::type::QuantileFloatType>([&](auto t) {
                printer << vpux::type::QuantileFloatType::getMnemonic();
                t.print(printer);
                return ::mlir::success();
            })
            .Default([](auto) {
                return ::mlir::failure();
            });
}

mlir::Type QuantileFloatDialect::parseType(mlir::DialectAsmParser& parser) const {
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    llvm::StringRef mnemonic;
    mlir::Type genType;
    auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
    if (parseResult.has_value())
        return genType;

    parser.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `" << getNamespace() << "`";
    return {};
}

/// Print a type registered to this dialect.
void QuantileFloatDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
    if (mlir::succeeded(generatedTypePrinter(type, printer)))
        return;
};

}  // namespace type
}  // namespace vpux

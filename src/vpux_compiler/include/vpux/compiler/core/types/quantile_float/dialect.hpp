// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

using namespace llvm;
using namespace mlir;
using namespace mlir::detail;

namespace vpux {
namespace type {

class QuantileFloatDialect : public mlir::Dialect {
private:
    void registerTypes();
    void initialize();

    static mlir::OptionalParseResult generatedTypeParser(mlir::AsmParser& parser, llvm::StringRef* mnemonic,
                                                         mlir::Type& value);
    static mlir::LogicalResult generatedTypePrinter(mlir::Type def, mlir::AsmPrinter& printer);

public:
    explicit QuantileFloatDialect(mlir::MLIRContext* ctx)
            : mlir::Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<QuantileFloatDialect>()) {
        initialize();
    }

    ~QuantileFloatDialect() override;
    static constexpr ::llvm::StringLiteral getDialectNamespace() {
        return ::llvm::StringLiteral("QuantileFloat");
    }

    mlir::Type parseType(mlir::DialectAsmParser& parser) const override;

    /// Print a type registered to this dialect.
    void printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const override;
};

}  // namespace type
}  // namespace vpux

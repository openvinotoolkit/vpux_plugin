//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/asm.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/STLExtras.h>

using namespace vpux;

//
// OptionalTypes
//

namespace {

bool isDefinedAbove(mlir::Value val, mlir::Operation* user) {
    auto* producer = val.getDefiningOp();
    if (producer == nullptr) {
        return true;
    }

    if (producer->getBlock() == user->getBlock()) {
        return producer->isBeforeInBlock(user);
    }

    auto* producerRegion = producer->getParentRegion();
    auto* userAncestor = producerRegion->findAncestorOpInRegion(*user);

    return userAncestor != nullptr && producer->isBeforeInBlock(userAncestor);
}

bool canSkipOperandTypes(mlir::Operation* op) {
    return llvm::all_of(op->getOperands(), [op](mlir::Value val) {
        return isDefinedAbove(val, op);
    });
}

}  // namespace

void vpux::printOptionalTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::TypeRange types) {
    if (canSkipOperandTypes(op)) {
        return;
    }

    const auto nonNull = [](mlir::Type type) {
        return type != nullptr;
    };
    VPUX_THROW_UNLESS(llvm::all_of(types, nonNull), "Optional values are not supported");

    printer << ": ";
    llvm::interleaveComma(types, printer);
}

mlir::ParseResult vpux::parseOptionalTypes(mlir::OpAsmParser& parser, SmallVectorImpl<mlir::Type>& types) {
    if (parser.parseOptionalColonTypeList(types)) {
        return mlir::success();
    }

    return mlir::success();
}

mlir::ParseResult vpux::details::parseOptionalTypes(mlir::OpAsmParser& parser, ArrayRef<mlir::Type*> dst) {
    SmallVector<mlir::Type> types;
    if (parser.parseOptionalColonTypeList(types)) {
        return mlir::success();
    }

    if (types.empty()) {
        return mlir::success();
    }

    if (types.size() != dst.size()) {
        return printTo(parser.emitError(parser.getCurrentLocation()),
                       "Got unexpected number of types '{0}', expected '{1}'", types.size(), dst.size());
    }

    for (auto ind : irange(dst.size())) {
        *dst[ind] = types[ind];
    }

    return mlir::success();
}

//
// OptionalResultTypes
//

void vpux::printOptionalResultTypes(mlir::OpAsmPrinter& printer, mlir::Operation* /*op*/, mlir::TypeRange types) {
    printer << ", ";
    llvm::interleaveComma(types, printer);
}

mlir::ParseResult vpux::parseOptionalResultTypes(mlir::OpAsmParser& parser, SmallVectorImpl<mlir::Type>& types) {
    if (mlir::failed(parser.parseOptionalComma())) {
        return mlir::success();
    }

    if (parser.parseTypeList(types)) {
        return mlir::success();
    }

    return mlir::success();
}

mlir::ParseResult vpux::details::parseOptionalResultTypes(mlir::OpAsmParser& parser, ArrayRef<mlir::Type*> dst) {
    if (mlir::failed(parser.parseOptionalComma())) {
        return mlir::success();
    }

    SmallVector<mlir::Type> types;
    if (parser.parseTypeList(types)) {
        return mlir::success();
    }

    if (types.empty()) {
        return mlir::success();
    }

    if (types.size() < dst.size()) {
        // TODO: remove hardcoded support for results of VPUIP.NCEClusterTask
        // VPUIP.NCEClusterTask has two optional results:
        //   - first, expecting MemRefOf<I1> or VPUIP_DistributedBuffer
        //   - second, expecting MemRefOf<UI64>
        // The output containers are currently populated based on the parsed type
        if (types.size() == 1 && dst.size() == 2) {
            auto type = types.front();
            if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
                if (memref.getElementType().isUnsignedInteger(64)) {
                    *dst[1] = type;
                } else {
                    *dst[0] = type;
                }
            } else {
                *dst[0] = type;
            }
            return mlir::success();
        }
        VPUX_THROW("Currently unsupported case: '{0}' types and '{1}' expected", types.size(), dst.size());
    }

    if (types.size() != dst.size()) {
        return printTo(parser.emitError(parser.getCurrentLocation()),
                       "Got unexpected number of types '{0}', expected '{1}'", types.size(), dst.size());
    }

    for (auto ind : irange(dst.size())) {
        *dst[ind] = types[ind];
    }

    return mlir::success();
}

//
// OptionalRegion
//

void vpux::printOptionalRegion(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::Region& region) {
    if (region.empty()) {
        return;
    }
    printer.printRegion(region);
}

mlir::ParseResult vpux::parseOptionalRegion(mlir::OpAsmParser& parser, mlir::Region& region) {
    const auto res = parser.parseOptionalRegion(region);
    return res.has_value() ? res.value() : mlir::success();
}

//
// OptionalBlockRegion
//

void vpux::printOptionalBlockRegion(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::Region& region) {
    if (region.empty()) {
        return;
    }
    if (region.hasOneBlock() && region.front().empty()) {
        return;
    }
    printer.printRegion(region);
}

mlir::ParseResult vpux::parseOptionalBlockRegion(mlir::OpAsmParser& parser, mlir::Region& region) {
    const auto res = parser.parseOptionalRegion(region);
    if (res.has_value()) {
        return res.value();
    }

    region.emplaceBlock();
    return mlir::success();
}

void vpux::printGroupOfOperands(mlir::OpAsmPrinter& p, mlir::Block* entry, mlir::StringRef groupName,
                                mlir::ValueRange operands, unsigned& opIdx) {
    p << " " << groupName << "(";
    llvm::interleaveComma(operands, p, [&](mlir::Value operand) mutable {
        auto argument = entry->getArgument(opIdx++);
        p << operand << " as " << argument << ": " << argument.getType();
    });
    p << ")";
}

mlir::ParseResult vpux::parseGroupOfOperands(mlir::OpAsmParser& parser, mlir::OperationState& result,
                                             SmallVector<mlir::OpAsmParser::Argument>& blockArgs,
                                             SmallVector<mlir::Type>& blockTypes, mlir::StringRef groupName,
                                             int32_t& count) {
    if (parser.parseKeyword(groupName)) {
        return mlir::failure();
    }

    SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    SmallVector<mlir::Type> operandRawTypes;

    // Parse a single instance of `%operand as %blockArg : <type>`.
    auto parseOperands = [&]() -> mlir::ParseResult {
        if (parser.parseOperand(operands.emplace_back()) || parser.parseKeyword("as") ||
            parser.parseArgument(blockArgs.emplace_back()) || parser.parseColonType(blockTypes.emplace_back())) {
            return mlir::failure();
        }

        operandRawTypes.push_back(mlir::Type{});
        blockArgs.back().type = blockTypes.back();
        count++;
        return mlir::success();
    };

    auto argsLoc = parser.getCurrentLocation();
    if (parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::OptionalParen, parseOperands) ||
        parser.resolveOperands(operands, operandRawTypes, argsLoc, result.operands)) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::ParseResult vpux::parseTensorDims(mlir::OpAsmParser& parser, mlir::OperationState& result,
                                        mlir::StringRef groupName, int32_t& dimsCount) {
    if (succeeded(parser.parseOptionalKeyword(groupName))) {
        SmallVector<mlir::OpAsmParser::UnresolvedOperand> tensorDimsOperands;
        SmallVector<mlir::Type> tensorDimsTypes;

        // Parse a single instance of `%operand: <type>`
        const auto parseOperand = [&]() -> mlir::ParseResult {
            mlir::OpAsmParser::UnresolvedOperand operand;
            const auto parsedOperandResult = parser.parseOptionalOperand(operand);
            if (parsedOperandResult.has_value()) {
                if (failed(*parsedOperandResult)) {
                    return mlir::failure();
                }
                tensorDimsOperands.push_back(operand);
            }

            if (parser.parseColon()) {
                return mlir::failure();
            }

            mlir::Type optionalType;
            const auto parsedTypeResult = parser.parseOptionalType(optionalType);
            if (parsedTypeResult.has_value()) {
                if (failed(*parsedTypeResult)) {
                    return mlir::failure();
                }
                tensorDimsTypes.push_back(optionalType);
            }

            dimsCount++;
            return mlir::success();
        };

        auto argsLoc = parser.getCurrentLocation();

        // Parse a range of `(%operand: <type>, ...)`
        if (parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::Paren, parseOperand)) {
            return mlir::failure();
        }

        if (parser.resolveOperands(tensorDimsOperands, tensorDimsTypes, argsLoc, result.operands)) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

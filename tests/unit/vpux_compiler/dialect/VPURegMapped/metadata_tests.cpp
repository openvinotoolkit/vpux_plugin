//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

mlir::IntegerType getSInt1Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Signed);
}

mlir::IntegerType getUInt1Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Unsigned);
}

std::vector<std::pair<mlir::Type, elf::OVNodeType>> getMLIR2OVTypes(mlir::MLIRContext* ctx) {
    return std::vector<std::pair<mlir::Type, elf::OVNodeType>>{
            // Float
            std::make_pair(mlir::Float64Type::get(ctx), elf::OVNodeType::OVNodeType_F64),
            std::make_pair(mlir::Float32Type::get(ctx), elf::OVNodeType::OVNodeType_F32),
            std::make_pair(mlir::Float16Type::get(ctx), elf::OVNodeType::OVNodeType_F16),
            std::make_pair(mlir::BFloat16Type::get(ctx), elf::OVNodeType::OVNodeType_BF16),
            std::make_pair(mlir::Float8E5M2Type::get(ctx), elf::OVNodeType::OVNodeType_F8E5M2),
            std::make_pair(mlir::Float8E4M3FNType::get(ctx), elf::OVNodeType::OVNodeType_F8E4M3),

            // Signed
            std::make_pair(getSInt64Type(ctx), elf::OVNodeType::OVNodeType_I64),
            std::make_pair(getSInt32Type(ctx), elf::OVNodeType::OVNodeType_I32),
            std::make_pair(getSInt16Type(ctx), elf::OVNodeType::OVNodeType_I16),
            std::make_pair(getSInt8Type(ctx), elf::OVNodeType::OVNodeType_I8),
            std::make_pair(getSInt4Type(ctx), elf::OVNodeType::OVNodeType_I4),
            std::make_pair(getSInt1Type(ctx), elf::OVNodeType::OVNodeType_U1),

            // Unsigned
            std::make_pair(getUInt64Type(ctx), elf::OVNodeType::OVNodeType_U64),
            std::make_pair(getUInt32Type(ctx), elf::OVNodeType::OVNodeType_U32),
            std::make_pair(getUInt16Type(ctx), elf::OVNodeType::OVNodeType_U16),
            std::make_pair(getUInt8Type(ctx), elf::OVNodeType::OVNodeType_U8),
            std::make_pair(getUInt4Type(ctx), elf::OVNodeType::OVNodeType_U4),
            std::make_pair(getUInt1Type(ctx), elf::OVNodeType::OVNodeType_U1),

            // Signless
            std::make_pair(getInt64Type(ctx), elf::OVNodeType::OVNodeType_U64),
            std::make_pair(getInt32Type(ctx), elf::OVNodeType::OVNodeType_U32),
            std::make_pair(getInt16Type(ctx), elf::OVNodeType::OVNodeType_U16),
            // Signless 8-bit integer use for BOOL, to distinguish it from U8
            std::make_pair(getBool8Type(ctx), elf::OVNodeType::OVNodeType_BOOLEAN),
            std::make_pair(getInt4Type(ctx), elf::OVNodeType::OVNodeType_U4),
            std::make_pair(getInt1Type(ctx), elf::OVNodeType::OVNodeType_U1)};
}

}  // namespace

using MLIR_Metadata = MLIR_UnitBase;

TEST_F(MLIR_Metadata, TypeConversion) {
    mlir::MLIRContext ctx(registry);
    auto mlir2ovTypes = getMLIR2OVTypes(&ctx);

    std::for_each(mlir2ovTypes.begin(), mlir2ovTypes.end(),
                  [](const std::pair<mlir::Type, elf::OVNodeType>& mlir2ovType) {
                      EXPECT_EQ(ELFNPU37XX::createOVNodeType(mlir2ovType.first), mlir2ovType.second);
                  });
}

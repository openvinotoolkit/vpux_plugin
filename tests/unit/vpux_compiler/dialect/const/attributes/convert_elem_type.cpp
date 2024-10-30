//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/utils/convert_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "common/utils.hpp"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectResourceBlobManager.h>

#include <gtest/gtest.h>

using namespace vpux;

class MLIR_SubByteTest : public MLIR_UnitBase {
public:
    mlir::MLIRContext ctx;

public:
    MLIR_SubByteTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }
};

TEST_F(MLIR_SubByteTest, ConvertElemTypeI4ToI8_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4}, getSInt4Type(&ctx));
    const char splatVal = 0x66;
    const std::vector<char> inputVals = {splatVal};
    constexpr auto noop = [](void*, size_t, size_t) {};
    constexpr bool isMutable = false;

    mlir::AsmResourceBlob blob(mlir::ArrayRef<char>(inputVals), noop, isMutable);
    auto& manager = mlir::DenseResourceElementsHandle::getManagerInterface(&ctx);
    const auto baseAttr = mlir::DenseResourceElementsAttr::get(
            baseType, manager.insert("FromSplatDenseResourceElementsAttr", std::move(blob)));

    auto contentSetup = Const::ContentSetup(baseAttr);
    auto conversionType = getSInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = contentSetup.get();
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    EXPECT_EQ(content.getSplatValue<char>(), 0x06);
    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], 0x06);
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeI4ToI8) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 4}, getSInt4Type(&ctx));
    std::vector<char> inputVals{0x66, 0x75, 0x2f, 0x01};
    constexpr auto noop = [](void*, size_t, size_t) {};
    constexpr bool isMutable = false;

    mlir::AsmResourceBlob blob(mlir::ArrayRef<char>(inputVals), noop, isMutable);
    auto& manager = mlir::DenseResourceElementsHandle::getManagerInterface(&ctx);
    const auto baseAttr = mlir::DenseResourceElementsAttr::get(
            baseType, manager.insert("FromSplatDenseResourceElementsAttr", std::move(blob)));

    auto contentSetup = Const::ContentSetup(baseAttr);
    auto conversionType = getSInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = contentSetup.get();
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    std::vector<char> expectedVals{0x06, 0x06, 0x05, 0x07, 0x0f, 0x02, 0x01, 0x00};
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeI1ToI8_Splat) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 8}, getInt1Type(&ctx));
    const char splatVal = 0xff;
    const std::vector<char> inputVals = {splatVal};
    constexpr auto noop = [](void*, size_t, size_t) {};
    constexpr bool isMutable = false;

    mlir::AsmResourceBlob blob(mlir::ArrayRef<char>(inputVals), noop, isMutable);
    auto& manager = mlir::DenseResourceElementsHandle::getManagerInterface(&ctx);
    const auto baseAttr = mlir::DenseResourceElementsAttr::get(
            baseType, manager.insert("FromSplatDenseResourceElementsAttr", std::move(blob)));

    auto contentSetup = Const::ContentSetup(baseAttr);
    auto conversionType = getSInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = contentSetup.get();
    const auto content = contentAttr.fold();
    EXPECT_TRUE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<char>();

    EXPECT_EQ(content.getSplatValue<char>(), 0x01);
    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], 0x01);
    }
}

TEST_F(MLIR_SubByteTest, ConvertElemTypeI1ToI8) {
    const auto baseType = mlir::RankedTensorType::get({1, 1, 2, 8}, getInt1Type(&ctx));
    std::vector<char> inputVals{0x07, 0x62};
    constexpr auto noop = [](void*, size_t, size_t) {};
    constexpr bool isMutable = false;

    mlir::AsmResourceBlob blob(mlir::ArrayRef<char>(inputVals), noop, isMutable);
    auto& manager = mlir::DenseResourceElementsHandle::getManagerInterface(&ctx);
    const auto baseAttr = mlir::DenseResourceElementsAttr::get(
            baseType, manager.insert("FromSplatDenseResourceElementsAttr", std::move(blob)));

    auto contentSetup = Const::ContentSetup(baseAttr);
    auto conversionType = getSInt8Type(&ctx);
    contentSetup = contentSetup.convertElemType(conversionType);
    auto contentAttr = contentSetup.get();
    const auto content = contentAttr.fold();
    EXPECT_FALSE(content.isSplat());
    EXPECT_EQ(content.getType(), contentAttr.getType());
    EXPECT_EQ(content.isSplat(), contentAttr.isSplat());

    auto contentElemType = content.getType().getElementType();
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(contentElemType));

    auto integerElemType = mlir::cast<mlir::IntegerType>(contentElemType);
    EXPECT_TRUE(integerElemType.isSigned() && integerElemType.getWidth() == 8);

    const auto contentVals = content.getValues<char>();

    EXPECT_EQ(contentVals.size(), baseType.getNumElements());

    std::vector<char> expectedVals{0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                                   0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00};
    EXPECT_EQ(contentVals.size(), expectedVals.size());

    for (size_t i = 0; i < contentVals.size(); ++i) {
        EXPECT_EQ(contentVals[i], expectedVals[i]);
    }
}

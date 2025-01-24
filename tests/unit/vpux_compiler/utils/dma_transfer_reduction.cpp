//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"

using namespace vpux;

namespace {

constexpr vpux::StringRef DDR_NAME = "DDR";

}  // namespace

struct DMAReductionTestParams {
    SmallVector<int64_t> dims;
    SmallVector<int64_t> strides;
    DimsOrder::StorageType storageOrder;
    std::string elementType;
    SmallVector<uint64_t> expectedReducedDims;
    SmallVector<uint64_t> expectedReducedStrides;
};

class DMAReductionTest : public testing::TestWithParam<DMAReductionTestParams> {};

TEST_P(DMAReductionTest, GetParams) {
    const auto params = GetParam();
    const auto dims = params.dims;
    const auto strides = params.strides;
    const auto storageOrder = params.storageOrder;
    const auto elementType = params.elementType;
    const auto expectedReducedDims = params.expectedReducedDims;
    const auto expectedReducedStrides = params.expectedReducedStrides;

    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);

    const auto shape = Shape(dims);
    const auto dimsOrder = DimsOrder::fromCode(storageOrder);
    const auto orderAttr = mlir::AffineMapAttr::get(dimsOrder.toAffineMap(&ctx));
    const auto stridesAttr = getIntArrayAttr(&ctx, strides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dTypeResolution = [&](mlir::StringRef typeStr) -> mlir::Type {
        if ("f16" == typeStr) {
            return mlir::Float16Type::get(&ctx);
        } else if ("u8" == typeStr) {
            return mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
        } else if ("i8" == typeStr) {
            return mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::SignednessSemantics::Signed);
        } else if ("u4" == typeStr) {
            return mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::SignednessSemantics::Unsigned);
        } else if ("i4" == typeStr) {
            return mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::SignednessSemantics::Signed);
        }
        VPUX_THROW("Unsupported dtype {0}", typeStr);
    };

    const auto memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto memrefType = mlir::MemRefType::get(shape.raw(), dTypeResolution(elementType), layout, memSpace);

    const auto ndType = memrefType.dyn_cast<vpux::NDTypeInterface>();

    const auto reducedDimsStrides = reduceDimsForDma(ndType);
    EXPECT_EQ(reducedDimsStrides.dims, expectedReducedDims);
    EXPECT_EQ(reducedDimsStrides.strides, expectedReducedStrides);
}

std::vector<DMAReductionTestParams> dmaReductionTestValues = {
        {// Compact, diff sizes
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{24, 12, 4, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{48},
         /*expectedReducedStrides=*/{48}},
        {// Compact, non 1 outermost dim
         /*dims=*/{3, 1, 1, 1},
         /*strides=*/{1, 1, 1, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{6},
         /*expectedReducedStrides=*/{6}},
        {// Compact, scalar transfer
         /*dims=*/{1, 1, 1, 1},
         /*strides=*/{1, 1, 1, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{2},
         /*expectedReducedStrides=*/{2}},
        {// Single stride, inner dim compact
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{48, 24, 8, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{6, 8},
         /*expectedReducedStrides=*/{96, 16}},
        {// Single stride, inner dim strided, scalar outer dims
         /*dims=*/{1, 1, 1, 3},
         /*strides=*/{6, 6, 6, 2},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{3, 2},
         /*expectedReducedStrides=*/{12, 4}},
        {// Compact stride, non 1 outermost dim
         /*dims=*/{5, 2, 3, 4},
         /*strides=*/{24, 12, 4, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{240},
         /*expectedReducedStrides=*/{240}},
        {// Single stride, non 1 outermost dim
         /*dims=*/{5, 2, 3, 4},
         /*strides=*/{48, 24, 8, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{30, 8},
         /*expectedReducedStrides=*/{480, 16}},
        {// Single stride, inner dim strided
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{48, 24, 8, 2},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{24, 2},
         /*expectedReducedStrides=*/{96, 4}},
        {// Dual stride, inner dim compact
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{64, 32, 8, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{2, 3, 8},
         /*expectedReducedStrides=*/{128, 64, 16}},
        {// Dual stride, inner dim strided
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{64, 32, 8, 2},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{2, 12, 2},
         /*expectedReducedStrides=*/{128, 64, 4}},
        {// 3 stride levels, non 1 outermost dim, inner dim compact
         /*dims=*/{5, 2, 3, 4},
         /*strides=*/{96, 32, 8, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{5, 2, 3, 8},
         /*expectedReducedStrides=*/{960, 192, 64, 16}},
        {// 4 stride levels, non 1 outermost dim, inner dim compact
         /*dims=*/{2, 5, 2, 3, 4},
         /*strides=*/{500, 96, 32, 8, 1},
         /*dimsOrder=*/0x12345,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{2, 5, 2, 3, 8},
         /*expectedReducedStrides=*/{2000, 1000, 192, 64, 16}},
        {// Single stride, inner dim compact
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{48, 24, 8, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{6, 8},
         /*expectedReducedStrides=*/{96, 16}},
        {// Single stride, 3D shape
         /*dims=*/{1, 2, 3},
         /*strides=*/{24, 8, 1},
         /*dimsOrder=*/0x123,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{2, 6},
         /*expectedReducedStrides=*/{48, 16}},
        {// Compact, diff sizes, NHWC layout
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{24, 1, 8, 2},
         /*dimsOrder=*/0x1342,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{48},
         /*expectedReducedStrides=*/{48}},
        {// Single stride, inner dim compact, NHWC layout
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{48, 1, 16, 4},
         /*dimsOrder=*/0x1342,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{12, 4},
         /*expectedReducedStrides=*/{96, 8}},
        {// Dual stride, inner dim strided, NHWC layout
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{72, 2, 24, 4},
         /*dimsOrder=*/0x1342,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{3, 8, 2},
         /*expectedReducedStrides=*/{144, 48, 4}},
        {// Compact, 4 bit
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{24, 12, 4, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"i4",
         /*expectedReducedDims=*/{12},
         /*expectedReducedStrides=*/{12}},
        {// Compact, scalar transfer
         /*dims=*/{1, 1, 1, 1},
         /*strides=*/{1, 1, 1, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"i4",
         /*expectedReducedDims=*/{1},
         /*expectedReducedStrides=*/{1}},
        {// Single stride, 4 bit, inner dim compact
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{48, 24, 8, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"i4",
         /*expectedReducedDims=*/{6, 2},
         /*expectedReducedStrides=*/{24, 4}},
        {// Single stride, 4 bit, inner dim strided
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{48, 24, 8, 2},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"i4",
         /*expectedReducedDims=*/{24, 1},
         /*expectedReducedStrides=*/{24, 1}},
        {// Dual stride, 8 bit, inner dim compact
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{64, 32, 8, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"u8",
         /*expectedReducedDims=*/{2, 3, 4},
         /*expectedReducedStrides=*/{64, 32, 8}},
        {// Single stride, 4 bit, inner dim compact
         /*dims=*/{1, 5, 308, 128},
         /*strides=*/{16515072, 589824, 128, 1},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"i4",
         /*expectedReducedDims=*/{5, 19712},
         /*expectedReducedStrides=*/{8257536, 294912}},
        {// Overlap stride, inner dim compact
         /*dims=*/{2, 1, 3, 56, 224},
         /*strides=*/{12544, 150528, 50176, 224, 1},
         /*dimsOrder=*/0x12345,
         /*elementType=*/"f16",
         /*expectedReducedDims=*/{2, 3, 25088},
         /*expectedReducedStrides=*/{50176, 25088, 100352}},
};

INSTANTIATE_TEST_SUITE_P(ArbitraryTest, DMAReductionTest, testing::ValuesIn(dmaReductionTestValues));

class DMAReductionTestExpectFail : public testing::TestWithParam<DMAReductionTestParams> {};

TEST_P(DMAReductionTestExpectFail, GetParams) {
    const auto params = GetParam();
    const auto dims = params.dims;
    const auto strides = params.strides;
    const auto storageOrder = params.storageOrder;
    const auto elementType = params.elementType;
    const auto expectedReducedDims = params.expectedReducedDims;
    const auto expectedReducedStrides = params.expectedReducedStrides;

    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);

    const auto shape = Shape(dims);
    const auto dimsOrder = DimsOrder::fromCode(storageOrder);
    const auto orderAttr = mlir::AffineMapAttr::get(dimsOrder.toAffineMap(&ctx));
    const auto stridesAttr = getIntArrayAttr(&ctx, strides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dTypeResolution = [&](mlir::StringRef typeStr) -> mlir::Type {
        if ("f16" == typeStr) {
            return mlir::Float16Type::get(&ctx);
        } else if ("u8" == typeStr) {
            return mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
        } else if ("i8" == typeStr) {
            return mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::SignednessSemantics::Signed);
        } else if ("u4" == typeStr) {
            return mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::SignednessSemantics::Unsigned);
        } else if ("i4" == typeStr) {
            return mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::SignednessSemantics::Signed);
        }
        VPUX_THROW("Unsupported dtype {0}", typeStr);
    };

    const auto memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto memrefType = mlir::MemRefType::get(shape.raw(), dTypeResolution(elementType), layout, memSpace);

    const auto ndType = memrefType.dyn_cast<vpux::NDTypeInterface>();

    EXPECT_ANY_THROW(reduceDimsForDma(ndType));
}

// To add a test that throws an error #E118627
std::vector<DMAReductionTestParams> EXPECT_THROW_DmaReductionTestValues = {
        {// Breaking case - expect throw (strides should be byte aligned)
         /*dims=*/{1, 2, 3, 4},
         /*strides=*/{72, 36, 12, 3},
         /*dimsOrder=*/0x1234,
         /*elementType=*/"i4",
         /*expectedReducedDims=*/{/*12, */ 1},
         /*expectedReducedStrides=*/{/*24, */ 2}}};

INSTANTIATE_TEST_SUITE_P(ArbitraryTest, DMAReductionTestExpectFail,
                         testing::ValuesIn(EXPECT_THROW_DmaReductionTestValues));

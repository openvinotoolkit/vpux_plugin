//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/types.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>

#include <functional>

using namespace vpux;

namespace {

using TestInfo = std::tuple<mlir::Type, size_t>;
using GetTestInfo = std::function<TestInfo(mlir::MLIRContext*)>;
struct MLIR_GetExpectedBufferSizeTests : MLIR_UnitBase, ::testing::WithParamInterface<GetTestInfo> {
    mlir::MLIRContext ctx;

public:
    MLIR_GetExpectedBufferSizeTests(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
    }
};

}  // namespace

TEST_P(MLIR_GetExpectedBufferSizeTests, getExpectedBufferSize) {
    const auto [type, expectedSize] = GetParam()(&ctx);
    ASSERT_EQ(static_cast<size_t>(vpux::getExpectedBufferSize(type).count()), expectedSize);
}

INSTANTIATE_TEST_SUITE_P(
        NormalTypes, MLIR_GetExpectedBufferSizeTests,
        ::testing::Values(
                [](mlir::MLIRContext* ctx) -> TestInfo {
                    const auto type = mlir::RankedTensorType::get({1, 3, 3, 3}, mlir::Float32Type::get(ctx));
                    return {type, 3 * 3 * 3 * sizeof(float)};
                },
                [](mlir::MLIRContext* ctx) -> TestInfo {
                    const auto type = mlir::RankedTensorType::get({1, 3, 3, 3}, getUInt8Type(ctx));
                    return {type, 3 * 3 * 3 * sizeof(uint8_t)};
                }));

INSTANTIATE_TEST_SUITE_P(SubbyteTypes, MLIR_GetExpectedBufferSizeTests,
                         ::testing::Values(
                                 [](mlir::MLIRContext* ctx) -> TestInfo {
                                     const auto type = mlir::RankedTensorType::get({1, 3, 3, 3}, getSInt4Type(ctx));
                                     constexpr auto supposedSize = (3 * 3 * 3) / 2 + 1;  // sizeof(uint4_t) == 1 / 2
                                     return {type, supposedSize};
                                 },
                                 [](mlir::MLIRContext* ctx) -> TestInfo {
                                     const auto i3Type = mlir::IntegerType::get(ctx, 3, mlir::IntegerType::Signless);
                                     const auto type = mlir::RankedTensorType::get({1, 3, 3, 3}, i3Type);
                                     constexpr auto supposedSize = (3 * 3 * 3) * 3 / 8 + 1;
                                     return {type, supposedSize};
                                 },
                                 [](mlir::MLIRContext* ctx) -> TestInfo {
                                     const auto ui2Type = mlir::IntegerType::get(ctx, 2, mlir::IntegerType::Unsigned);
                                     const auto type = mlir::RankedTensorType::get({1, 3, 1, 4}, ui2Type);
                                     constexpr auto supposedSize = (3 * 4) * 2 / 8;  // precise size here
                                     return {type, supposedSize};
                                 }));

//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>

using namespace vpux;

struct MLIR_IR_BroadcastUtils : MLIR_UnitBase {
    mlir::MLIRContext ctx;

    MLIR_IR_BroadcastUtils(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }

    mlir::Type defaultStorageType() {
        return mlir::Float32Type::get(&ctx);
    }

    static SmallVector<float> generateValues(size_t n) {
        SmallVector<float> values(n, 0);
        for (size_t i = 0; i < n; ++i) {
            values[i] = static_cast<float>(i);
        }
        return values;
    }

    static size_t totalSize(ArrayRef<int64_t> shape) {
        return static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>{}));
    }

    static ArrayRef<char> toRawBuf(ArrayRef<float> x) {
        return ArrayRef<char>(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
    }
};

TEST_F(MLIR_IR_BroadcastUtils, AlignShapes_MismatchingRanks) {
    const auto type1 = mlir::RankedTensorType::get({1, 2, 3, 4}, defaultStorageType());
    const auto type2 = mlir::RankedTensorType::get({3, 4}, defaultStorageType());
    const auto data1 = generateValues(totalSize(type1.getShape()));
    const auto data2 = generateValues(totalSize(type2.getShape()));
    auto content1 = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(type1), toRawBuf(data1),
                                                  defaultStorageType(), false);
    auto content2 = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(type2), toRawBuf(data2),
                                                  defaultStorageType(), false);
    EXPECT_TRUE(mlir::failed(vpux::IE::broadcastAlignShapes(&ctx, content1, content2, Logger::global())));
}

TEST_F(MLIR_IR_BroadcastUtils, AlignShapes_MismatchingAxis) {
    const auto type1 = mlir::RankedTensorType::get({2, 3}, defaultStorageType());
    const auto type2 = mlir::RankedTensorType::get({2, 4}, defaultStorageType());
    const auto data1 = generateValues(totalSize(type1.getShape()));
    const auto data2 = generateValues(totalSize(type2.getShape()));
    auto content1 = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(type1), toRawBuf(data1),
                                                  defaultStorageType(), false);
    auto content2 = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(type2), toRawBuf(data2),
                                                  defaultStorageType(), false);
    EXPECT_TRUE(mlir::failed(vpux::IE::broadcastAlignShapes(&ctx, content1, content2, Logger::global())));
}

TEST_F(MLIR_IR_BroadcastUtils, AlignShapes_Noop) {
    const auto nonSplatType = mlir::RankedTensorType::get({1, 2, 3, 4}, defaultStorageType());
    const auto nonSplatData = generateValues(totalSize(nonSplatType.getShape()));
    auto content = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(nonSplatType), toRawBuf(nonSplatData),
                                                 defaultStorageType(), false);
    EXPECT_TRUE(mlir::succeeded(vpux::IE::broadcastAlignShapes(&ctx, content, content, Logger::global())));
    EXPECT_EQ(content.getType().getShape(), ShapeRef(nonSplatType.getShape()));
}

TEST_F(MLIR_IR_BroadcastUtils, AlignShapes_OneIsSplat) {
    const auto nonSplatType = mlir::RankedTensorType::get({1, 2, 3, 4}, defaultStorageType());
    const auto splatType = mlir::RankedTensorType::get({1}, defaultStorageType());
    const auto nonSplatData = generateValues(totalSize(nonSplatType.getShape()));
    const float splatValue = 42.f;

    auto nonSplatContent = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(nonSplatType),
                                                         toRawBuf(nonSplatData), defaultStorageType(), false);
    auto splatContent1 = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(splatType),
                                                       toRawBuf(ArrayRef(splatValue)), defaultStorageType(), true);
    auto splatContent2 = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(splatType),
                                                       toRawBuf(ArrayRef(splatValue)), defaultStorageType(), true);

    EXPECT_TRUE(
            mlir::succeeded(vpux::IE::broadcastAlignShapes(&ctx, splatContent1, nonSplatContent, Logger::global())));
    EXPECT_EQ(nonSplatContent.getType().getShape(), splatContent1.getType().getShape());

    EXPECT_TRUE(
            mlir::succeeded(vpux::IE::broadcastAlignShapes(&ctx, nonSplatContent, splatContent2, Logger::global())));
    EXPECT_EQ(nonSplatContent.getType().getShape(), splatContent2.getType().getShape());
}

TEST_F(MLIR_IR_BroadcastUtils, AlignShapes_NonSplats) {
    const auto type1 = mlir::RankedTensorType::get({1, 5, 1, 5}, defaultStorageType());
    const auto type2 = mlir::RankedTensorType::get({3, 1, 8, 1}, defaultStorageType());
    const auto data1 = generateValues(totalSize(type1.getShape()));
    const auto data2 = generateValues(totalSize(type2.getShape()));
    auto content1 = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(type1), toRawBuf(data1),
                                                  defaultStorageType(), false);
    auto content2 = Const::Content::fromRawBuffer(mlir::cast<NDTypeInterface>(type2), toRawBuf(data2),
                                                  defaultStorageType(), false);

    EXPECT_TRUE(mlir::succeeded(vpux::IE::broadcastAlignShapes(&ctx, content1, content2, Logger::global())));
    const auto finalType = mlir::RankedTensorType::get({3, 5, 8, 5}, defaultStorageType());
    EXPECT_EQ(ShapeRef(finalType.getShape()), content1.getType().getShape());
    EXPECT_EQ(ShapeRef(finalType.getShape()), content2.getType().getShape());
}

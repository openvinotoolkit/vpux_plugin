//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include "common/utils.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}  // namespace

using MLIR_NDTypeInterface = MLIR_UnitBase;

TEST_F(MLIR_NDTypeInterface, BufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto ndType =
            VPUIP::BufferType::get(&ctx, shape, elemType, layout, dimsSpace).dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 13, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 16 * 13);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::NHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({212992_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> memStrides({212992_Bit, 16384_Bit, 1024_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 13 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 13 * 16);

    const SmallVector<int64_t> newShape({1, 32, 32, 13});
    const auto changedShape = ndType.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), vpux::ShapeRef(newShape));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElementType.getElementType().isa<mlir::Float32Type>());

    const auto changedShapeAndElementType =
            ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(changedShapeAndElementType.getElementType().isa<mlir::Float32Type>());

    const auto changedDimsOrder = ndType.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);

    const auto changedMemSpace = ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    const auto denseTile = ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(denseTile.getStrides().raw(), tileStrides);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile.getStrides().raw(), strides);

    const SmallVector<int64_t> tileElemStrides2({2, 1, 1, 1});
    const SmallVector<Bit> newStrides2({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const auto viewTile2 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides2));
    EXPECT_EQ(viewTile2.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile2.getStrides().raw(), newStrides2);

    const SmallVector<int64_t> tileElemStrides3({3, 1, 2, 1});
    const SmallVector<Bit> newStrides3({638976_Bit, 16_Bit, 32768_Bit, 1024_Bit});
    const auto viewTile3 =
            ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides3));
    EXPECT_EQ(viewTile3.getShape(), ShapeRef(tileShape));
    EXPECT_EQ(viewTile3.getStrides().raw(), newStrides3);

    const auto noStridesType =
            VPUIP::BufferType::get(&ctx, shape, elemType, orderAttr, dimsSpace).dyn_cast<vpux::NDTypeInterface>();

    EXPECT_EQ(ndType.eraseTiledInfo(), noStridesType);

    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));

    // Test out sub byte logic
    const SmallVector<int64_t> newSubByteShape({1, 27, 3, 3});
    const auto changedShapeAndSubByteElementType =
            ndType.changeShapeElemType(vpux::ShapeRef(newSubByteShape), mlir::IntegerType::get(&ctx, 4));

    EXPECT_EQ(changedShapeAndSubByteElementType.getShape(), vpux::ShapeRef(newSubByteShape));
    EXPECT_TRUE(changedShapeAndSubByteElementType.getElementType().isa<mlir::IntegerType>());
    EXPECT_EQ(changedShapeAndSubByteElementType.getElemTypeSize().count(), 4);
    // Align to multiple of 1 Byte
    EXPECT_EQ(changedShapeAndSubByteElementType.getTotalAllocSize().count(), divUp(27 * 3 * 3, 2));
    EXPECT_EQ(changedShapeAndSubByteElementType.getCompactAllocSize().count(), divUp(27 * 3 * 3, 2));
    const SmallVector<Bit> newSubByteStrides({972_Bit, 4_Bit, 324_Bit, 108_Bit});
    EXPECT_EQ(changedShapeAndSubByteElementType.getStrides().raw(), newSubByteStrides);
}

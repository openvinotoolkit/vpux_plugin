//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}  // namespace
using MLIR_NDTypeInterface = MLIR_UnitBase;

TEST_F(MLIR_NDTypeInterface, ITIBufferType_Output) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    // SoH - ITIBuffer in tile 1
    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto elemStrides = SmallVector<int64_t>({64 * 16 * 13, 1, 64 * 16, 64});
    const auto stridesAttr = getIntArrayAttr(&ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto tile0 = getIntAttr(&ctx, 0);
    const auto tile1 = getIntAttr(&ctx, 1);
    const auto tile2 = getIntAttr(&ctx, 2);
    const auto haloShape = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 64, 1, 16}));

    // Inward halos
    const auto inwardHaloOffset0 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 0, 0}));
    const auto inwardHalo0 = VPUIP::HaloRegionAttr::get(&ctx, haloShape, inwardHaloOffset0, tile1);

    const auto inwardHaloOffset1 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 12, 0}));
    const auto inwardHalo1 = VPUIP::HaloRegionAttr::get(&ctx, haloShape, inwardHaloOffset1, tile1);

    const auto inwardHalos = SmallVector<VPUIP::HaloRegionAttr>({inwardHalo0, inwardHalo1});

    // Outward halo for tile 0
    const auto outwardHaloOffset0 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 1, 0}));
    const auto inwardHaloOffsetInTile0 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 12, 0}));
    const auto inwardHaloInTile0 = VPUIP::HaloRegionAttr::get(&ctx, haloShape, inwardHaloOffsetInTile0, tile0);

    const auto inwardHalosInTile0Attr = mlir::ArrayAttr::get(&ctx, SmallVector<mlir::Attribute>({inwardHaloInTile0}));
    const auto outwardHalo0 =
            VPUIP::OutwardHaloRegionAttr::get(&ctx, haloShape, outwardHaloOffset0, tile1, inwardHalosInTile0Attr);

    // Outward halo for tile 2
    const auto outwardHaloOffset1 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 11, 0}));
    const auto inwardHaloOffsetInTile2 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 0, 0}));
    const auto inwardHaloInTile2 = VPUIP::HaloRegionAttr::get(&ctx, haloShape, inwardHaloOffsetInTile2, tile2);

    const auto inwardHalosInTile2Attr = mlir::ArrayAttr::get(&ctx, SmallVector<mlir::Attribute>({inwardHaloInTile2}));
    const auto outwardHalo1 =
            VPUIP::OutwardHaloRegionAttr::get(&ctx, haloShape, outwardHaloOffset1, tile1, inwardHalosInTile2Attr);

    const auto outwardHalos = SmallVector<VPUIP::OutwardHaloRegionAttr>({outwardHalo0, outwardHalo1});

    const auto ndType =
            VPUIP::ITIBufferType::get(&ctx, shape, elemType, layout, dimsSpace, nullptr, inwardHalos, outwardHalos)
                    .dyn_cast<vpux::NDTypeInterface>();
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
    EXPECT_ANY_THROW(ndType.changeShape(vpux::ShapeRef(newShape)));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElementType.getElementType().isa<mlir::Float32Type>());

    EXPECT_ANY_THROW(ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx)));

    EXPECT_ANY_THROW(ndType.changeDimsOrder(DimsOrder::NCHW));

    const auto changedMemSpace = ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    EXPECT_ANY_THROW(ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape))));

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    EXPECT_ANY_THROW(ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape)));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));

    // Test out sub byte logic
    const auto subByteElemType = ndType.changeElemType(mlir::IntegerType::get(&ctx, 4));
    EXPECT_TRUE(subByteElemType.getElementType().isa<mlir::IntegerType>());
    EXPECT_EQ(subByteElemType.getElemTypeSize().count(), 4);
    EXPECT_EQ(subByteElemType.getTotalAllocSize().count(), 64 * 13 * 16 / 2);
    EXPECT_EQ(subByteElemType.getCompactAllocSize().count(), 64 * 13 * 16 / 2);
    const SmallVector<Bit> newSubByteStrides({53248_Bit, 4_Bit, 4096_Bit, 256_Bit});
    EXPECT_EQ(subByteElemType.getStrides().raw(), newSubByteStrides);
}

TEST_F(MLIR_NDTypeInterface, ITIBufferType_Input) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    // SoH - ITIBuffer in tile 0
    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);

    const auto orderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto layout = vpux::MemRefAttr::get(orderAttr, nullptr,
                                              /*allocSize=*/nullptr, &ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto tile0 = getIntAttr(&ctx, 0);
    const auto tile1 = getIntAttr(&ctx, 1);
    const auto haloShape = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 64, 1, 16}));

    // Inward halos for tile 0
    const auto inwardHaloOffset0 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 12, 0}));
    const auto inwardHalo0 = VPUIP::HaloRegionAttr::get(&ctx, haloShape, inwardHaloOffset0, tile0);

    const auto inwardHalos0 = SmallVector<VPUIP::HaloRegionAttr>({inwardHalo0});

    // Outward halo in tile 0
    const auto outwardHaloOffset0 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 11, 0}));
    const auto inwardHaloOffsetInTile1 = getIntArrayAttr(&ctx, SmallVector<int64_t>({0, 0, 0, 0}));
    const auto inwardHaloInTile1 = VPUIP::HaloRegionAttr::get(&ctx, haloShape, inwardHaloOffsetInTile1, tile1);

    const auto inwardHalosInTile1ArrayAttr =
            mlir::ArrayAttr::get(&ctx, SmallVector<mlir::Attribute>({inwardHaloInTile1}));
    const auto outwardHalo0 =
            VPUIP::OutwardHaloRegionAttr::get(&ctx, haloShape, outwardHaloOffset0, tile0, inwardHalosInTile1ArrayAttr);

    const auto outwardHalos = SmallVector<VPUIP::OutwardHaloRegionAttr>({outwardHalo0});

    const auto isIduSegmentation = mlir::UnitAttr::get(&ctx);
    const auto ndType = VPUIP::ITIBufferType::get(&ctx, shape, elemType, layout, dimsSpace, isIduSegmentation,
                                                  inwardHalos0, outwardHalos)
                                .dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Buffer is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndType.getShape(), vpux::ShapeRef({1, 64, 13, 16}));
    EXPECT_EQ(ndType.getMemShape(), vpux::MemShape({1, 12, 16, 64}));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 16 * 12);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), vpux::DimsOrder::NHWC);

    EXPECT_EQ(ndType.getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(ndType.getMemoryKind(), vpux::VPU::MemoryKind::CMX_NN);

    const SmallVector<vpux::Bit> strides({196608_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> memStrides({196608_Bit, 16384_Bit, 1024_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), memStrides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 2 * 64 * 12 * 16);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 2 * 64 * 12 * 16);

    const SmallVector<int64_t> newShape({1, 32, 32, 13});
    EXPECT_ANY_THROW(ndType.changeShape(vpux::ShapeRef(newShape)));

    const auto changedElementType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElementType.getElementType().isa<mlir::Float32Type>());

    EXPECT_ANY_THROW(ndType.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx)));

    EXPECT_ANY_THROW(ndType.changeDimsOrder(DimsOrder::NCHW));

    const auto changedMemSpace = ndType.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), DDR_NAME);

    EXPECT_ANY_THROW(ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape))));

    const SmallVector<Bit> newStrides({425984_Bit, 16_Bit, 16384_Bit, 1024_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(newStrides)));

    const SmallVector<int64_t> tileOffset({0, 0, 32, 0});
    const SmallVector<int64_t> tileShape({1, 32, 20, 8});
    const SmallVector<Bit> tileStrides({81920_Bit, 16_Bit, 4096_Bit, 512_Bit});
    EXPECT_ANY_THROW(ndType.extractDenseTile(ShapeRef(tileOffset), ShapeRef(tileShape)));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    EXPECT_ANY_THROW(ndType.extractViewTile(ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides)));

    const SmallVector<int64_t> pads({0, 0, 2, 2});
    EXPECT_ANY_THROW(ndType.pad(vpux::ShapeRef(pads), vpux::ShapeRef(pads)));

    // Explicit strides

    const auto explicitElemStrides = SmallVector<int64_t>({64 * 20 * 13, 1, 64 * 20, 64});
    const auto explicitStridesAttr = getIntArrayAttr(&ctx, explicitElemStrides);
    const auto explicitStrideslayout = vpux::MemRefAttr::get(orderAttr, explicitStridesAttr,
                                                             /*allocSize=*/nullptr, &ctx);

    const auto explicitStridesNdType =
            VPUIP::ITIBufferType::get(&ctx, shape, elemType, explicitStrideslayout, dimsSpace, isIduSegmentation,
                                      inwardHalos0, outwardHalos)
                    .dyn_cast<vpux::NDTypeInterface>();

    const SmallVector<vpux::Bit> explicitStrides({266240_Bit, 16_Bit, 20480_Bit, 1024_Bit});
    const SmallVector<vpux::Bit> explicitMemStrides({266240_Bit, 20480_Bit, 1024_Bit, 16_Bit});
    EXPECT_EQ(explicitStridesNdType.getStrides().raw(), explicitStrides);
    EXPECT_EQ(explicitStridesNdType.getMemStrides().raw(), explicitMemStrides);

    EXPECT_EQ(explicitStridesNdType.getTotalAllocSize().count(), 2 * 64 * 13 * 20);
    EXPECT_EQ(explicitStridesNdType.getCompactAllocSize().count(), 2 * 64 * 12 * 16);
}

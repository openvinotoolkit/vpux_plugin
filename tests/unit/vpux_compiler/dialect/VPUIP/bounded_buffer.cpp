//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
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

TEST_F(MLIR_NDTypeInterface, BoundedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    auto dataShape = Shape({1, 64, 13, 16});
    auto dataElementType = mlir::Float16Type::get(&ctx);
    auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    auto dataBuffer = mlir::MemRefType::get(dataShape.raw(),  // shape
                                            dataElementType,  // elementType
                                            nullptr,          // orderAttr
                                            dimsSpace         // memorySpace
    );

    auto dynamicShape = Shape({4});
    auto dynamicShapeType = mlir::IntegerType::get(&ctx, 32);
    auto dynamicShapeBuffer = mlir::MemRefType::get(dynamicShape.raw(),  // shape
                                                    dynamicShapeType     // elementType
    );
    auto boundedBuffer = VPUIP::BoundedBufferType::get(dataBuffer, dynamicShapeBuffer);

    const auto ndTypeBoundedBuffer = boundedBuffer.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndTypeBoundedBuffer != nullptr) << "BoundedBuffer is not of vpux::NDTypeInterface type";

    const auto ndTypeData = boundedBuffer.getData().dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndTypeData != nullptr) << "BoundedBuffer.getData() is not of vpux::NDTypeInterface type";
    const auto ndTypeDynamicShape = boundedBuffer.getDynamicShape().dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndTypeDynamicShape != nullptr)
            << "BoundedBuffer.getDynamicShape() is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndTypeBoundedBuffer.getShape(), ndTypeData.getShape());
    EXPECT_EQ(ndTypeBoundedBuffer.getMemShape(), ndTypeData.getMemShape());

    EXPECT_EQ(ndTypeBoundedBuffer.hasRank(), ndTypeData.hasRank());
    EXPECT_EQ(ndTypeBoundedBuffer.getRank(), ndTypeData.getRank());
    EXPECT_EQ(ndTypeBoundedBuffer.getNumElements(), ndTypeData.getNumElements());
    EXPECT_EQ(ndTypeBoundedBuffer.getElementType(), ndTypeData.getElementType());
    EXPECT_EQ(ndTypeBoundedBuffer.getDimsOrder(), ndTypeData.getDimsOrder());
    EXPECT_EQ(ndTypeBoundedBuffer.getMemSpace(), ndTypeData.getMemSpace());
    EXPECT_EQ(ndTypeBoundedBuffer.getMemoryKind(), ndTypeData.getMemoryKind());

    const SmallVector<vpux::Bit> strides({212992_Bit, 3328_Bit, 256_Bit, 16_Bit});
    EXPECT_EQ(ndTypeBoundedBuffer.getStrides().raw(), strides);
    EXPECT_EQ(ndTypeBoundedBuffer.getMemStrides().raw(), strides);

    EXPECT_EQ(ndTypeBoundedBuffer.getTotalAllocSize(),
              ndTypeData.getTotalAllocSize() + ndTypeDynamicShape.getTotalAllocSize());
    EXPECT_EQ(ndTypeBoundedBuffer.getCompactAllocSize(),
              ndTypeData.getCompactAllocSize() + ndTypeDynamicShape.getCompactAllocSize());

    const SmallVector<int64_t> newShape({1, 32, 13});
    const auto changedShape = ndTypeBoundedBuffer.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.cast<VPUIP::BoundedBufferType>().getData().cast<NDTypeInterface>().getShape(),
              vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.cast<VPUIP::BoundedBufferType>().getDynamicShape().cast<NDTypeInterface>().getShape(),
              vpux::ShapeRef({3}));

    const auto changedElementType = ndTypeBoundedBuffer.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElementType.getElementType().isa<mlir::Float32Type>());
    EXPECT_TRUE(changedElementType.cast<VPUIP::BoundedBufferType>()
                        .getDynamicShape()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());

    const auto changedShapeAndElementType =
            ndTypeBoundedBuffer.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(changedShapeAndElementType.getElementType().isa<mlir::Float32Type>());
    EXPECT_EQ(changedShapeAndElementType.cast<VPUIP::BoundedBufferType>().getData().cast<NDTypeInterface>().getShape(),
              vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShapeAndElementType.cast<VPUIP::BoundedBufferType>()
                      .getDynamicShape()
                      .cast<NDTypeInterface>()
                      .getShape(),
              vpux::ShapeRef({3}));

    const auto changedDimsOrder = ndTypeBoundedBuffer.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);

    const auto changedMemoryKind = ndTypeBoundedBuffer.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME));
    EXPECT_EQ(changedMemoryKind.getMemoryKind(), vpux::VPU::MemoryKind::DDR);
    EXPECT_EQ(changedMemoryKind.cast<VPUIP::BoundedBufferType>().getData().cast<NDTypeInterface>().getMemoryKind(),
              vpux::VPU::MemoryKind::DDR);
    EXPECT_EQ(changedMemoryKind.cast<VPUIP::BoundedBufferType>()
                      .getDynamicShape()
                      .cast<NDTypeInterface>()
                      .getMemoryKind(),
              vpux::VPU::MemoryKind::DDR);

    const SmallVector<Bit> newStrides({851968_Bit, 13312_Bit, 1024_Bit, 16_Bit});
    const auto changedStrides = ndTypeBoundedBuffer.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>
#include <numeric>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}  // namespace

using MLIR_NDTypeInterface = MLIR_UnitBase;

TEST_F(MLIR_NDTypeInterface, SparseBufferType_Weights) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{64, 16, 3, 3};
    const Shape sparsityMapShape{64, 1, 1, 256};
    const DimsOrder order = DimsOrder::NCHW;
    const DimsOrder sparsityMapOrder = DimsOrder::NCHW;
    const mlir::AffineMapAttr layout = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const mlir::AffineMapAttr sparsityMapLayout = mlir::AffineMapAttr::get(sparsityMapOrder.toAffineMap(&ctx));
    const IndexedSymbolAttr memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto data = mlir::MemRefType::get(shape.raw(), mlir::Float16Type::get(&ctx), layout, memSpace);
    const auto sparsityMap =
            mlir::MemRefType::get(sparsityMapShape.raw(), mlir::IntegerType::get(&ctx, 1), sparsityMapLayout, memSpace);
    const auto isWeights = mlir::UnitAttr::get(&ctx);
    SmallVector<int64_t> numElems(64);
    std::iota(numElems.begin(), numElems.end(), 0);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));
    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    const auto sparsityCompression = VPUIP::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                         numElemsAttr, getIntAttr(&ctx, alignment));
    const auto sparseBufferType =
            VPUIP::SparseBufferType::get(data, sparsityMap, nullptr, isWeights, sparsityCompression);

    const auto ndType = sparseBufferType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0)));

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), order);

    EXPECT_EQ(ndType.getMemSpace(), memSpace);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({2304_Bit, 144_Bit, 48_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);

    int64_t dataByteSize = 0;
    for (auto elems : numElems) {
        dataByteSize += alignValUp<int64_t>(elems * sizeof(vpux::type::float16), alignment);
    }
    EXPECT_EQ(ndType.getTotalAllocSize().count(), dataByteSize + 64 * 256 / CHAR_BIT);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), dataByteSize + 64 * 256 / CHAR_BIT);

    const SmallVector<int64_t> newShape({32, 16, 3, 3});
    const SmallVector<int64_t> newSMShape({32, 1, 1, 256});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), ShapeRef(newShape));

    const auto sparseChangedShape = changedShape.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(sparseChangedShape.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));
    EXPECT_EQ(sparseChangedShape.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newSMShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());
    const auto sparseChangedElemType = changedElemType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(sparseChangedElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float32Type>());
    EXPECT_TRUE(
            sparseChangedElemType.getSparsityMap().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());

    const SmallVector<int64_t> newShape2({32, 16, 5, 5});
    const SmallVector<int64_t> newSMShape2({32, 1, 1, 512});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::Float64Type>());
    const auto sparseChangedShapeElemType = changedShapeElemType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newSMShape2));
    EXPECT_TRUE(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float64Type>());
    EXPECT_TRUE(sparseChangedShapeElemType.getSparsityMap()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());

    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = changedDimsOrder.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(sparseChangedDimsOrder.getData().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getSparsityMap().cast<NDTypeInterface>().getDimsOrder(), sparsityMapOrder);

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = changedMemSpace.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = changedMemSpace.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);

    const SmallVector<Bit> newStrides({4608_Bit, 144_Bit, 48_Bit, 16_Bit});
    const SmallVector<Bit> sparsityMapStrides({256_Bit, 256_Bit, 256_Bit, 1_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides(), StridesRef(newStrides));
    const auto sparseChangedStrides = changedStrides.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedStrides != nullptr);
    EXPECT_EQ(sparseChangedStrides.getData().cast<NDTypeInterface>().getStrides(), StridesRef(newStrides));
    EXPECT_EQ(sparseChangedStrides.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(sparsityMapStrides));

    const SmallVector<int64_t> tileOffsets({0, 8, 0, 0});
    const SmallVector<int64_t> tileShape({32, 8, 3, 3});
    const SmallVector<int64_t> smTileShape({32, 1, 1, 128});
    const SmallVector<Bit> tileStrides({1152_Bit, 144_Bit, 48_Bit, 16_Bit});
    const SmallVector<Bit> smTileStrides({128_Bit, 128_Bit, 128_Bit, 1_Bit});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = tiledType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getStrides(), StridesRef(tileStrides));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(smTileShape));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getStrides(), StridesRef(smTileStrides));
    auto tiledNumElems = sparseTiledType.getSparsityCompression().getNumElems().getValues<int64_t>();
    EXPECT_EQ(tiledNumElems.size(), tileShape[compressionAxis]);

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    const auto sparseViewTile = viewTile.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseViewTile != nullptr);
    EXPECT_EQ(sparseViewTile.getData().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseViewTile.getData().cast<NDTypeInterface>().getStrides(), StridesRef(strides));
    EXPECT_EQ(sparseViewTile.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(smTileShape));
    EXPECT_EQ(sparseViewTile.getSparsityMap().cast<NDTypeInterface>().getStrides(), StridesRef(smTileStrides));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 0, 1, 1});
    const SmallVector<int64_t> padAfter({0, 0, 1, 1});
    const SmallVector<int64_t> paddedShape({64, 16, 5, 5});
    const SmallVector<int64_t> paddedSMShape({64, 1, 1, 512});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = paddedType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(sparsePaddedType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(sparsePaddedType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(paddedSMShape));
}

TEST_F(MLIR_NDTypeInterface, SparseBufferType_Activation) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, 64, 32, 32};
    const Shape seTableShape{1, 1, 32, 32};
    const DimsOrder order = DimsOrder::NCHW;
    const mlir::AffineMapAttr layout = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const IndexedSymbolAttr memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto data = mlir::MemRefType::get(shape.raw(), mlir::Float16Type::get(&ctx), layout, memSpace);
    const auto sparsityMap = mlir::MemRefType::get(shape.raw(), mlir::IntegerType::get(&ctx, 1), layout, memSpace);
    const auto storageElementTable =
            mlir::MemRefType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32), layout, memSpace);
    const auto sparseBufferType = VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);

    const auto ndType = sparseBufferType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(shape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(shape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 32 * 32);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), order);

    EXPECT_EQ(ndType.getMemSpace(), memSpace);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({1048576_Bit, 16384_Bit, 512_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 143360);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 143360);

    const SmallVector<int64_t> newShape({1, 64, 32, 16});
    const SmallVector<int64_t> newSETableShape({1, 1, 32, 16});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), ShapeRef(newShape));

    const auto sparseChangedShape = changedShape.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(sparseChangedShape.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));
    EXPECT_EQ(sparseChangedShape.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));
    EXPECT_EQ(sparseChangedShape.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(newSETableShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());
    const auto sparseChangedElemType = changedElemType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(sparseChangedElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float32Type>());
    EXPECT_TRUE(
            sparseChangedElemType.getSparsityMap().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());
    EXPECT_TRUE(sparseChangedElemType.getStorageElementTable()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());

    const SmallVector<int64_t> newShape2({1, 32, 32, 32});
    const SmallVector<int64_t> newSETableShape2({1, 1, 32, 32});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::Float64Type>());
    const auto sparseChangedShapeElemType = changedShapeElemType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(newSETableShape2));
    EXPECT_TRUE(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float64Type>());
    EXPECT_TRUE(sparseChangedShapeElemType.getSparsityMap()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());
    EXPECT_TRUE(sparseChangedShapeElemType.getStorageElementTable()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());

    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = changedDimsOrder.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(sparseChangedDimsOrder.getData().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getSparsityMap().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getStorageElementTable().cast<NDTypeInterface>().getDimsOrder(), DimsOrder::NCHW);

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = changedMemSpace.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getStorageElementTable().cast<NDTypeInterface>().getMemSpace().getLeafName(),
              CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = changedMemSpace.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getStorageElementTable().cast<NDTypeInterface>().getMemSpace().getLeafName(),
              CMX_NAME);

    const SmallVector<Bit> newStrides({2097152_Bit, 16384_Bit, 512_Bit, 16_Bit});
    const SmallVector<Bit> sparsityMapStrides({65536_Bit, 1024_Bit, 32_Bit, 1_Bit});
    const SmallVector<Bit> seTableStrides({32768_Bit, 32768_Bit, 1024_Bit, 32_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides(), StridesRef(newStrides));
    const auto sparseChangedStrides = changedStrides.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedStrides != nullptr);
    EXPECT_EQ(sparseChangedStrides.getData().cast<NDTypeInterface>().getStrides(), StridesRef(newStrides));
    EXPECT_EQ(sparseChangedStrides.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(sparsityMapStrides));
    EXPECT_EQ(sparseChangedStrides.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(seTableStrides));

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> tileShape({1, 64, 16, 32});
    const SmallVector<int64_t> tileSETableShape({1, 1, 16, 32});
    const SmallVector<Bit> tileStrides({524288_Bit, 8192_Bit, 512_Bit, 16_Bit});
    const SmallVector<Bit> tileSparsityMapStrides({32768_Bit, 512_Bit, 32_Bit, 1_Bit});
    const SmallVector<Bit> tileSETableStrides({16384_Bit, 16384_Bit, 1024_Bit, 32_Bit});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = tiledType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getStrides(), StridesRef(tileStrides));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSparsityMapStrides));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(tileSETableShape));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSETableStrides));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const SmallVector<Bit> viewTileSparsityMapStrides({65536_Bit, 1024_Bit, 32_Bit, 1_Bit});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    const auto sparseViewTile = viewTile.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseViewTile != nullptr);
    EXPECT_EQ(sparseViewTile.getData().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseViewTile.getData().cast<NDTypeInterface>().getStrides(), StridesRef(strides));
    EXPECT_EQ(sparseViewTile.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseViewTile.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(viewTileSparsityMapStrides));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(tileSETableShape));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSETableStrides));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 2, 1, 1});
    const SmallVector<int64_t> padAfter({0, 2, 1, 1});
    const SmallVector<int64_t> paddedShape({1, 68, 34, 34});
    const SmallVector<int64_t> paddedSETableShape({1, 1, 34, 34});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = paddedType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(sparsePaddedType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(sparsePaddedType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(sparsePaddedType.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(paddedSETableShape));
}

TEST_F(MLIR_NDTypeInterface, SparseBufferType__SETable_Interp_NEAREST) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, 64, 32, 32};
    const Shape interpolatedShape{1, 64, 64, 64};
    const Shape seTableShape{1, 1, 64, 64};

    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);
    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);
    const SmallVector<int64_t> sizes({1, 64, 64, 64});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    const DimsOrder order = DimsOrder::NCHW;
    const mlir::AffineMapAttr layout = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const IndexedSymbolAttr memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto data = mlir::MemRefType::get(shape.raw(), mlir::Float16Type::get(&ctx), layout, memSpace);
    const auto sparsityMap =
            mlir::MemRefType::get(interpolatedShape.raw(), mlir::IntegerType::get(&ctx, 1), layout, memSpace);
    const auto storageElementTable =
            mlir::MemRefType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32), layout, memSpace);

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::NEAREST);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto nearestModeAttr = IE::InterpolateNearestModeAttr::get(&ctx, IE::InterpolateNearestMode::FLOOR);
    const auto SEInterpolateAttr = VPU::SEInterpolateAttr::get(
            &ctx, modeAttr, coordTransformModeAttr, scaleAttr, nearestModeAttr, offsetsAttr, sizesAttr,
            /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);

    const auto sparseBufferType =
            VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, nullptr, nullptr, SEInterpolateAttr);

    const auto ndType = sparseBufferType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(interpolatedShape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(interpolatedShape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 64 * 64);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), order);

    EXPECT_EQ(ndType.getMemSpace(), memSpace);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({4194304_Bit, 65536_Bit, 1024_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 180224);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 180224);

    const SmallVector<int64_t> newShape({1, 64, 32, 16});
    const SmallVector<int64_t> newInShape({1, 64, 16, 8});
    const SmallVector<int64_t> newSETableShape({1, 1, 32, 16});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), ShapeRef(newShape));

    const auto sparseChangedShape = changedShape.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(sparseChangedShape.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newInShape));
    EXPECT_EQ(sparseChangedShape.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));
    EXPECT_EQ(sparseChangedShape.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(newSETableShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());
    const auto sparseChangedElemType = changedElemType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(sparseChangedElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float32Type>());
    EXPECT_TRUE(
            sparseChangedElemType.getSparsityMap().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());
    EXPECT_TRUE(sparseChangedElemType.getStorageElementTable()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());

    const SmallVector<int64_t> newShape2({1, 32, 32, 32});
    const SmallVector<int64_t> newInShape2({1, 32, 16, 16});
    const SmallVector<int64_t> newSETableShape2({1, 1, 32, 32});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::Float64Type>());
    const auto sparseChangedShapeElemType = changedShapeElemType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newInShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(newSETableShape2));
    EXPECT_TRUE(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float64Type>());
    EXPECT_TRUE(sparseChangedShapeElemType.getSparsityMap()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());
    EXPECT_TRUE(sparseChangedShapeElemType.getStorageElementTable()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());

    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = changedDimsOrder.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(sparseChangedDimsOrder.getData().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getSparsityMap().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getStorageElementTable().cast<NDTypeInterface>().getDimsOrder(), DimsOrder::NCHW);

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = changedMemSpace.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getStorageElementTable().cast<NDTypeInterface>().getMemSpace().getLeafName(),
              CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = changedMemSpace.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getStorageElementTable().cast<NDTypeInterface>().getMemSpace().getLeafName(),
              CMX_NAME);

    // Trivial case, compact strides
    const SmallVector<Bit> newStrides({4194304_Bit, 65536_Bit, 1024_Bit, 16_Bit});
    const SmallVector<Bit> newInStrides({1048576_Bit, 16384_Bit, 512_Bit, 16_Bit});
    const SmallVector<Bit> sparsityMapStrides({262144_Bit, 4096_Bit, 64_Bit, 1_Bit});
    const SmallVector<Bit> seTableStrides({131072_Bit, 131072_Bit, 2048_Bit, 32_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides(), StridesRef(newStrides));
    const auto sparseChangedStrides = changedStrides.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedStrides != nullptr);
    EXPECT_EQ(sparseChangedStrides.getData().cast<NDTypeInterface>().getStrides(), StridesRef(newInStrides));
    EXPECT_EQ(sparseChangedStrides.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(sparsityMapStrides));
    EXPECT_EQ(sparseChangedStrides.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(seTableStrides));

    // Extract dense tile
    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> inTileShape({1, 64, 8, 16});
    const SmallVector<int64_t> tileShape({1, 64, 16, 32});
    const SmallVector<int64_t> tileSETableShape({1, 1, 16, 32});
    const SmallVector<Bit> tileStrides({524288_Bit, 8192_Bit, 512_Bit, 16_Bit});
    const SmallVector<Bit> inTileStrides({131072_Bit, 2048_Bit, 256_Bit, 16_Bit});
    const SmallVector<Bit> tileSparsityMapStrides({32768_Bit, 512_Bit, 32_Bit, 1_Bit});
    const SmallVector<Bit> tileSETableStrides({16384_Bit, 16384_Bit, 1024_Bit, 32_Bit});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = tiledType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(inTileShape));
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getStrides(), StridesRef(inTileStrides));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSparsityMapStrides));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(tileSETableShape));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSETableStrides));

    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const SmallVector<Bit> viewTileSparsityMapStrides({262144_Bit, 4096_Bit, 64_Bit, 1_Bit});
    const SmallVector<Bit> inViewTileStrides({1048576_Bit, 16384_Bit, 512_Bit, 16_Bit});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    const auto sparseViewTile = viewTile.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseViewTile != nullptr);
    EXPECT_EQ(sparseViewTile.getData().cast<NDTypeInterface>().getShape(), ShapeRef(inTileShape));
    EXPECT_EQ(sparseViewTile.getData().cast<NDTypeInterface>().getStrides(), StridesRef(inViewTileStrides));
    EXPECT_EQ(sparseViewTile.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseViewTile.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(viewTileSparsityMapStrides));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(tileSETableShape));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSETableStrides));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 0, 1, 1});
    const SmallVector<int64_t> padAfter({0, 0, 1, 1});
    const SmallVector<int64_t> paddedShape({1, 64, 66, 66});
    const SmallVector<int64_t> paddedInShape({1, 64, 33, 33});
    const SmallVector<int64_t> paddedSETableShape({1, 1, 66, 66});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = paddedType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(sparsePaddedType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(paddedInShape));
    EXPECT_EQ(sparsePaddedType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(sparsePaddedType.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(paddedSETableShape));
}

TEST_F(MLIR_NDTypeInterface, SparseBufferType__SETable_Interp_BILINEAR) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, 64, 32, 32};
    const Shape interpolatedShape{1, 64, 65, 65};
    const Shape seTableShape{1, 1, 65, 65};

    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);
    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);
    const SmallVector<int64_t> sizes({1, 64, 65, 65});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    const DimsOrder order = DimsOrder::NCHW;
    const mlir::AffineMapAttr layout = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const IndexedSymbolAttr memSpace = IndexedSymbolAttr::get(&ctx, DDR_NAME);
    const auto data = mlir::MemRefType::get(shape.raw(), mlir::Float16Type::get(&ctx), layout, memSpace);
    const auto sparsityMap =
            mlir::MemRefType::get(interpolatedShape.raw(), mlir::IntegerType::get(&ctx, 1), layout, memSpace);
    const auto storageElementTable =
            mlir::MemRefType::get(seTableShape.raw(), mlir::IntegerType::get(&ctx, 32), layout, memSpace);

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::BILINEAR);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto SEInterpolateAttr =
            VPU::SEInterpolateAttr::get(&ctx, modeAttr, coordTransformModeAttr, scaleAttr,
                                        /*nearestModeAttr=*/nullptr, offsetsAttr, sizesAttr,
                                        /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);

    const auto sparseBufferType =
            VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, nullptr, nullptr, SEInterpolateAttr);

    const auto ndType = sparseBufferType.dyn_cast<vpux::NDTypeInterface>();
    ASSERT_TRUE(ndType != nullptr) << "Type cannot be cast to vpux::NDTypeInterface";

    EXPECT_EQ(ndType.getShape(), ShapeRef(interpolatedShape));
    EXPECT_EQ(ndType.getMemShape(), MemShape(interpolatedShape.raw()));

    EXPECT_TRUE(ndType.hasRank());
    EXPECT_EQ(ndType.getRank(), 4);
    EXPECT_EQ(ndType.getNumElements(), 64 * 65 * 65);

    EXPECT_TRUE(ndType.getElementType().isa<mlir::Float16Type>());

    EXPECT_EQ(ndType.getDimsOrder(), order);

    EXPECT_EQ(ndType.getMemSpace(), memSpace);
    EXPECT_EQ(ndType.getMemoryKind(), VPU::MemoryKind::DDR);

    const SmallVector<Bit> strides({4326400_Bit, 67600_Bit, 1040_Bit, 16_Bit});
    EXPECT_EQ(ndType.getStrides().raw(), strides);
    EXPECT_EQ(ndType.getMemStrides().raw(), strides);

    EXPECT_EQ(ndType.getElemTypeSize().count(), 16);
    EXPECT_EQ(ndType.getTotalAllocSize().count(), 181772);
    EXPECT_EQ(ndType.getCompactAllocSize().count(), 181772);

    const SmallVector<int64_t> newShape({1, 64, 33, 17});
    const SmallVector<int64_t> newInShape({1, 64, 16, 8});
    const SmallVector<int64_t> newSETableShape({1, 1, 33, 17});
    const auto changedShape = ndType.changeShape(ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), ShapeRef(newShape));
    const auto changedShape2 = ndType.changeTypeComponents(TypeComponents().setShape(ShapeRef(newShape)));
    EXPECT_EQ(changedShape2.getShape(), ShapeRef(newShape));

    const auto sparseChangedShape = changedShape.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedShape != nullptr);
    EXPECT_EQ(sparseChangedShape.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newInShape));
    EXPECT_EQ(sparseChangedShape.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape));
    EXPECT_EQ(sparseChangedShape.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(newSETableShape));

    const auto changedElemType = ndType.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(changedElemType.getElementType().isa<mlir::Float32Type>());
    const auto sparseChangedElemType = changedElemType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedElemType != nullptr);
    EXPECT_TRUE(sparseChangedElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float32Type>());
    EXPECT_TRUE(
            sparseChangedElemType.getSparsityMap().cast<NDTypeInterface>().getElementType().isa<mlir::IntegerType>());
    EXPECT_TRUE(sparseChangedElemType.getStorageElementTable()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());

    const SmallVector<int64_t> newInShape2({1, 32, 16, 16});
    const SmallVector<int64_t> newShape2({1, 32, 33, 33});
    const SmallVector<int64_t> newSETableShape2({1, 1, 33, 33});
    const auto changedShapeElemType = ndType.changeShapeElemType(ShapeRef(newShape2), mlir::Float64Type::get(&ctx));
    EXPECT_EQ(changedShapeElemType.getShape(), ShapeRef(newShape2));
    EXPECT_TRUE(changedShapeElemType.getElementType().isa<mlir::Float64Type>());
    const auto sparseChangedShapeElemType = changedShapeElemType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedShapeElemType != nullptr);
    EXPECT_EQ(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(newInShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(newShape2));
    EXPECT_EQ(sparseChangedShapeElemType.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(newSETableShape2));
    EXPECT_TRUE(sparseChangedShapeElemType.getData().cast<NDTypeInterface>().getElementType().isa<mlir::Float64Type>());
    EXPECT_TRUE(sparseChangedShapeElemType.getSparsityMap()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());
    EXPECT_TRUE(sparseChangedShapeElemType.getStorageElementTable()
                        .cast<NDTypeInterface>()
                        .getElementType()
                        .isa<mlir::IntegerType>());

    const auto dimsOrder = DimsOrder::NHWC;
    const auto changedDimsOrder = ndType.changeDimsOrder(dimsOrder);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), dimsOrder);
    const auto sparseChangedDimsOrder = changedDimsOrder.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedDimsOrder != nullptr);
    EXPECT_EQ(sparseChangedDimsOrder.getData().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getSparsityMap().cast<NDTypeInterface>().getDimsOrder(), dimsOrder);
    EXPECT_EQ(sparseChangedDimsOrder.getStorageElementTable().cast<NDTypeInterface>().getDimsOrder(), DimsOrder::NCHW);

    auto changedMemSpace = ndType.changeMemSpace(IndexedSymbolAttr::get(&ctx, CMX_NAME));
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    auto sparseChangedMemSpace = changedMemSpace.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getStorageElementTable().cast<NDTypeInterface>().getMemSpace().getLeafName(),
              CMX_NAME);

    changedMemSpace = ndType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    EXPECT_EQ(changedMemSpace.getMemSpace().getLeafName(), CMX_NAME);
    sparseChangedMemSpace = changedMemSpace.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedMemSpace != nullptr);
    EXPECT_EQ(sparseChangedMemSpace.getData().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getSparsityMap().cast<NDTypeInterface>().getMemSpace().getLeafName(), CMX_NAME);
    EXPECT_EQ(sparseChangedMemSpace.getStorageElementTable().cast<NDTypeInterface>().getMemSpace().getLeafName(),
              CMX_NAME);

    // Only trivial case
    const SmallVector<Bit> newStrides({4326400_Bit, 67600_Bit, 1040_Bit, 16_Bit});
    const SmallVector<Bit> newInStrides({1048576_Bit, 16384_Bit, 512_Bit, 16_Bit});
    const SmallVector<Bit> sparsityMapStrides({270400_Bit, 4225_Bit, 65_Bit, 1_Bit});
    const SmallVector<Bit> seTableStrides({135200_Bit, 135200_Bit, 2080_Bit, 32_Bit});
    const auto changedStrides = ndType.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides(), StridesRef(newStrides));
    const auto sparseChangedStrides = changedStrides.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseChangedStrides != nullptr);
    EXPECT_EQ(sparseChangedStrides.getData().cast<NDTypeInterface>().getStrides(), StridesRef(newInStrides));
    EXPECT_EQ(sparseChangedStrides.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(sparsityMapStrides));
    EXPECT_EQ(sparseChangedStrides.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(seTableStrides));

    // Check that only compact strides are accepted
    const SmallVector<Bit> unsupportedStrides({8652800_Bit, 135200_Bit, 2080_Bit, 32_Bit});
    EXPECT_ANY_THROW(ndType.changeStrides(StridesRef(unsupportedStrides)));

    const SmallVector<int64_t> tileOffsets({0, 0, 0, 0});
    const SmallVector<int64_t> inTileShape({1, 64, 9, 17});
    const SmallVector<int64_t> tileShape({1, 64, 17, 33});
    const SmallVector<int64_t> tileSETableShape({1, 1, 17, 33});
    const SmallVector<Bit> inTileStrides({156672_Bit, 2448_Bit, 272_Bit, 16_Bit});
    const SmallVector<Bit> tileStrides({524288_Bit, 8192_Bit, 512_Bit, 16_Bit});
    const SmallVector<Bit> tileSparsityMapStrides({35904_Bit, 561_Bit, 33_Bit, 1_Bit});
    const SmallVector<Bit> tileSETableStrides({17952_Bit, 17952_Bit, 1056_Bit, 32_Bit});
    const auto tiledType = ndType.extractDenseTile(ShapeRef(tileOffsets), ShapeRef(tileShape));
    EXPECT_EQ(tiledType.getShape(), ShapeRef(tileShape));
    const auto sparseTiledType = tiledType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseTiledType != nullptr);
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(inTileShape));
    EXPECT_EQ(sparseTiledType.getData().cast<NDTypeInterface>().getStrides(), StridesRef(inTileStrides));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseTiledType.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSparsityMapStrides));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(tileSETableShape));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSETableStrides));

    // Only dense strides are supported
    const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
    const SmallVector<Bit> viewTileSparsityMapStrides({270400_Bit, 4225_Bit, 65_Bit, 1_Bit});
    const SmallVector<Bit> viewInTileStrides({1048576_Bit, 16384_Bit, 512_Bit, 16_Bit});
    const auto viewTile = ndType.extractViewTile(ShapeRef(tileOffsets), ShapeRef(tileShape), ShapeRef(tileElemStrides));
    EXPECT_EQ(viewTile.getShape(), ShapeRef(tileShape));
    const auto sparseViewTile = viewTile.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparseViewTile != nullptr);
    EXPECT_EQ(sparseViewTile.getData().cast<NDTypeInterface>().getShape(), ShapeRef(inTileShape));
    EXPECT_EQ(sparseViewTile.getData().cast<NDTypeInterface>().getStrides(), StridesRef(viewInTileStrides));
    EXPECT_EQ(sparseViewTile.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(tileShape));
    EXPECT_EQ(sparseViewTile.getSparsityMap().cast<NDTypeInterface>().getStrides(),
              StridesRef(viewTileSparsityMapStrides));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getShape(), ShapeRef(tileSETableShape));
    EXPECT_EQ(sparseTiledType.getStorageElementTable().cast<NDTypeInterface>().getStrides(),
              StridesRef(tileSETableStrides));

    EXPECT_EQ(ndType.eraseTiledInfo(), ndType);

    const SmallVector<int64_t> padBefore({0, 2, 1, 1});
    const SmallVector<int64_t> padAfter({0, 2, 1, 1});
    const SmallVector<int64_t> paddedInShape({1, 68, 33, 33});
    const SmallVector<int64_t> paddedShape({1, 68, 67, 67});
    const SmallVector<int64_t> paddedSETableShape({1, 1, 67, 67});
    const auto paddedType = ndType.pad(ShapeRef(padBefore), ShapeRef(padAfter));
    EXPECT_EQ(paddedType.getShape(), ShapeRef(paddedShape));
    const auto sparsePaddedType = paddedType.dyn_cast<VPUIP::SparseBufferType>();
    ASSERT_TRUE(sparsePaddedType != nullptr);
    EXPECT_EQ(sparsePaddedType.getData().cast<NDTypeInterface>().getShape(), ShapeRef(paddedInShape));
    EXPECT_EQ(sparsePaddedType.getSparsityMap().cast<NDTypeInterface>().getShape(), ShapeRef(paddedShape));
    EXPECT_EQ(sparsePaddedType.getStorageElementTable().cast<NDTypeInterface>().getShape(),
              ShapeRef(paddedSETableShape));
}

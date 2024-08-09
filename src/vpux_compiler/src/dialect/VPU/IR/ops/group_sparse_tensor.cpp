//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;

//
// build
//

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           bool isWeights, VPU::SparsityCompressionAttr sparsityCompression) {
    build(builder, state, data, nullptr, nullptr, isWeights, sparsityCompression);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, bool isWeights,
                                           VPU::SparsityCompressionAttr sparsityCompression) {
    build(builder, state, data, sparsityMap, nullptr, isWeights, sparsityCompression);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, mlir::Value storageElementTable, bool isWeights,
                                           VPU::SparsityCompressionAttr sparsityCompression) {
    const auto isWeightsAttr = isWeights ? mlir::UnitAttr::get(builder.getContext()) : nullptr;
    build(builder, state, data, sparsityMap, storageElementTable, isWeightsAttr, sparsityCompression, nullptr);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap, mlir::Value storageElementTable,
                                           VPU::SEAttr seAttr) {
    build(builder, state, data, sparsityMap, storageElementTable, nullptr, nullptr, seAttr);
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::GroupSparseTensorOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*ranges*/,
        SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GroupSparseTensorOpAdaptor groupOp(operands, attrs, prop);
    if (mlir::failed(groupOp.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = groupOp.getData().getType();
    const auto sparsityMapType = groupOp.getSparsityMap() != nullptr ? groupOp.getSparsityMap().getType() : nullptr;
    const auto storageElementTableType =
            groupOp.getStorageElementTable() != nullptr ? groupOp.getStorageElementTable().getType() : nullptr;

    inferredReturnTypes.push_back(
            VPU::SparseTensorType::get(dataType, sparsityMapType, storageElementTableType, groupOp.getIsWeightsAttr(),
                                       groupOp.getSparsityCompressionAttr(), groupOp.getSeAttrAttr()));

    return mlir::success();
}

//
// MoveViewLikeOps
//

/*
 * Patterns such as the following:
 *      Data   Const SM   SETable
 *        \       |       /
 *        GroupSparseTensor
 *       /              \
 *     Slice           Slice
 *
 * get transformed into:
 *      Data    Const SM*   SETable     Const Data  Const SM* SETable
 *        |        |         |             |           |       |
 *      Slice      |       Slice         Slice         |      Slice
 *        \        |        /               \          |      /
 *         GroupSparseTensor                 GroupSparseTensor
 *
 * This can allow the Slice canonicalizer convert the operation into a constant transformation.
 * The sparsity map for weights is attached directly as a transformation since the original constant's type has the
 * shape flattened for each output channel (i.e. OCx1x1xSIZExi1), making it incompatible with the attributes of the
 * Slice operation. Therefore, it is applied as a transformation to the dense constant before the
 * transformation that generates the sparsity map.
 * StorageElementTableOp has its own canonicalizer and SliceOp will be fused into it.
 */

namespace {

class MoveViewLikeOps final : public mlir::OpRewritePattern<VPU::GroupSparseTensorOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::GroupSparseTensorOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult MoveViewLikeOps::matchAndRewrite(VPU::GroupSparseTensorOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    auto origDataValue = origOp.getData();
    if (origDataValue == nullptr) {
        return mlir::failure();
    }
    auto sparsityMapValue = origOp.getSparsityMap();

    VPU::StorageElementTableOp seTableOp = nullptr;
    if (origOp.getStorageElementTable() != nullptr) {
        seTableOp = origOp.getStorageElementTable().getDefiningOp<VPU::StorageElementTableOp>();
        if (seTableOp == nullptr) {
            return mlir::failure();
        }
    }

    auto ctx = getContext();
    for (auto userOp : llvm::make_early_inc_range(origOp.getOutput().getUsers())) {
        if (auto sliceUserOp = mlir::dyn_cast<VPU::SliceOp>(userOp)) {
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceUserOp.getStaticOffsets());
            const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceUserOp.getStaticSizes());

            auto seAttr = origOp.getSeAttr().value_or(nullptr);
            auto sparsityCompressionAttr = origOp.getSparsityCompression().value_or(nullptr);
            if (sparsityCompressionAttr != nullptr) {
                sparsityCompressionAttr =
                        VPU::tileSparsityCompression(sparsityCompressionAttr, Shape(sliceOffsets), Shape(sliceSizes));
            }

            auto rewriteInput = [&](mlir::Value value, ShapeRef offsets, ShapeRef sizes) {
                if (auto constOp = value.getDefiningOp<Const::DeclareOp>()) {
                    auto newContentAttr = constOp.getContentAttr().subview(offsets, sizes);
                    auto newConstOp = rewriter.create<Const::DeclareOp>(constOp.getLoc(), newContentAttr.getType(),
                                                                        newContentAttr);
                    return newConstOp.getOutput();
                }
                auto newSliceOp = rewriter.create<VPU::SliceOp>(value.getLoc(), value, getIntArrayAttr(ctx, offsets),
                                                                getIntArrayAttr(ctx, sizes));
                return newSliceOp.getResult();
            };

            // Data
            auto newDataOffsets = Shape(sliceOffsets);
            auto newDataSizes = Shape(sliceSizes);
            if (seAttr != nullptr) {
                seAttr = seAttr.extractTile(Shape(sliceOffsets), Shape(sliceSizes),
                                            origDataValue.getType().cast<NDTypeInterface>().getShape(), newDataOffsets,
                                            newDataSizes);
            }
            auto newDataValue = rewriteInput(origDataValue, newDataOffsets, newDataSizes);

            // SM
            mlir::Value newSparsityMapValue = nullptr;
            if (sparsityMapValue != nullptr) {
                newSparsityMapValue = rewriteInput(sparsityMapValue, Shape(sliceOffsets), Shape(sliceSizes));
            }

            // SETable
            mlir::Value newSETableValue = nullptr;
            if (seTableOp != nullptr) {
                auto seTableOffsets = sliceOffsets;
                auto seTableSizes = sliceSizes;
                seTableOffsets[Dims4D::Act::N.ind()] = 0;
                seTableSizes[Dims4D::Act::N.ind()] = 1;

                const auto seSliceOffset = std::div(sliceOffsets[Dims4D::Act::C.ind()], seTableOp.getSeSize());
                VPUX_THROW_WHEN(seSliceOffset.rem != 0, "Slice over channels offset is not aligned with SE size");
                seTableOffsets[Dims4D::Act::C.ind()] = seSliceOffset.quot;

                const auto seSliceSize = std::div(sliceSizes[Dims4D::Act::C.ind()], seTableOp.getSeSize());
                VPUX_THROW_WHEN(seSliceSize.rem != 0, "Slice over channels size is not aligned with SE size");
                seTableSizes[Dims4D::Act::C.ind()] = seSliceSize.quot;

                auto seTableSliceOp = rewriter.create<VPU::SliceOp>(
                        sliceUserOp.getLoc(), origOp.getStorageElementTable(), getIntArrayAttr(ctx, seTableOffsets),
                        getIntArrayAttr(ctx, seTableSizes));
                newSETableValue = seTableSliceOp.getResult();
            }
            rewriter.replaceOpWithNewOp<VPU::GroupSparseTensorOp>(sliceUserOp, newDataValue, newSparsityMapValue,
                                                                  newSETableValue, origOp.getIsWeightsAttr(),
                                                                  sparsityCompressionAttr, seAttr);
        }
    }

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::GroupSparseTensorOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results,
                                                                 mlir::MLIRContext* ctx) {
    results.add<MoveViewLikeOps>(ctx);
}

//
// TilingViewLikeOpInterface
//

InputTiling vpux::VPU::GroupSparseTensorOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    VPU::StorageElementTableOp seTableOp = nullptr;
    if (auto seTable = getStorageElementTable()) {
        if (auto blockArg = seTable.dyn_cast<mlir::BlockArgument>()) {
            if (auto blockParent = this->getOperation()->getParentOp()) {
                VPUX_THROW_WHEN(blockParent->getNumOperands() < blockArg.getArgNumber(),
                                "Number of block operands {0} doesn't match with argument number {1}",
                                blockParent->getNumOperands(), blockArg.getArgNumber());
                seTableOp =
                        blockParent->getOperand(blockArg.getArgNumber()).getDefiningOp<VPU::StorageElementTableOp>();
            }
        } else {
            seTableOp = getStorageElementTable().getDefiningOp<VPU::StorageElementTableOp>();
        }
    }

    auto inputTile = TileInfo(outputTile.shape, outputTile.offsets, outputTile.axis);
    if (auto seAttr = getSeAttr().value_or(nullptr)) {
        seAttr.extractTile(outputTile.offsets, outputTile.shape, getShape(getData()), inputTile.offsets,
                           inputTile.shape);
    }

    SmallVector<TileInfo> inputTiles = {inputTile};
    auto sparsityMapValue = getSparsityMap();

    if (sparsityMapValue != nullptr) {
        inputTiles.push_back(outputTile);
    }

    if (seTableOp != nullptr) {
        Shape seTableOffsets = outputTile.offsets;
        Shape seTableSizes = outputTile.shape;
        seTableOffsets[Dims4D::Act::N] = 0;
        seTableSizes[Dims4D::Act::N] = 1;

        const auto seSliceOffset = std::div(seTableOffsets[Dims4D::Act::C], seTableOp.getSeSize());
        VPUX_THROW_WHEN(seSliceOffset.rem != 0, "Slice over channels offset is not aligned with SE size");
        seTableOffsets[Dims4D::Act::C] = seSliceOffset.quot;

        const auto seSliceSize = std::div(seTableSizes[Dims4D::Act::C], seTableOp.getSeSize());
        VPUX_THROW_WHEN(seSliceSize.rem != 0, "Slice over channels size is not aligned with SE size");
        seTableSizes[Dims4D::Act::C] = seSliceSize.quot;

        inputTiles.push_back(TileInfo(seTableSizes, seTableOffsets, Shape(seTableOffsets.size(), 1)));
    }
    return InputTiling(inputTiles);
}

void vpux::VPU::GroupSparseTensorOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile,
                                                 ShapeRef outputShape) {
    VPUX_THROW_WHEN(inputTiling.tiles.empty(), "There is no tiling for {0}", getLoc());
    if (auto seAttr = getSeAttr().value_or(nullptr)) {
        auto inputTile = inputTiling.tiles.front();
        const auto inputShape = seAttr.backInferInputShape(outputShape);
        seAttr = seAttr.extractTile(outputTile.offsets, outputTile.shape, inputShape, inputTile.offsets,
                                    inputTile.shape);
        setSeAttrAttr(seAttr);

        auto dataType = getData().getType().cast<vpux::NDTypeInterface>();
        const auto sparsityMapType = getSparsityMap() != nullptr ? getSparsityMap().getType() : nullptr;
        const auto storageElementTableType =
                getStorageElementTable() != nullptr ? getStorageElementTable().getType() : nullptr;

        auto newType = VPU::SparseTensorType::get(dataType, sparsityMapType, storageElementTableType,
                                                  getIsWeightsAttr(), getSparsityCompressionAttr(), seAttr);

        getOutput().setType(newType);
    }
}

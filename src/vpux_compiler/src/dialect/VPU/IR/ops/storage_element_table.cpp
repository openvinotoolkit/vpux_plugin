//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

void vpux::VPU::StorageElementTableOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                             ArrayRef<int64_t> dataShape, mlir::Type dataElemType, int64_t seSize,
                                             int64_t seDepth, VPU::SEAttr seAttr) {
    auto dataShapeAttr = getIntArrayAttr(odsBuilder.getContext(), dataShape);
    build(odsBuilder, odsState, dataShapeAttr, dataElemType, seSize, seDepth, seAttr, nullptr, nullptr);
}

mlir::LogicalResult vpux::VPU::StorageElementTableOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::StorageElementTableOpAdaptor setOp(operands, attrs, prop);
    if (mlir::failed(setOp.verify(loc))) {
        return mlir::failure();
    }

    const auto depth = setOp.getSeDepth();
    const auto dataShape = parseIntArrayAttr<int64_t>(setOp.getDataShape());
    VPUX_THROW_UNLESS(dataShape.size() == 4, "Expected 4D input data, got {0} dimensions", dataShape.size());

    Shape shapeAfterSERead(dataShape);
    if (auto seAttrValue = setOp.getSeAttr().value_or(nullptr)) {
        shapeAfterSERead = seAttrValue.inferOutputShape(shapeAfterSERead);
    }
    const auto height = shapeAfterSERead[Dims4D::Act::H];
    const auto width = shapeAfterSERead[Dims4D::Act::W];
    SmallVector<int64_t> shape{1, depth, height, width};

    const auto outType = getTensorType(ShapeRef(shape), getInt32Type(ctx), DimsOrder::NHWC, nullptr);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::StorageElementTableOp::verify() {
    const auto setOp = getOperation();
    using namespace VPU::NCESparsity;

    if (auto seAttrValue = getSeAttr().value_or(nullptr)) {
        if (!seAttrValue.isa<VPU::SEAttr>()) {
            return errorAt(setOp->getLoc(), "Only VPU::SEAttr is supported for Storage Element Table");
        }
    }

    if (!getBasePtrs().has_value()) {
        return mlir::success();
    }

    const auto opBasePtrs = getBasePtrs().value().getValues<int32_t>();
    const auto expectedNumPtrs = getOutput().getType().cast<vpux::NDTypeInterface>().getNumElements();
    if (static_cast<size_t>(expectedNumPtrs) != opBasePtrs.size()) {
        return errorAt(setOp->getLoc(), "StorageElementTable expects to have {0}, but got {1}", expectedNumPtrs,
                       opBasePtrs.size());
    }

    return mlir::success();
}

namespace {
/*
 * StorageElementTableOp is constant operation therefore
 * child pure view like operations can be fused into it.
 * Currently SliceOp can be fused.
 */
class FuseChildSliceOps final : public mlir::OpRewritePattern<VPU::StorageElementTableOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::StorageElementTableOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseChildSliceOps::matchAndRewrite(VPU::StorageElementTableOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto seAttr = origOp.getSeAttr().value_or(nullptr);
    if (seAttr == nullptr) {
        return mlir::failure();
    }

    for (auto userOp : llvm::make_early_inc_range(origOp.getOutput().getUsers())) {
        if (auto sliceUserOp = mlir::dyn_cast<VPU::SliceOp>(userOp)) {
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceUserOp.getStaticOffsets());
            const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceUserOp.getStaticSizes());

            auto effectiveOutputOffsets = sliceOffsets;
            auto effectiveOutputSizes = sliceSizes;
            effectiveOutputOffsets[Dims4D::Act::C.ind()] *= origOp.getSeSize();
            effectiveOutputSizes[Dims4D::Act::C.ind()] *= origOp.getSeSize();

            const auto inputDataShape = Shape(parseIntArrayAttr<int64_t>(origOp.getDataShape()));
            auto inputTileShape = Shape(inputDataShape.size());
            auto inputTileOffset = inputTileShape;
            auto newSeAttr = seAttr.extractTile(Shape(effectiveOutputOffsets), Shape(effectiveOutputSizes),
                                                inputDataShape, inputTileOffset, inputTileShape);
            rewriter.replaceOpWithNewOp<VPU::StorageElementTableOp>(sliceUserOp, inputTileShape.raw(),
                                                                    origOp.getDataElemType(), origOp.getSeSize(),
                                                                    sliceSizes[Dims4D::Act::C.ind()], newSeAttr);
        }
    }
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::StorageElementTableOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results,
                                                                   mlir::MLIRContext* ctx) {
    results.add<FuseChildSliceOps>(ctx);
}

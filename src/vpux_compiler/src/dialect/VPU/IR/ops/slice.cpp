//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// build
//

void vpux::VPU::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void vpux::VPU::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::SliceOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SliceOpAdaptor sliceOp(operands, attrs, prop);
    if (mlir::failed(sliceOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = sliceOp.getSource().getType().dyn_cast<vpux::NDTypeInterface>();
    if (origType == nullptr) {
        return errorAt(loc, "VPU::SliceOp operand must have vpux::NDTypeInterface type");
    }

    const auto sliceShape = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());

    if (sliceShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice shape '{0}' doesn't match RankedTensor rank '{1}'", sliceShape, origType.getRank());
    }
    if (sliceOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice offsets '{0}' doesn't match RankedTensor rank '{1}'", sliceOffsets,
                       origType.getRank());
    }

    auto inferExplicitDistributedAttr = [&](VPU::DistributionInfoAttr origDistribution,
                                            ArrayRef<int64_t> inShape) -> VPU::DistributionInfoAttr {
        if (origDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED ||
            !VPU::isSegmentedOverlappedAxisSameAsSliceAxis(origDistribution.getNumTiles(), inShape, sliceShape)) {
            return VPU::getExplicitDistrAttrForSliceLikeOps(origDistribution, sliceShape, inShape, ctx);
        }

        // When clustering axis == slice axis, we cannot infer per cluster shape from op itself
        // and therefore this should be correctly computed in pass that creates the Slice Op
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(origDistribution.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < inShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (sliceShape[dim] != inShape[dim]) {
                    memoryShapes[cluster][dim] = sliceShape[dim];
                }
            }
        }
        const auto perClusterShapesAttr = vpux::getIntArrayOfArray(ctx, memoryShapes);
        const auto zeroOffsets =
                SmallVector<SmallVector<int64_t>>(memoryShapes.size(), SmallVector<int64_t>(inShape.size(), 0));
        const auto perClusterOffsetsAttr = vpux::getIntArrayOfArray(ctx, zeroOffsets);

        return VPU::DistributionInfoAttr::get(
                ctx, origDistribution.getMode(), origDistribution.getNumTiles(), origDistribution.getKernel(),
                origDistribution.getPads(), origDistribution.getStrides(), origDistribution.getNumClusters(),
                origDistribution.getAlignment(), origDistribution.getUniformDistributedSegments(), perClusterShapesAttr,
                perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr,
                origDistribution.getEqualMemoryAndComputeView());
    };

    const auto distributedIn = origType.dyn_cast<VPU::DistributedTypeInterface>();
    VPU::DistributionInfoAttr possibleDistribution =
            distributedIn != nullptr && distributedIn.containsDistributedTypes()
                    ? distributedIn.getDistributedTypes().front().cast<VPU::DistributedTensorType>().getDistribution()
                    : nullptr;

    if (possibleDistribution != nullptr && VPU::isDistributedAttrWithExplicitShapesAndOffsets(possibleDistribution)) {
        if (auto sparseType = distributedIn.dyn_cast<VPU::SparseTensorType>()) {
            possibleDistribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
        }

        auto newDistribution =
                VPU::updateSliceLikeOpsAlignment(ctx, origType.getShape(), ShapeRef(sliceShape), possibleDistribution);

        const auto sliceDistributedAttr = inferExplicitDistributedAttr(newDistribution, origType.getShape().raw());

        const auto newType = distributedIn.extractDenseTileForExplicitDistribution(
                ShapeRef(sliceOffsets), ShapeRef(sliceShape), sliceDistributedAttr);
        inferredTypes.emplace_back(newType);
    } else {
        const auto newType = origType.extractDenseTile(ShapeRef(sliceOffsets), ShapeRef(sliceShape));
        inferredTypes.emplace_back(newType);
    }

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult VPU::SliceOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getSource().getType() == getResult().getType()) {
        return getSource();
    }

    if (const auto origContent = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto offset = Shape(parseIntArrayAttr<int64_t>(getStaticOffsets()));
        const auto shape = Shape(parseIntArrayAttr<int64_t>(getStaticSizes()));
        return static_cast<Const::ContentAttr>(origContent).transform().subview(offset, shape).get();
    }

    return nullptr;
}

//
// ComposeSlice
//

namespace {

class ComposeSlice final : public mlir::OpRewritePattern<VPU::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::SliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeSlice::matchAndRewrite(VPU::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSliceOp = origOp.getSource().getDefiningOp<VPU::SliceOp>();
    if (producerSliceOp == nullptr) {
        return mlir::failure();
    }

    auto finalOffsets = parseIntArrayAttr<int64_t>(producerSliceOp.getStaticOffsets());
    const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
    for (auto i : irange(finalOffsets.size())) {
        finalOffsets[i] += secondOffsets[i];
    }

    const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
    const auto finalShapeAttr = origOp.getStaticSizes();
    rewriter.replaceOpWithNewOp<VPU::SliceOp>(origOp, producerSliceOp.getSource(), finalOffsetsAttr, finalShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::SliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSlice>(ctx);
}

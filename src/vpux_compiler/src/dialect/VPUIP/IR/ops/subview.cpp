//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// build
//

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             mlir::ArrayAttr static_offsets, mlir::ArrayAttr static_sizes) {
    build(builder, state, input, static_offsets, static_sizes, nullptr, nullptr);
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             mlir::ArrayAttr static_offsets, mlir::ArrayAttr static_sizes,
                             mlir::ArrayAttr static_strides) {
    build(builder, state, input, static_offsets, static_sizes, static_strides, nullptr);
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ShapeRef static_offsets, ShapeRef static_sizes, ShapeRef static_strides) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw(), static_strides.raw());
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes,
                             ArrayRef<int64_t> static_strides) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes), getIntArrayAttr(builder.getContext(), static_strides),
          nullptr);
}

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::SubViewOp::getViewSource() {
    return getSource();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult VPUIP::SubViewOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties props, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::SubViewOpAdaptor subViewOp(operands, attrs, props);
    if (mlir::failed(subViewOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = subViewOp.getSource().getType().cast<NDTypeInterface>();

    const auto subViewShape = parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes());
    const auto subViewOffsets = parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets());
    const auto subViewStrides = subViewOp.getStaticStrides().has_value()
                                        ? parseIntArrayAttr<int64_t>(subViewOp.getStaticStrides().value())
                                        : SmallVector<int64_t>(origType.getRank(), 1);

    if (subViewShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile shape '{0}' doesn't match MemRef rank '{1}'", subViewShape, origType.getRank());
    }
    if (subViewOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile offsets '{0}' doesn't match MemRef rank '{1}'", subViewOffsets, origType.getRank());
    }
    if (subViewStrides.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile strides '{0}' doesn't match MemRef rank '{1}'", subViewStrides, origType.getRank());
    }

    const auto hasExplcitOutputShapes = subViewOp.getExplicitOutputShapes().has_value();

    auto inferExplicitDistributedAttr = [&](VPU::DistributionInfoAttr origDistribution,
                                            ArrayRef<int64_t> inShape) -> VPU::DistributionInfoAttr {
        auto mode = origDistribution.getMode().getValue();
        if (hasExplcitOutputShapes) {
            // Track #E125638
            // Other modes should be supported
            VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED, "Can not set explicit shapes with mode {0}",
                              VPU::stringifyDistributionMode(mode));
            auto explicitOutputShapes = subViewOp.getExplicitOutputShapes().value();
            return VPU::getSegmentedExplicitDistrAttrForSliceLikeOps(origDistribution, subViewShape,
                                                                     explicitOutputShapes, ctx);
        }
        if (mode != VPU::DistributionMode::OVERLAPPED ||
            !VPU::isSegmentedOverlappedAxisSameAsSliceAxis(origDistribution.getNumTiles(), inShape, subViewShape)) {
            return VPU::getExplicitDistrAttrForSliceLikeOps(origDistribution, subViewShape, inShape, ctx);
        }

        // When clustering axis == slice axis, we cannot infer per cluster shape from op itself
        // and therefore this should be correctly computed in pass that creates the Subview Op
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(origDistribution.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < inShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (subViewShape[dim] != inShape[dim]) {
                    memoryShapes[cluster][dim] = subViewShape[dim];
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
                    ? distributedIn.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>().getDistribution()
                    : nullptr;
    if (possibleDistribution != nullptr) {
        if (hasExplcitOutputShapes || VPU::isDistributedAttrWithExplicitShapesAndOffsets(possibleDistribution)) {
            if (auto sparseType = distributedIn.dyn_cast<VPUIP::SparseBufferType>()) {
                possibleDistribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
            }

            // update subview alignment
            auto newDistribution = VPU::updateSliceLikeOpsAlignment(ctx, origType.getShape(), ShapeRef(subViewShape),
                                                                    possibleDistribution);

            const auto subViewDistributedAttr =
                    inferExplicitDistributedAttr(newDistribution, origType.getShape().raw());

            const auto subViewType = distributedIn.extractViewTileForExplicitDistribution(
                    ShapeRef(subViewOffsets), ShapeRef(subViewShape), ShapeRef(subViewStrides), subViewDistributedAttr);
            inferredTypes.push_back(subViewType);
        } else {
            // todo: update alignment for non-explict sparseBuffer to enable 37XX unaligned shave tiling
            // ticket E#114487
            if (auto sparseType = distributedIn.dyn_cast<VPUIP::SparseBufferType>()) {
                const auto subViewType = sparseType.extractViewTile(ShapeRef(subViewOffsets), ShapeRef(subViewShape),
                                                                    ShapeRef(subViewStrides));
                inferredTypes.push_back(subViewType);
            } else {
                auto newDistribution = VPU::updateSliceLikeOpsAlignment(ctx, origType.getShape(),
                                                                        ShapeRef(subViewShape), possibleDistribution);

                const auto origBufferType =
                        distributedIn.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>();
                auto newBufferType = VPUIP::DistributedBufferType::get(
                        ctx, origBufferType.getShape().raw(), origBufferType.getElementType(),
                        origBufferType.getLayout(), origBufferType.getMemSpace(), newDistribution,
                        origBufferType.getSparsityCompression());

                const auto subViewType = newBufferType.extractViewTile(ShapeRef(subViewOffsets), ShapeRef(subViewShape),
                                                                       ShapeRef(subViewStrides));
                inferredTypes.push_back(subViewType);
            }
        }
    } else {
        const auto subViewType =
                origType.extractViewTile(ShapeRef(subViewOffsets), ShapeRef(subViewShape), ShapeRef(subViewStrides));

        inferredTypes.push_back(subViewType);
    }

    return mlir::success();
}

// A sparsity map constant has the workload size flattened, so that its shape is OCx1x1xSIZE.
// Therefore, only subviews over the OC dimension are allowed.
// Additionally, the GetSparsityMap transformation is the last one in the list. When folding
// subviews into the constant, it will be introduced as a transformation before it, so its
// subview dimensions have to be adapted for the shape before flattening.
void adaptSparsityMapConstant(mlir::Value source, Shape& offset, Shape& shape) {
    auto constParentOp = source.getDefiningOp<Const::DeclareOp>();
    if (constParentOp == nullptr) {
        return;
    }
    const auto transformations = constParentOp.getContentAttr().getTransformations();
    if (transformations.empty()) {
        return;
    }

    auto getSparistyMapTransIt = std::find_if(transformations.rbegin(), transformations.rend(),
                                              [&](vpux::Const::TransformAttrInterface trans) {
                                                  return trans.isa<Const::GetSparsityMapAttr>();
                                              });
    if (getSparistyMapTransIt == transformations.rend()) {
        return;
    }

    auto posFromEnd = std::distance(transformations.rbegin(), getSparistyMapTransIt);

    const auto zeroWorkloadOffsets = std::all_of(offset.begin() + 1, offset.end(), [](const int64_t value) {
        return value == 0;
    });
    VPUX_THROW_UNLESS(zeroWorkloadOffsets, "Offsets with non-zero values for workloads are not supported. Got {0}",
                      offset);

    const auto sparsityMapShape = constParentOp.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto sparsityMapWorkloadShape = SmallVector<int64_t>(sparsityMapShape.begin() + 1, sparsityMapShape.end());
    const auto shapeWorkloadShape = SmallVector<int64_t>(shape.begin() + 1, shape.end());
    for (auto p : zip(sparsityMapWorkloadShape, shapeWorkloadShape)) {
        const auto sparsityMapDim = std::get<0>(p);
        const auto shapeDim = std::get<1>(p);
        VPUX_THROW_UNLESS(sparsityMapDim == shapeDim,
                          "Subview shape with different workload size is not supported: original dim {0}, new dim {1}",
                          sparsityMapDim, shapeDim);
    }

    auto inputType = constParentOp.getContentAttr().getBaseContent().getType().cast<vpux::NDTypeInterface>();
    for (auto idx : irange(transformations.size() - (1 + posFromEnd))) {
        inputType = transformations[idx].inferOutputType(inputType);
    }
    const auto inputShape = inputType.getShape().raw();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Expected a 4-dimensional type, got {0} dimensions", inputShape.size());
    const auto OC = shape.raw()[0];
    shape = Shape({OC, inputShape[1], inputShape[2], inputShape[3]});
}

//
// fold
//

mlir::OpFoldResult VPUIP::SubViewOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getSource().getType() == getResult().getType()) {
        return getSource();
    }

    if (const auto origContent = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        auto offset = Shape(parseIntArrayAttr<int64_t>(getStaticOffsets()));
        auto shape = Shape(parseIntArrayAttr<int64_t>(getStaticSizes()));
        adaptSparsityMapConstant(getSource(), offset, shape);
        return static_cast<Const::ContentAttr>(origContent).transform().subview(offset, shape).get();
    }

    return nullptr;
}

//
// ComposeSubView
//

namespace {

class ComposeSubView final : public mlir::OpRewritePattern<VPUIP::SubViewOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::SubViewOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeSubView::matchAndRewrite(VPUIP::SubViewOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSubViewOp = origOp.getSource().getDefiningOp<VPUIP::SubViewOp>();
    if (producerSubViewOp == nullptr) {
        return mlir::failure();
    }

    if (origOp.getStaticStrides().has_value() || producerSubViewOp.getStaticStrides().has_value()) {
        return mlir::failure();
    }

    auto finalOffsets = parseIntArrayAttr<int64_t>(producerSubViewOp.getStaticOffsets());
    const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
    for (auto i : irange(finalOffsets.size())) {
        finalOffsets[i] += secondOffsets[i];
    }

    const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
    const auto finalShapeAttr = origOp.getStaticSizes();
    rewriter.replaceOpWithNewOp<VPUIP::SubViewOp>(origOp, producerSubViewOp.getSource(), finalOffsetsAttr,
                                                  finalShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::SubViewOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSubView>(ctx);
}

//
// verify
//

mlir::LogicalResult VPUIP::SubViewOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };
    mlir::SmallVector<mlir::Type> inferredTypes;
    if (inferReturnTypes(getContext(), getLoc(), op->getOperands(), op->getAttrDictionary(), op->getPropertiesStorage(),
                         op->getRegions(), inferredTypes)
                .failed()) {
        logCb(formatv("Can't infer return types"));
        return mlir::failure();
    }
    const auto expectedStrides = inferredTypes.front().cast<NDTypeInterface>().getStrides();
    const auto outputStrides = getResult().getType().cast<NDTypeInterface>().getStrides();
    if (expectedStrides.size() != outputStrides.size()) {
        logCb(formatv("The output stride size != infered stride size"));
        return mlir::failure();
    }

    for (auto j : irange(expectedStrides.size())) {
        if (outputStrides[Dim(j)] != expectedStrides[Dim(j)]) {
            logCb(formatv("The output stride({0}) != infered stride({1})", outputStrides, expectedStrides));
            return mlir::failure();
        }
    }
    return mlir::success();
}

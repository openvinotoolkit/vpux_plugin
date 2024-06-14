//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinOps.h>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputSortOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputSortOpAdaptor sort(operands, attrs);
    if (mlir::failed(sort.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = sort.getConfidence().getType().cast<NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    const auto numClasses = inputShape[Dims4D::Act::H];
    const auto numPriors = inputShape[Dims4D::Act::W];

    const auto outConfidenceShape = SmallVector<int64_t>{1, 1, numClasses, numPriors};
    const auto outIndicesShape = SmallVector<int64_t>{1, 1, numClasses, numPriors};
    const auto outSizesShape = SmallVector<int64_t>{1, 1, numClasses, 1};

    const auto outConfidenceType = inputType.changeShape(Shape(outConfidenceShape));

    const auto outIndicesElemType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const auto outIndicesType = mlir::RankedTensorType::get(outIndicesShape, outIndicesElemType);
    const auto outSizesType = mlir::RankedTensorType::get(outSizesShape, outIndicesElemType);

    inferredReturnTypes.push_back(outConfidenceType);
    inferredReturnTypes.push_back(outIndicesType);
    inferredReturnTypes.push_back(outSizesType);

    return mlir::success();
}

mlir::Value createIndicesAuxiliaryBuffer(mlir::OpBuilder& rewriter, ShapeRef shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "Class predictions tensor must be 4D");
    auto auxIndicesContent = std::vector<int32_t>(shape.totalSize());
    const auto width = shape[Dims4D::Act::W];
    const auto height = shape[Dims4D::Act::H];
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            auxIndicesContent[h * width + w] = w;
        }
    }

    const auto auxIndicesType = mlir::RankedTensorType::get(shape.raw(), getSInt32Type(rewriter.getContext()));
    const auto auxIndicesAttr = mlir::DenseElementsAttr::get(auxIndicesType, ArrayRef(auxIndicesContent));
    auto auxIndicesContentAttr = Const::ContentAttr::get(auxIndicesAttr);

    return rewriter.create<Const::DeclareOp>(
            appendLoc(mlir::UnknownLoc::get(rewriter.getContext()), "sort_IndicesBuffer"),
            auxIndicesContentAttr.getType(), auxIndicesContentAttr);
}

mlir::Value createSortingAuxiliaryBuffer(mlir::OpBuilder& rewriter, ShapeRef shape) {
    const auto auxIndicesType = mlir::RankedTensorType::get(shape.raw(), getSInt32Type(rewriter.getContext()));
    const auto auxIndicesAttr = mlir::DenseElementsAttr::get(auxIndicesType, 0);
    auto auxIndicesContentAttr = Const::ContentAttr::get(auxIndicesAttr);

    return rewriter.create<Const::DeclareOp>(
            appendLoc(mlir::UnknownLoc::get(rewriter.getContext()), "sort_SortingBuffer"),
            auxIndicesContentAttr.getType(), auxIndicesContentAttr);
}

static mlir::ModuleOp getModule(::mlir::OpBuilder& odsBuilder) {
    auto block = odsBuilder.getInsertionBlock();
    auto parentOp = block->getParentOp();
    while (parentOp && !llvm::isa<mlir::ModuleOp>(parentOp)) {
        parentOp = parentOp->getParentOp();
    }
    return llvm::cast<mlir::ModuleOp>(parentOp);
}

void vpux::VPU::DetectionOutputSortOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                             ::mlir::Value classPredictions, ::mlir::FloatAttr confidenceThreshold,
                                             ::mlir::IntegerAttr topK) {
    auto indicesBuffer = createIndicesAuxiliaryBuffer(odsBuilder, getShape(classPredictions));

    auto module = getModule(odsBuilder);
    auto numShaves = IE::getTotalNumOfActShaveEngines(module);

    // 4 buffers of size 256 elements are required for counting sort
    // tensor has SEGMENTED distribution mode
    // multiply the buffer numShaves times to provide unique buffer for each shave
    auto sortingBuffer = createSortingAuxiliaryBuffer(odsBuilder, Shape{1, 1, 4 * numShaves, 256});

    build(odsBuilder, odsState, classPredictions, indicesBuffer, sortingBuffer, confidenceThreshold, topK, nullptr);
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::DetectionOutputSortOp::backInferTileInfo(const vpux::TileInfo& firstOutputTile,
                                                                vpux::Logger /*log*/) {
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numShaves = IE::getTotalNumOfActShaveEngines(module);
    return DetectionOutputSortOpInputTiling(firstOutputTile, numShaves);
}

void vpux::VPU::DetectionOutputSortOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    return;
}

mlir::FailureOr<OutputTiling> vpux::VPU::DetectionOutputSortOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

OutputTiling vpux::VPU::DetectionOutputSortOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                               vpux::Logger /*log*/) {
    return DetectionOutputSortOpOutputTiling(firstOutputTile);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::DetectionOutputSortOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::DetectionOutputSortOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                   distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
}

bool VPU::DetectionOutputSortOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& outputTile) {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto tileOp = IE::getTileExecutor(moduleOp);
    auto height = outputTile.shape[Dims4D::Act::H];
    return height >= tileOp.getCount();
}

bool VPU::DetectionOutputSortOp::isOperationSplitOverWidthCompatible(ShapeRef, ShapeRef, ShapeRef) {
    return false;
}

bool VPU::DetectionOutputSortOp::isOperationSplitOverKernelCompatible(ShapeRef, ShapeRef, ShapeRef) {
    return false;
}

bool VPU::DetectionOutputSortOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto op = mlir::cast<VPU::DetectionOutputSortOp>(getOperation());
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(op, outputType.getShape()[Dims4D::Act::H], strategy);

    auto distributedTensorTypes = SmallVector<vpux::NDTypeInterface>();
    for (const auto& operand : op.getOperands()) {
        const auto distributedType = getDistributedActivationTypeFromOp(op, operand.getType(), numClusters, strategy);
        distributedTensorTypes.push_back(distributedType);
    }

    for (const auto& result : op.getResults()) {
        const auto distributedType = getDistributedOutputTypeFromOp(op, result.getType(), numClusters, strategy);
        distributedTensorTypes.push_back(distributedType);
    }

    return fitIntoCMX(distributedTensorTypes, reservedMem);
}

//
// SWOpInterface
//

bool vpux::VPU::DetectionOutputSortOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::DetectionOutputSortOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DetectionOutputSortOp::supportCycleCostCalculation() {
    return false;
}

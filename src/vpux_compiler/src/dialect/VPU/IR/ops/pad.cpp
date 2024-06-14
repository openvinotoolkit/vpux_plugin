//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::PadOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PadOpAdaptor pad(operands, attrs);
    if (mlir::failed(pad.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = pad.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape();

    auto padBegin = IE::extractPads(loc, pad.getPadsBegin(), pad.getPadsBeginAttr(), inputShape);
    if (mlir::failed(padBegin)) {
        return mlir::failure();
    }
    const auto padEnd = IE::extractPads(loc, pad.getPadsEnd(), pad.getPadsEndAttr(), inputShape);
    if (mlir::failed(padEnd)) {
        return mlir::failure();
    }
    if (pad.getMode() == IE::PadMode::CONSTANT && pad.getPadValue() == nullptr && !pad.getPadValueAttr().has_value()) {
        return errorAt(loc, "pad_mode is CONSTANT but pad_value hasn't provided");
    }

    const auto newType = inType.pad(ShapeRef(padBegin.value()), ShapeRef(padEnd.value()));
    inferredReturnTypes.push_back(newType);

    return mlir::success();
}

InputTiling vpux::VPU::PadOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto inShape = getShape(getInput());
    const auto outShape = getShape(getOutput());
    const auto padsBegin = Shape(parseIntArrayAttr<int64_t>(getPadsBeginAttrAttr()));
    const auto padsEnd = Shape(parseIntArrayAttr<int64_t>(getPadsEndAttrAttr()));

    return vpux::backInferPadTile(outputTile, inShape, outShape, padsBegin, padsEnd, log);
}

void vpux::VPU::PadOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& outputTile) {
    const auto outShape = getShape(getOutput());
    auto padsBegin = parseIntArrayAttr<int64_t>(getPadsBeginAttr().value());
    auto padsEnd = parseIntArrayAttr<int64_t>(getPadsEndAttr().value());

    vpux::updatePadOpAttrsAfterTiling(outShape, outputTile, padsBegin, padsEnd);

    const auto newPadsBeginAttr = getIntArrayAttr(getContext(), padsBegin);
    const auto newPadsEndAttr = getIntArrayAttr(getContext(), padsEnd);
    setPadsBeginAttrAttr(newPadsBeginAttr);
    setPadsEndAttrAttr(newPadsEndAttr);
}

mlir::FailureOr<OutputTiling> vpux::VPU::PadOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::PadOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// build
//

void vpux::VPU::PadOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                             ::mlir::Value pads_begin, ::mlir::Value pads_end, ::mlir::Value pad_value,
                             ::mlir::ArrayAttr pads_begin_attr, ::mlir::ArrayAttr pads_end_attr,
                             ::mlir::FloatAttr pad_value_attr, vpux::IE::PadModeAttr mode) {
    build(builder, state, input, pads_begin, pads_end, pad_value, pads_begin_attr, pads_end_attr, pad_value_attr, mode,
          nullptr);
}

void vpux::VPU::PadOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state,
                             vpux::NDTypeInterface& input_type, ::mlir::Value input, ::mlir::Value pads_begin,
                             ::mlir::Value pads_end, ::mlir::Value pad_value, ::mlir::ArrayAttr pads_begin_attr,
                             ::mlir::ArrayAttr pads_end_attr, ::mlir::FloatAttr pad_value_attr,
                             vpux::IE::PadMode mode) {
    build(builder, state, input_type, input, pads_begin, pads_end, pad_value, pads_begin_attr, pads_end_attr,
          pad_value_attr, mode, {});
}

//
// ClusteredOpInterface
//

bool vpux::VPU::PadOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t /*numTiles*/) {
    // Limit split strategy to axes which do NOT contain pads, to be aligned with how
    // i/o VPU.DistributedTensor split is computed
    VPUX_THROW_UNLESS(getPadsBeginAttr().has_value(), "Expecting padsBeginAttr to exist");
    VPUX_THROW_UNLESS(getPadsEndAttr().has_value(), "Expecting padsEndAttr to exist");
    const auto padsBegin = parseIntArrayAttr<int64_t>(getPadsBeginAttr().value());
    const auto padsEnd = parseIntArrayAttr<int64_t>(getPadsEndAttr().value());

    const auto noPadsOnDim{[&](auto dim) {
        return (padsBegin[dim] == 0) && (padsEnd[dim] == 0);
    }};

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return noPadsOnDim(Dims4D::Act::C.ind());
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return noPadsOnDim(Dims4D::Act::H.ind());
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return noPadsOnDim(Dims4D::Act::W.ind());
    }

    return false;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::PadOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                   distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
}

//
// SWOpInterface
//

bool vpux::VPU::PadOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "PadOp requires 1 input and 1 output, but the number of buffer is {0}",
                      buffers.size());

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

bool vpux::VPU::PadOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::PadOp::supportCycleCostCalculation() {
    return false;
}

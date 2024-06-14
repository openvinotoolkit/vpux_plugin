//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace VPU {

//
// Tiling
//

// Returns a WeightsTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getWeightsTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile) {
    const auto origWeightsTable = origOp->getWeightsTable();
    VPUX_THROW_UNLESS(origWeightsTable != nullptr, "The operation {0} doesn't have a WeightsTable", *origOp);

    const auto origWeightsTableShape = getShape(origWeightsTable);
    VPUX_THROW_UNLESS(origWeightsTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C] &&
                              origWeightsTableShape[Dim(1)] == 1 && origWeightsTableShape[Dim(2)] == 1 &&
                              origWeightsTableShape[Dim(3)] == VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                      "Unexpected WeightsTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origWeightsTableShape, getShape(origOp->getOutput()));

    // Each N-wise batch of the WeightsTable corresponds to its own output channel
    TileInfo weightsTableTile(origWeightsTableShape);
    weightsTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    weightsTableTile.shape[Dim(0)] = outputTile.shape[Dims4D::Act::C];
    return weightsTableTile;
}

// Returns an ActivationWindow tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getActivationWindowTile(ConcreteOp* origOp, const vpux::TileInfo& /*outputTile*/) {
    const auto origActivationWindow = origOp->getActivationWindow();
    VPUX_THROW_UNLESS(origActivationWindow != nullptr, "The operation {0} doesn't have an ActivationWindow", *origOp);

    const auto origActivationWindowShape = getShape(origActivationWindow);
    VPUX_THROW_UNLESS(origActivationWindowShape[Dim(0)] == 1 && origActivationWindowShape[Dim(1)] == 1 &&
                              origActivationWindowShape[Dim(2)] == 1,
                      "Unexpected ActivationWindow shape type or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origActivationWindowShape, getShape(origOp->getOutput()));

    // All output channels use the same only-one string in the table, so we just copy the whole thing
    return TileInfo(origActivationWindowShape);
}

// Returns an getInstructionListTableTile tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getInstructionListTableTile(ConcreteOp* origOp, const vpux::TileInfo& /*outputTile*/) {
    const auto origInstructionListTable = origOp->getInstructionListTable();
    VPUX_THROW_UNLESS(origInstructionListTable != nullptr, "The operation {0} doesn't have an InstructionListTable",
                      *origOp);

    const auto origInstructionListTableShape = getShape(origInstructionListTable);
    VPUX_THROW_UNLESS(origInstructionListTableShape[Dim(0)] == 1 && origInstructionListTableShape[Dim(1)] == 1 &&
                              origInstructionListTableShape[Dim(2)] == 1,
                      "Unexpected InstructionListTable shape type or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origInstructionListTableShape, getShape(origOp->getOutput()));

    // All output channels use the same only-one string in the table, so we just copy the whole thing
    return TileInfo(origInstructionListTableShape);
}

// Adjust paddings attributes for tiled input
template <typename ConcreteOp>
void adjustPaddings(ConcreteOp* op, const TilingInfo& inputTiling) {
    VPUX_THROW_UNLESS(inputTiling.pads.has_value(), "Missing tile information for paddings");

    auto newPadAttr = getPaddingAttr(op->getContext(), inputTiling.pads.value());

    op->setPadAttr(newPadAttr);
}

// Adjust rawFilterShape attribute for specific output tile
template <typename ConcreteOp>
void adjustRawFilterShape(ConcreteOp* op, const TileInfo& outputTile) {
    auto newRawFilterShape = Shape(parseIntArrayAttr<int64_t>(op->getRawFilterShape()));

    newRawFilterShape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    op->setRawFilterShapeAttr(getIntArrayAttr(op->getContext(), newRawFilterShape));
}

//
// Misc
//

bool isVFNCESupported(VPU::NCEOpInterface op);

mlir::LogicalResult sameLayout(VPU::DistributedTensorType inDistributedType,
                               VPU::DistributedTensorType outDistributedType, LogCb logCb = emptyLogCb);
mlir::LogicalResult sameLayout(VPUIP::DistributedBufferType inDistributedType,
                               VPUIP::DistributedBufferType outDistributedType, LogCb logCb = emptyLogCb);

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
bool arePerClusterMemoryShapeAndOffsetsEqual(T sourceType, T targetType) {
    // Ensure the memory view for the source and target distributions are the same,
    // no matter the attributes of the distribution.
    // For example, given:
    // sourceAttr = SEGMENTED across 2 clusters without uniformDistributedSegments
    // targetAttr = SEGMENTED across 2 clusters with uniformDistributedSegments
    // memory view will always be the same, so the distribution attrs are compatible.
    auto srcMemoryOffsets = sourceType.getPerClusterMemoryShapeOffsets();
    auto targetMemoryOffsets = targetType.getPerClusterMemoryShapeOffsets();

    auto srcMemoryShapes = sourceType.getPerClusterMemoryShapes();
    auto targetMemoryShapes = targetType.getPerClusterMemoryShapes();

    return (srcMemoryOffsets == targetMemoryOffsets) && (srcMemoryShapes == targetMemoryShapes);
}

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::LogicalResult areDistributionAttrsCompatible(T sourceType, T targetType,
                                                   const bool allowDifferentPerClusterMemoryView = false) {
    const auto sourceAttr = sourceType.getDistribution();
    const auto targetAttr = targetType.getDistribution();

    const auto inDistributionMode = sourceAttr.getMode().getValue();
    const auto outDistributionMode = targetAttr.getMode().getValue();

    if (inDistributionMode != outDistributionMode) {
        if (VPU::canTheDistributionModesBeCompatible(inDistributionMode, outDistributionMode).failed()) {
            return mlir::failure();
        }
    }

    const auto inDistributionNumClusters = sourceAttr.getNumClusters();
    const auto outDistributionNumClusters = targetAttr.getNumClusters();

    if (VPU::areDistributionNumClustersCompatible(inDistributionNumClusters, outDistributionNumClusters).failed()) {
        return mlir::failure();
    }

    if ((inDistributionMode == VPU::DistributionMode::SEGMENTED ||
         inDistributionMode == VPU::DistributionMode::OVERLAPPED) &&
        (outDistributionMode == VPU::DistributionMode::SEGMENTED ||
         outDistributionMode == VPU::DistributionMode::OVERLAPPED)) {
        const auto inDistributionNumTiles = parseIntArrayAttr<int64_t>(sourceAttr.getNumTiles());
        const auto outDistributionNumTiles = parseIntArrayAttr<int64_t>(targetAttr.getNumTiles());
        if (inDistributionNumTiles != outDistributionNumTiles) {
            return mlir::failure();
        }

        // When the source & target types are the types of an op's input & output, there is no generally applicable
        // way to verify the compatibility without having information about the op itself.
        // This util will indicate the types are compatible, with any extra checks having to be done at calling
        // location.
        if (allowDifferentPerClusterMemoryView) {
            return mlir::success();
        }

        // If source & target types are the type of a producer op's output and the type of a consumer op's input,
        // respectively, then as long as memory view is equal, the two distributed attributes are equivalent
        return arePerClusterMemoryShapeAndOffsetsEqual(sourceType, targetType) ? mlir::success() : mlir::failure();
    }

    return mlir::success();
}

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::LogicalResult isDistributedCastCompatible(T inDistributedType, T outDistributedType, LogCb logCb = emptyLogCb) {
    if (inDistributedType.getShape() != outDistributedType.getShape()) {
        logCb(formatv("Mismatch between shapes for input ({0}) and output ({1}).", inDistributedType.getShape(),
                      outDistributedType.getShape()));
        return mlir::failure();
    }

    if (areDistributionElementTypesCompatible(inDistributedType.getElementType(), outDistributedType.getElementType())
                .failed()) {
        logCb(formatv("Mismatch between element types for input ({0}) and output ({1}).",
                      inDistributedType.getElementType(), outDistributedType.getElementType()));
        return mlir::failure();
    }

    if (inDistributedType.getMemSpace() != outDistributedType.getMemSpace()) {
        logCb(formatv("Mismatch between memspaces for input ({0}) and output ({1}).", inDistributedType.getMemSpace(),
                      outDistributedType.getMemSpace()));
        return mlir::failure();
    }

    const auto sameLayoutCheck = sameLayout(inDistributedType, outDistributedType, logCb);
    if (sameLayoutCheck.failed()) {
        return mlir::failure();
    }

    if (areDistributionAttrsCompatible(inDistributedType, outDistributedType).failed()) {
        logCb(formatv("Mismatch between distributionAttr for input ({0}) and output ({1}).",
                      inDistributedType.getDistribution(), outDistributedType.getDistribution()));
        return mlir::failure();
    }

    return mlir::success();
}

template <typename T>
T vpux::VPU::NCEClusterTilingOp::getInnerTaskOpOfType() {
    return mlir::dyn_cast<T>(&getBody().front().front());
}
}  // namespace VPU
}  // namespace vpux

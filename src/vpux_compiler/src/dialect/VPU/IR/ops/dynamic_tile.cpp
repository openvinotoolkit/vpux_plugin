//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DynamicTileOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DynamicTileOpAdaptor tile(operands, attrs, prop);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(tile.getInput().getType());

    const auto outShape = parseIntArrayAttr<int64_t>(tile.getOutputShapeAttr());
    const auto outBounds = parseIntArrayAttr<int64_t>(tile.getOutputBoundsAttr());

    auto outType = inType.changeShape(Shape(outShape));

    inferredReturnTypes.push_back(
            outType.cast<vpux::BoundedTypeInterface>().changeBounds(getIntArrayAttr(ctx, outBounds)));

    return mlir::success();
}

//
// fit into CMX
//

bool vpux::VPU::DynamicTileOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3,
                      "DynamicTileOp requires 2 inputs and 1 output, but the number of buffers is {0}", buffers.size());

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

bool vpux::VPU::DynamicTileOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DynamicTileOp::supportCycleCostCalculation() {
    return false;
}

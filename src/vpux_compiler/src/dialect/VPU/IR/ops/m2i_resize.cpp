//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::M2IResizeOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output,
                                        Byte reservedMem) {
    auto totalAvailableCMXSize =
            reservedMem.count() == 0 ? getTotalCMXSize(op).count() : getTotalCMXFragmentationAwareSize(op).count();
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(op), buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::M2IResizeOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input,
                                        vpux::NDTypeInterface output) {
    return fitIntoCMX(op, input, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::M2IResizeOp::isSupported(IE::InterpolateOp op, LogCb logCb, bool /*checkLayout*/,
                                         bool /*checkChannelAlignment*/) {
    const auto inType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outType = op.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (!fitIntoCMX(op, inType, outType)) {
        logCb(llvm::formatv("Op doesn't fit into CMX memory"));
        return false;
    }

    return VPU::isM2IResizeSupported<IE::InterpolateOp>(op, logCb, true /*checkFp16Interleaved*/);
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::M2IResizeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    M2IResizeOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    // Note: limited to 'shape_calculation_mode = sizes'
    const auto outSize = parseIntArrayAttr<int64_t>(op.getSizes());
    const auto outAxes = parseIntArrayAttr<int64_t>(op.getAxes());

    SmallVector<int64_t> outShape;

    for (size_t i = 0; i < inShape.size(); i++) {
        outShape.emplace_back(inShape[i]);
    }

    // Patch dims
    if (outSize.size() != outAxes.size()) {
        VPUX_THROW("Sizes and Axes vectors must have same size !");
    }
    for (size_t i = 0; i < outAxes.size(); i++) {
        outShape[outAxes[i]] = outSize[i];
    }

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

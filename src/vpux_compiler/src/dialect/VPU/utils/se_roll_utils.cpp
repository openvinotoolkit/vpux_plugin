//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace VPU;

namespace {
bool isSupportedSEPRollImpl(VPU::ArchKind arch, NDTypeInterface inputType, NDTypeInterface outputType,
                            ArrayRef<int64_t> axes, mlir::MLIRContext* ctx, LogCb logCb, bool checkLayout,
                            bool /*checkChannelAlignment*/, bool supportsInputActCompression) {
    const auto inputShape = inputType.getShape();
    if (inputShape.size() != 4 || outputType.getRank() != 4) {
        logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputShape.size()));
        return false;
    }

    if (axes.size() != 2) {
        logCb(formatv("{0} dimensions to roll", axes.size()));
        return false;
    }
    if (axes[0] != Dims4D::Act::H.ind() || axes[1] != Dims4D::Act::W.ind()) {
        logCb(formatv("it's not spatial rolling"));
        return false;
    }

    const int64_t KY = 1;
    const int64_t KX = 1;

    auto weightShape = Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::C], KY, KX});
    mlir::Type elemType = mlir::Float16Type::get(ctx);
    if (mlir::isa<mlir::quant::QuantizedType>(inputType.getElementType())) {
        elemType = mlir::quant::UniformQuantizedType::get(
                /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                /*scale=*/static_cast<double>(1.0f), /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    }
    const auto tensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
    const auto weightsType =
            mlir::RankedTensorType::get(weightShape.raw(), elemType, tensorAttr).cast<vpux::NDTypeInterface>();

    const int64_t SY = 1;
    const int64_t SX = 1;

    PadInfo pads(0, 0, 0, 0);
    const auto dilations = SmallVector<int64_t>{1, 1};

    return VPU::isNCEConvSupported(arch, inputType, weightsType, outputType, dilations, KY, KX, SY, SX, pads,
                                   checkLayout, /*checkChannelAlignment*/ true, logCb, supportsInputActCompression);
}

}  // namespace

bool VPU::isSupportedSEPRoll(IE::RollOp op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                             bool supportsInputActCompression) {
    const auto inputType = op.getData().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();

    const auto inputShape = inputType.getShape();
    auto shiftAndAxesOrFail = IE::getShiftAndAxesForRollOp(op.getLoc(), op.getShift(), op.getAxes(), inputShape);
    if (mlir::failed(shiftAndAxesOrFail)) {
        return false;
    }
    const auto shiftAndAxes = shiftAndAxesOrFail.value();
    const auto axes = shiftAndAxes.axes;

    return isSupportedSEPRollImpl(getArch(op), inputType, outputType, axes, op.getContext(), logCb, checkLayout,
                                  checkChannelAlignment, supportsInputActCompression);
}

bool VPU::isSupportedSEPRoll(VPU::RollOp op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                             bool supportsInputActCompression) {
    const auto inputType = op.getData().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();

    const auto inputShape = inputType.getShape();
    auto shiftAndAxesOrFail = IE::getShiftAndAxesForRollOp(op.getLoc(), op.getShift(), op.getAxes(), inputShape);
    if (mlir::failed(shiftAndAxesOrFail)) {
        return false;
    }
    const auto shiftAndAxes = shiftAndAxesOrFail.value();
    const auto axes = shiftAndAxes.axes;

    return isSupportedSEPRollImpl(getArch(op), inputType, outputType, axes, op.getContext(), logCb, checkLayout,
                                  checkChannelAlignment, supportsInputActCompression);
}

DimArr VPU::getRollSEPConvTilingOrder(VPU::SERollAttr seAttr) {
    const auto shift = parseIntArrayAttr<int64_t>(seAttr.getShift());
    if (shift[SE_ROLL_SPATIAL_H] != 0 && shift[SE_ROLL_SPATIAL_W] != 0) {
        return SmallVector<Dim>{Dims4D::Act::C};
    } else if (shift[SE_ROLL_SPATIAL_H] != 0) {
        return DimArr{Dims4D::Act::W, Dims4D::Act::C};
    } else {
        return DimArr{Dims4D::Act::H, Dims4D::Act::C};
    }
}

bool VPU::isRollSEPConvCompatibleWithClusterStrategy(VPU::SERollAttr seAttr, VPU::MultiClusterStrategy strategy) {
    const auto shift = parseIntArrayAttr<int64_t>(seAttr.getShift());
    if (shift[SE_ROLL_SPATIAL_H] != 0 && (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
                                          strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
                                          strategy == VPU::MultiClusterStrategy::SplitOverHeightKernel ||
                                          strategy == VPU::MultiClusterStrategy::SplitOverHeightWidth ||
                                          strategy == VPU::MultiClusterStrategy::HKSwitch)) {
        return false;
    }
    if (shift[SE_ROLL_SPATIAL_W] != 0 && (strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
                                          strategy == VPU::MultiClusterStrategy::SplitOverHeightWidth)) {
        return false;
    }

    if (shift[SE_ROLL_SPATIAL_H] != 0 && shift[SE_ROLL_SPATIAL_W] != 0) {
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverKernel;
    }
    return true;
}

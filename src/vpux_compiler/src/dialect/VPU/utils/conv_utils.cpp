//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isNCEConvSupported(VPU::ArchKind arch, NDTypeInterface inputType, NDTypeInterface filterType,
                                   NDTypeInterface outputType, ArrayRef<int64_t> dilations, int64_t KY, int64_t KX,
                                   int64_t SY, int64_t SX, PadInfo pads, bool checkLayout, bool checkChannelAlignment,
                                   LogCb logCb, bool supportsInputActCompression) {
    if (outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (dilations.size() != 2) {
        logCb(formatv("Expected dilations size to be 2, got '{0}'", dilations.size()));
        return false;
    }
    if (dilations[0] != 1 || dilations[1] != 1) {
        logCb(formatv("Dilated convolution is not supported"));
        return false;
    }

    if (!NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right, logCb)) {
        return false;
    }

    const auto inputOrder = inputType.getDimsOrder();
    const auto isChannelMajor = inputOrder == DimsOrder::NCHW;
    const auto inputChannelAlignment =
            !isChannelMajor ? VPU::NCEConvolutionOp::getInputChannelAlignmentImpl(inputType) : 1;

    if (checkChannelAlignment) {
        if (!NCEInvariant::isInputActTypeSupported(arch, inputType, inputChannelAlignment,
                                                   supportsInputActCompression) ||
            !NCEInvariant::isOutputActTypeSupported(outputType,
                                                    VPU::NCEConvolutionOp::getOutputChannelAlignmentImpl(outputType))) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        const auto filterOrder = filterType.getDimsOrder();
        const auto outputOrder = outputType.getDimsOrder();

        if (inputOrder != DimsOrder::NHWC && inputOrder != DimsOrder::NCHW) {
            logCb(formatv("Unsupported input layout '{0}'", inputOrder));
            return false;
        }
        if (filterOrder != DimsOrder::OYXI) {
            logCb(formatv("Unsupported filter layout '{0}'", filterOrder));
            return false;
        }
        const std::set<VPU::ArchKind> compatibleTargets = {
                VPU::ArchKind::NPU37XX,
                VPU::ArchKind::NPU40XX,
        };
        if (compatibleTargets.count(arch) <= 0 && outputOrder != DimsOrder::NHWC) {
            logCb(formatv("Unsupported output layout '{0}'", outputOrder));
            return false;
        }
    }

    return true;
}

bool vpux::VPU::isSupportedConv(IE::ConvolutionOp op, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                                bool supportsInputActCompression) {
    const auto arch = getArch(op);

    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());

    const auto filterShape = getShape(op.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto pads = PadInfo(op.getPadsBegin(), op.getPadsEnd());

    const auto inputType = op.getInput().getType().cast<NDTypeInterface>();
    const auto filterType = op.getFilter().getType().cast<NDTypeInterface>();
    const auto outputType = op.getOutput().getType().cast<NDTypeInterface>();

    return VPU::isNCEConvSupported(arch, inputType, filterType, outputType, dilations, KY, KX, SY, SX, pads,
                                   checkLayout, checkChannelAlignment, logCb, supportsInputActCompression);
}

namespace {

bool isFilterConst(mlir::Value filter) {
    // While adjusting the layout, an intermediate Reorder operation can be introduced, before it gets fused into the
    // filter constant
    if (auto reorderOp = filter.getDefiningOp<IE::ReorderOp>()) {
        filter = reorderOp.getInput();
    }

    auto constOp = filter.getDefiningOp<Const::DeclareOp>();
    if (auto fqOp = filter.getDefiningOp<IE::FakeQuantizeOp>()) {
        constOp = fqOp.getInput().getDefiningOp<Const::DeclareOp>();
    }

    if (auto dequantOp = filter.getDefiningOp<IE::DequantizeOp>()) {
        constOp = dequantOp.getInput().getDefiningOp<Const::DeclareOp>();
    }

    return constOp != nullptr;
}

bool isSupportedSEPTransposedConvImpl(VPU::ArchKind arch, NDTypeInterface inputType, NDTypeInterface filterType,
                                      NDTypeInterface outputType, mlir::ArrayAttr kernelStridesAttr,
                                      mlir::ArrayAttr dilationsAttr, mlir::ArrayAttr padsBeginAttr,
                                      mlir::ArrayAttr padsEndAttr, mlir::ArrayAttr outputPaddingAttr, LogCb logCb,
                                      bool checkLayout, bool checkChannelAlignment, bool supportsInputActCompression) {
    const auto dilations = parseIntArrayAttr<int64_t>(dilationsAttr);
    if (dilations[Dims4D::Dilation::X.ind()] > 1 || dilations[Dims4D::Dilation::Y.ind()] > 1) {
        logCb(formatv("Dilated transposed convolution is not supported"));
        return false;
    }

    const auto origKernelStrides = Shape(parseIntArrayAttr<int64_t>(kernelStridesAttr));
    const auto stridesY = origKernelStrides[Dims4D::Strides::Y];
    const auto stridesX = origKernelStrides[Dims4D::Strides::X];
    const auto origPads = PadInfo(padsBeginAttr, padsEndAttr);
    if (origPads.left > (stridesX - 1) || origPads.top > (stridesY - 1) || origPads.right > (stridesX - 1) ||
        origPads.bottom > (stridesY - 1)) {
        logCb(formatv("Padding larger than strides are not supported"));
        return false;
    }

    const auto filterShape = filterType.getShape().raw();
    const auto KY = filterShape[filterShape.size() - 2];
    const auto KX = filterShape[filterShape.size() - 1];

    const auto outputPadding = Shape(parseIntArrayAttr<int64_t>(outputPaddingAttr));

    const auto inputShape = inputType.getShape();
    const auto zerosY = origKernelStrides[Dims4D::Strides::Y] - 1;
    const auto zerosX = origKernelStrides[Dims4D::Strides::X] - 1;
    const auto newPadTop = KY - 1;
    const auto newPadBottom = KY - 1 + outputPadding[Dims4D::PadsOutput::Y];
    const auto newPadLeft = KX - 1;
    const auto newPadRight = KX - 1 + outputPadding[Dims4D::PadsOutput::X];
    const auto newY = inputShape[Dims4D::Act::H] + zerosY * (inputShape[Dims4D::Act::H] - 1) + newPadTop + newPadBottom;
    const auto newX = inputShape[Dims4D::Act::W] + zerosX * (inputShape[Dims4D::Act::W] - 1) + newPadLeft + newPadRight;
    const Shape newInputShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], newY, newX};
    inputType = inputType.changeShape(newInputShape);

    const int64_t SY = 1;
    const int64_t SX = 1;

    PadInfo pads(0, 0, 0, 0);

    return VPU::isNCEConvSupported(arch, inputType, filterType, outputType, dilations, KY, KX, SY, SX, pads,
                                   checkLayout, checkChannelAlignment, logCb, supportsInputActCompression);
}

}  // namespace

bool VPU::isSupportedSEPTransposedConv(IE::TransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                       bool checkChannelAlignment, bool supportsInputActCompression) {
    if (!isFilterConst(op.getFilter())) {
        logCb(formatv("The filter is not a constant"));
        return false;
    }
    auto inputType = op.getInput().getType().cast<NDTypeInterface>();
    auto filterType = op.getFilter().getType().cast<NDTypeInterface>();
    auto outputType = op.getOutput().getType().cast<NDTypeInterface>();
    if (inputType.getShape().size() != 4) {
        logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
        return false;
    }
    if (filterType.getShape().size() != 4) {
        logCb(formatv("Only 4D filters are supported, got {0} dimensions", filterType.getShape().size()));
        return false;
    }
    if (outputType.getShape().size() != 4) {
        logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
        return false;
    }
    return isSupportedSEPTransposedConvImpl(getArch(op), inputType, filterType, outputType, op.getStrides(),
                                            op.getDilations(), op.getPadsBegin(), op.getPadsEnd(),
                                            op.getOutputPadding(), logCb, checkLayout, checkChannelAlignment,
                                            supportsInputActCompression);
}

bool VPU::isSupportedSEPTransposedConv(IE::GroupTransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                       bool checkChannelAlignment, bool supportsInputActCompression) {
    if (!isFilterConst(op.getFilter())) {
        return false;
    }
    auto inputType = op.getInput().getType().cast<NDTypeInterface>();
    auto filterType = op.getFilter().getType().cast<NDTypeInterface>();
    auto outputType = op.getOutput().getType().cast<NDTypeInterface>();
    if (inputType.getShape().size() != 4) {
        logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
        return false;
    }
    if (filterType.getShape().size() != 5) {
        logCb(formatv("Only 5D filters are supported, got {0} dimensions", filterType.getShape().size()));
        return false;
    }
    if (outputType.getShape().size() != 4) {
        logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
        return false;
    }
    return isSupportedSEPTransposedConvImpl(getArch(op), inputType, filterType, outputType, op.getStrides(),
                                            op.getDilations(), op.getPadsBegin(), op.getPadsEnd(),
                                            op.getOutputPadding(), logCb, checkLayout, checkChannelAlignment,
                                            supportsInputActCompression);
}

bool VPU::isSupportedSEPTransposedConv(VPU::TransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                       bool checkChannelAlignment, bool supportsInputActCompression) {
    if (!isFilterConst(op.getFilter())) {
        return false;
    }
    auto inputType = op.getInput().getType().cast<NDTypeInterface>();
    auto filterType = op.getFilter().getType().cast<NDTypeInterface>();
    auto outputType = op.getOutput().getType().cast<NDTypeInterface>();
    if (inputType.getShape().size() != 4) {
        logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
        return false;
    }
    if (filterType.getShape().size() != 4) {
        logCb(formatv("Only 4D filters are supported, got {0} dimensions", filterType.getShape().size()));
        return false;
    }
    if (outputType.getShape().size() != 4) {
        logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
        return false;
    }
    return isSupportedSEPTransposedConvImpl(getArch(op), inputType, filterType, outputType, op.getStrides(),
                                            op.getDilations(), op.getPadsBegin(), op.getPadsEnd(),
                                            op.getOutputPadding(), logCb, checkLayout, checkChannelAlignment,
                                            supportsInputActCompression);
}

std::optional<bool> VPU::isSEPConvCompatibleWithClusterStrategy(VPU::NCEConvolutionOp nceConv,
                                                                VPU::MultiClusterStrategy strategy) {
    auto sparseInput = nceConv.getInput().getType().dyn_cast<VPU::SparseTensorType>();
    if (sparseInput == nullptr) {
        return std::nullopt;
    }

    auto seAttr = sparseInput.getSeAttr().dyn_cast_or_null<VPU::SERollAttr>();
    if (seAttr != nullptr) {
        return VPU::isRollSEPConvCompatibleWithClusterStrategy(seAttr, strategy);
    }
    return std::nullopt;
}

mlir::LogicalResult vpux::VPU::verifyConvUtil(mlir::Location loc, VPU::ArchKind arch, Shape filterShape,
                                              Shape kernelStrides, PaddingAttr padAttr, ShapeRef weightsTableShape,
                                              mlir::Value output) {
    const auto logCb = [loc](const formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    const auto outputShape = getShape(output);
    const auto OC = outputShape[Dims4D::Act::C];

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = padAttr.getTop().getValue().getSExtValue();
    const auto padBottom = padAttr.getBottom().getValue().getSExtValue();
    const auto padLeft = padAttr.getLeft().getValue().getSExtValue();
    const auto padRight = padAttr.getRight().getValue().getSExtValue();

    if (!VPU::NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    const auto expectedWeightsTableShape = VPU::NCESparsity::inferWeightsTableShape(OC);

    if (weightsTableShape != expectedWeightsTableShape) {
        return errorAt(loc, "Got wrong shape for 'weightsTable' '{0}', expected '{1}'", weightsTableShape,
                       expectedWeightsTableShape);
    }

    return mlir::success();
}

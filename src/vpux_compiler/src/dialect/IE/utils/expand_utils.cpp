//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/utils/core/logger.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <numeric>

namespace vpux {
namespace IE {

// With respect to eltwise ops in a chain, for example:
//   Expand -> Add -> Slice -> Expand -> Add -> Slice
// It will be beneficial to keep the 2nd Expand for the 2nd Add instead of folding with Slice.
// So that the 2nd Add can utilize AdjustInputShapeForEltwise pass
bool beneficialToKeepExpand(ShapeRef unExpandedShape, ShapeRef expandedShape, mlir::Operation* childOp) {
    if (!childOp->hasOneUse()) {
        return false;
    }
    const auto isEltwiseOp = [](mlir::Operation* op) {
        if (op == nullptr) {
            return false;
        }
        // Mul/Sub/Add are selected since they are covered by the AdjustInputShapeForEltwise pass
        if (auto grpConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op)) {
            return groupConvIsEltwise(grpConvOp);
        } else if (mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp>(op)) {
            return true;
        }
        return false;
    };

    vpux::Logger log("beneficialToKeepExpand", vpux::LogLevel::Info);
    while (isEltwiseOp(childOp) && VPU::NCEInvariant::isSupported(childOp).succeeded()) {
        auto shapeCastResult = getShapeCastExpandedShape(childOp, expandedShape, unExpandedShape, log);
        if (mlir::failed(shapeCastResult)) {
            return false;
        }
        auto sliceChildOp = mlir::dyn_cast_or_null<IE::SliceOp>(*childOp->getResult(0).getUsers().begin());
        if (sliceChildOp == nullptr) {
            return true;
        }
        auto expandChildOp = mlir::dyn_cast_or_null<IE::ExpandOp>(*sliceChildOp->getResult(0).getUsers().begin());
        if (expandChildOp == nullptr) {
            return true;
        }
        childOp = *childOp->getResult(0).getUsers().begin();
        if (childOp == nullptr) {
            return true;
        } else if (!childOp->hasOneUse()) {
            return false;
        }
    }
    return false;
}

int64_t calculateAlignmentRequirementForExpandOpConversion(const vpux::NDTypeInterface expandInType) {
    const auto channelAlignment = VPU::NCEInvariant::getAlignment(expandInType.getElementType());
    const auto expandInChannels = expandInType.getShape()[Dims4D::Act::C];
    const auto leastChannelMultiple = std::lcm(channelAlignment, expandInChannels);
    return leastChannelMultiple / expandInChannels;
}

bool beneficialToPadHeight(IE::ExpandOp origOp) {
    const auto expandInType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto expandInShape = expandInType.getShape();
    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);

    // TODO (E#128867): need to repeat the experiment of E#118379 and E#120473 for the unaligned case (W%16 != 0), to
    // understand the correct tradeoff between expand to conv solution and dma solution (expected to be different due to
    // the overhead of padding) and correctly set up the constraints. Current constraints are only avoiding regressions
    // manifesting in CI, but they should be formally studied through measurements.
    if (expandInShape[Dims4D::Act::W] % convolutionAlignment != 0 &&
        ((expandInShape[Dims4D::Act::H] <= 48) || (expandInShape[Dims4D::Act::H] >= 1440) ||
         (expandInShape[Dims4D::Act::W] <= 128))) {
        return false;
    }

    return true;
}

bool beneficialToPadWidth(IE::ExpandOp origOp) {
    const auto expandInType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto expandInShape = expandInType.getShape();
    const auto expandInW = expandInShape[Dims4D::Act::W];
    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);

    //
    // Padding W to make Expand could convert to Conv
    // For example:
    //            Expand         :    1x1x32x79741 -> 1x16x32x79741
    //              |
    //            Conv
    // Convert to:
    //            Expand         :    1x1x32x79741 -> 1x1x32x79744
    //              |
    //            AffineReshape  :    1x1x32x79744 -> 1x16x32x4984xf16
    //              |
    //            New Conv
    //              |
    //            AffineReshape
    //              |
    //            Conv
    //
    if (expandInW % convolutionAlignment == 0) {
        return false;
    }

    if (!origOp.getResult().hasOneUse()) {
        return false;
    }

    mlir::Operation* userOp = *origOp.getResult().getUsers().begin();
    if (!mlir::isa<IE::ConvolutionOp, IE::TransposedConvolutionOp, IE::MaxPoolOp, IE::AvgPoolOp>(userOp)) {
        return false;
    }

    SmallVector<int64_t> padsBegin = {0, 0};
    SmallVector<int64_t> padsEnd = {0, 0};
    SmallVector<int64_t> stridesData = {1, 1};
    llvm::TypeSwitch<mlir::Operation*, void>(userOp)
            .Case<IE::ConvolutionOp, IE::TransposedConvolutionOp, IE::MaxPoolOp, IE::AvgPoolOp>([&](auto userOp) {
                padsBegin = parseIntArrayAttr<int64_t>(userOp.getPadsBeginAttr());
                padsEnd = parseIntArrayAttr<int64_t>(userOp.getPadsEndAttr());
                stridesData = parseIntArrayAttr<int64_t>(userOp.getStrides());
            });

    if (padsBegin.size() != 2 || padsBegin.size() != 2) {
        return false;
    }

    // Beneficial to pad W for case that the new width / strideX is same as original width then sliceOp is not required
    const auto alignedExpandInW = alignValUp(expandInShape[Dims4D::Act::W], convolutionAlignment);
    const auto inWWithPad =
            expandInW + padsBegin[Dims4D::PadsBegin::Left.ind()] + padsEnd[Dims4D::PadsEnd::Right.ind()];
    const auto alignedInWWithPad =
            alignedExpandInW + padsBegin[Dims4D::PadsBegin::Left.ind()] + padsEnd[Dims4D::PadsEnd::Right.ind()];
    return inWWithPad / stridesData[Dims4D::Strides::X.ind()] ==
           alignedInWWithPad / stridesData[Dims4D::Strides::X.ind()];
}

bool isEligibleConvertToConv(IE::ExpandOp expandOp, Logger log, StringRef debugName) {
    const auto expandInType = expandOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto expandOutType = expandOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto supportedLayout = DimsOrder::NHWC;
    const auto expandInLayout = expandInType.getDimsOrder();
    if (expandInLayout != supportedLayout) {
        log.trace("[{0}]: Expand at {1} has {2} input layout, expected {3}", debugName, expandOp.getLoc(),
                  expandInLayout, supportedLayout);
        return false;
    }
    const auto expandOutLayout = expandOutType.getDimsOrder();
    if (expandOutLayout != supportedLayout) {
        log.trace("[{0}]: Expand at {1} has {2} output layout, expected {3}", debugName, expandOp.getLoc(),
                  expandOutLayout, supportedLayout);
        return false;
    }
    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBeginAttr());
    if (expandPadsBegin.size() != 4) {
        log.trace("[{0}]: Expand at {1} has {2}-d start padding. Only 4-d shapes are supported", debugName,
                  expandOp.getLoc(), expandPadsBegin.size());
        return false;
    }
    const auto isConflictingPadBegin = [](const int64_t pad) -> bool {
        return pad != 0;
    };
    if (std::any_of(expandPadsBegin.begin(), expandPadsBegin.end(), isConflictingPadBegin)) {
        log.trace("[{0}]: Expand at {1} has {2} start padding. Expected to have [0, 0, 0, 0]", debugName,
                  expandOp.getLoc(), expandPadsBegin);
        return false;
    }
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEndAttr());
    if (expandPadsEnd.size() != 4) {
        log.trace("[{0}]: Expand at {1} has {2}-d end padding. Only 4-d shapes are supported", debugName,
                  expandOp.getLoc(), expandPadsEnd.size());
        return false;
    }
    if (expandPadsEnd[Dims4D::Act::N.ind()] != 0 || expandPadsEnd[Dims4D::Act::C.ind()] <= 0 ||
        expandPadsEnd[Dims4D::Act::H.ind()] != 0 || expandPadsEnd[Dims4D::Act::W.ind()] != 0) {
        log.trace("[{0}]: Expand at {1} has {2} end padding. Expected to have [0, C, 0, 0]", debugName,
                  expandOp.getLoc(), expandPadsEnd);
        return false;
    }
    const auto expandInShape = expandInType.getShape();
    if (expandInShape.size() != 4) {
        log.trace("[{0}]: Expand at {1} has {2}-d shape. Only 4-d shapes are supported", debugName, expandOp.getLoc(),
                  expandInShape.size());
        return false;
    }
    if (expandInShape[Dims4D::Act::N] != 1) {
        log.trace("[{0}]: Expand at {1} has batch {2}. Expected to have 1", debugName, expandOp.getLoc(),
                  expandInShape[Dims4D::Act::N]);
        return false;
    }
    if (!expandInType.getElementType().isF16() && !expandInType.getElementType().isa<mlir::quant::QuantizedType>()) {
        log.trace("[{0}]: Expand at {1} has {2} element type. Only float16 and quantized types are supported",
                  debugName, expandOp.getLoc(), expandInType.getElementType());
        return false;
    }

    // There are two conversion methods for Expand Op
    // 1. Convert to Convolution
    // 2. Convert to DMA
    // Experimental data shows the inference time is related to the channel-size and dimC/dimH(dimC/dimW).
    // Experimental Constraint E#118379:
    //    Utilize DMA conversion when channel size exceeds 32 (since expand with small channel will cause lots of stride
    //    DMAs, convert to conv is more efficient) and ratioH/ratioW exceeds 0.7 (since if channel is much smaller than
    //    height/wight also cause lots of stride DMAs).
    auto ratioH = static_cast<float>(expandInShape[Dims4D::Act::C]) / expandInShape[Dims4D::Act::H];
    auto ratioW = static_cast<float>(expandInShape[Dims4D::Act::C]) / expandInShape[Dims4D::Act::W];
    if (expandInShape[Dims4D::Act::C] > 32 && (ratioH > 0.7 || ratioW > 0.7)) {
        log.trace("[{0}]: Expansion at {1} has {2} channels exceeds '32', and spatial size (H {3} * W {4})."
                  "Converting to convolution is not beneficial",
                  debugName, expandOp.getLoc(), expandInShape[Dims4D::Act::C], expandInShape[Dims4D::Act::H],
                  expandInShape[Dims4D::Act::W]);
        return false;
    }

    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);
    if (!beneficialToPadHeight(expandOp) && !beneficialToPadWidth(expandOp)) {
        log.trace("[{0}]: Expand at {1} has width {2} not multiple of {3}. Expand to conv only for case "
                  "beneficialToPadHeight or beneficialToPadWidth",
                  debugName, expandOp.getLoc(), expandInShape[Dims4D::Act::W], convolutionAlignment);
        return false;
    }

    return true;
}

std::optional<vpux::Dim> getExpandAxis(IE::ExpandOp expandOp) {
    const auto expandAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(expandOp.getInput()), getShape(expandOp.getResult()));
    if (expandAxes.empty() || expandAxes.size() != 1) {
        return std::nullopt;
    }
    return expandAxes.front();
}

}  // namespace IE
}  // namespace vpux

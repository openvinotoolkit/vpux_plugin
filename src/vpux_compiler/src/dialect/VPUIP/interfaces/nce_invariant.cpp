//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Operation.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    auto loc = origOp->getLoc();
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();

    if (filterType.getRank() != 4) {
        log.trace("[{0}] Filter has unsupported rank: {1}", loc, filterType.getRank());
        return mlir::failure();
    }

    const auto filterShape = filterType.getShape();

    const auto OC = filterShape[Dims4D::Filter::OC];
    auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    const auto outAlignment = channelsInfo.getOutputChannelAlignment();
    if (OC % outAlignment != 0) {
        log.trace("[{0}] Convolution output channels are not aligned", loc);
        return mlir::failure();
    }

    if (inputType.getDimsOrder() == DimsOrder::NHWC) {
        const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
        const auto IC = filterShape[Dims4D::Filter::IC];
        if (arch == VPU::ArchKind::NPU37XX || arch == VPU::ArchKind::NPU40XX) {
            if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation())) {
                if (IC % iface.getInputChannelAlignment() == 0) {
                    return mlir::success();
                }
            }
        }
        if (IC % VPU::NCEInvariant::getAlignment(inputType.getElementType()) != 0) {
            log.trace("[{0}] ZMajor Convolution input channels are not aligned", loc);
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEConvolutionOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCECompressConvolutionOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

//
// verifyPoolChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPoolChannels(mlir::Location loc, vpux::NDTypeInterface inputType,
                                                                  Logger log) {
    log.setName("NCEInvariant");

    if (inputType.getRank() != 4) {
        log.trace("[{0}] Input has unsupported rank: {1}", loc, inputType.getRank());
        return mlir::failure();
    }

    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    if (IC % VPU::NCEInvariant::getAlignment(inputType.getElementType()) != 0) {
        log.trace("[{0}] Pooling channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::MaxPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.getInput().getType().cast<vpux::NDTypeInterface>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEMaxPoolOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::AvgPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.getInput().getType().cast<vpux::NDTypeInterface>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEAveragePoolOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

//
// verifyReduceChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyReduceChannels(mlir::Location loc, vpux::NDTypeInterface inputType,
                                                                    Logger log) {
    log.setName("NCEInvariant");
    if (inputType.getRank() != 4) {
        log.trace("[{0}] Reduce input shape does not have 4 dimensions. Not supported.", loc);
        return mlir::failure();
    }

    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    if (IC % VPU::NCEInvariant::getAlignment(inputType.getElementType()) != 0) {
        log.trace("[{0}] Reduce input channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyEltwiseChannels(mlir::Location loc,
                                                                     vpux::NDTypeInterface firstInputType,
                                                                     vpux::NDTypeInterface secondInputType,
                                                                     Logger log) {
    log.setName("NCEInvariant");
    if (firstInputType.getRank() != 4) {
        log.trace("[{0}] Eltwise input1 shape does not have 4 dimensions. Not supported.", loc);
        return mlir::failure();
    }

    if (secondInputType.getRank() != 4) {
        log.trace("[{0}] Eltwise input2 shape does not have 4 dimensions. Not supported.", loc);
        return mlir::failure();
    }

    const auto firstInputShape = firstInputType.getShape();
    const auto secondInputShape = secondInputType.getShape();
    const auto firstIC = firstInputShape[Dims4D::Act::C];
    const auto secondIC = secondInputShape[Dims4D::Act::C];

    if (firstIC % VPU::NCEInvariant::getAlignment(firstInputType.getElementType()) != 0) {
        log.trace("[{0}] Eltwise input1 channels are not aligned", loc);
        return mlir::failure();
    }

    if (secondIC % VPU::NCEInvariant::getAlignment(secondInputType.getElementType()) != 0) {
        log.trace("[{0}] Eltwise input2 channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::AddOp origOp, Logger log) {
    auto input1Type = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.getInput2().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::MultiplyOp origOp, Logger log) {
    auto input1Type = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.getInput2().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::SubtractOp origOp, Logger log) {
    auto input1Type = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.getInput2().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::ReduceMeanOp origOp, Logger log) {
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    return verifyReduceChannels(origOp->getLoc(), inputType, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::AndOp origOp, Logger log) {
    auto input1Type = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.getInput2().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseChannels(origOp->getLoc(), input1Type, input2Type, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEEltwiseOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

//
// verifyGroupConvChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyGroupConvChannels(mlir::Location loc,
                                                                       vpux::NDTypeInterface inputType,
                                                                       vpux::NDTypeInterface filterType,
                                                                       IE::AlignedChannelsOpInterface channelsIface,
                                                                       Logger log) {
    log.setName("NCEInvariant");

    if (inputType.getRank() != 4) {
        log.trace("[{0}] Input has unsupported rank: {1}", loc, inputType.getRank());
        return mlir::failure();
    }

    if (filterType.getRank() != 4) {
        log.trace("[{0}] Filter has unsupported rank: {1}", loc, filterType.getRank());
        return mlir::failure();
    }

    const auto filterShape = filterType.getShape();
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    if (filtersPerInChan != 1) {
        log.trace("[{0}] Group Convolution with more than one filter per channel is not supported", loc);
        return mlir::failure();
    }

    const auto inputShape = inputType.getShape();
    const auto inputChan = inputShape[Dims4D::Act::C];
    const auto OC = filterShape[Dims4D::Filter::OC];
    if (OC != inputChan) {
        log.trace("[{0}] Group Convolution has {1} groups, expected {2}", loc, OC, inputChan);
        return mlir::failure();
    }

    if (OC % channelsIface.getOutputChannelAlignment() != 0) {
        log.trace("[{0}] Group Convolution output channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::GroupConvolutionOp origOp, Logger log) {
    return verifyGroupConvChannels(origOp->getLoc(), origOp.getInput().getType().cast<vpux::NDTypeInterface>(),
                                   origOp.getFilter().getType().cast<vpux::NDTypeInterface>(),
                                   mlir::cast<IE::AlignedChannelsOpInterface>(origOp.getOperation()), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEDepthConvolutionOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEPermuteOp, Logger) {
    // VPU.NCE operation guarantees that invariant satisifies channel constraints
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::InterpolateOp origOp, Logger log) {
    log.setName("NCEInvariant");

    auto loc = origOp->getLoc();
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    const auto IC = inputShape[Dims4D::Act::C];
    const auto OC = outputShape[Dims4D::Act::C];
    if (IC % VPU::NCEInvariant::getAlignment(inputType.getElementType()) != 0) {
        log.trace("[{0}] Interpolate input channels '{1}' are not aligned", loc, IC);
        return mlir::failure();
    }
    auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    const auto outAlignment = channelsInfo.getOutputChannelAlignment();
    if (OC % outAlignment != 0) {
        log.trace("[{0}] Interpolate output channels '{1}' are not aligned", loc, OC);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(VPU::NCEInterpolateOp, Logger) {
    // VPU.NCE operations guarantees that invariants
    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::TransposedConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    if (filterType.getRank() != 4) {
        log.trace("[{0}] Filter has unsupported rank: {1}", origOp->getLoc(), filterType.getRank());
        return mlir::failure();
    }

    const auto filterShape = filterType.getShape();
    const auto OC = filterShape[Dims4D::Filter::OC];
    auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    const auto outAlignment = channelsInfo.getOutputChannelAlignment();
    if (OC % outAlignment != 0) {
        log.trace("[{0}] Output channels '{1}' are not aligned", origOp->getLoc(), OC);
        return mlir::failure();
    }
    if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation())) {
        const auto IC = filterShape[Dims4D::Filter::IC];
        if (IC % iface.getInputChannelAlignment() != 0) {
            log.trace("[{0}] Input channels '{1}' are not aligned", origOp->getLoc(), IC);
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::PadOp origOp, Logger log) {
    log.setName("NCEInvariant");

    auto loc = origOp->getLoc();
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    const auto IC = inputShape[Dims4D::Act::C];
    const auto OC = outputShape[Dims4D::Act::C];
    if (IC % VPU::NCEInvariant::getAlignment(inputType.getElementType()) != 0) {
        log.trace("[{0}] Pad input channels '{1}' are not aligned", loc, IC);
        return mlir::failure();
    }

    auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    const auto outAlignment = channelsInfo.getOutputChannelAlignment();
    if (OC % outAlignment != 0) {
        log.trace("[{0}] Pad output channels '{1}' are not aligned", loc, OC);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::MatMulOp origOp, Logger log) {
    log.setName("NCEInvariant");

    auto loc = origOp->getLoc();
    auto inputType = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    if (inputType.getRank() != 4) {
        log.trace("[{0}] Input has unsupported rank: {1}", loc, inputType.getRank());
        return mlir::failure();
    }

    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (outputType.getRank() != 4) {
        log.trace("[{0}] Output has unsupported rank: {1}", loc, outputType.getRank());
        return mlir::failure();
    }

    const auto outputShape = outputType.getShape();
    const auto OC = outputShape.back();
    if (OC % VPU::NCEInvariant::getAlignment(outputType.getElementType()) != 0) {
        log.trace("[{0}] MatMul output channels are not aligned", loc);
        return mlir::failure();
    }

    VPUX_THROW_WHEN(origOp.getTransposeA(), "MatMul with transposeA is not supported.");

    const auto inputShape = inputType.getShape();
    const auto IC = inputShape.back();
    if (IC % VPU::NCEInvariant::getAlignment(inputType.getElementType()) != 0) {
        log.trace("[{0}] MatMul input channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

Byte getCMXSizeForTiling(mlir::ModuleOp module) {
    return vpux::VPU::getTotalCMXSize(module);
}

// verifyPipeliningCMX

bool isNestedTiling(const OutputTiling& tiling) {
    return tiling[0].axis[Dims4D::Act::C] > 1 && tiling[0].axis[Dims4D::Act::H] > 1;
}

vpux::NDTypeInterface getAlignedFilterType(const SmallVector<vpux::NDTypeInterface>& tileTypes) {
    const auto outputTileType = tileTypes[2];
    const auto filterTileType = tileTypes[1];
    const auto filterTileShape = filterTileType.getShape();
    const auto OC = filterTileShape[Dims4D::Filter::OC];
    const auto IC = filterTileShape[Dims4D::Filter::IC];
    const auto KY = filterTileShape[Dims4D::Filter::KY];
    const auto KX = filterTileShape[Dims4D::Filter::KX];

    const auto alignment = VPU::NCEInvariant::getAlignment(outputTileType.getElementType());
    const auto remainder = (IC * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, IC * KY * KX + padding};
    const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterTileType.getElementType());
    return alignedFilterType;
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::ConvolutionOp origOp,
                                                                    const OutputTiling& tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;

    const auto& curTileTypes = getTileTypes(origOp, curTile);
    const auto& nextTileTypes = getTileTypes(origOp, nextTile);

    SmallVector<vpux::NDTypeInterface> requiredOperands{curTileTypes[0], getAlignedFilterType(curTileTypes),
                                                        curTileTypes[2]};
    if (isWeightPrefetch) {
        requiredOperands.push_back(getAlignedFilterType(nextTileTypes));
    } else {
        requiredOperands.push_back(nextTileTypes[0]);
    }
    return requiredOperands;
}

template <class ConcreteOp>
SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipeliningConvBased(ConcreteOp origOp,
                                                                             const OutputTiling& tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = VPU::getTileTypes(origOp, curTile);
    const auto& nextTileTypes = VPU::getTileTypes(origOp, nextTile);

    const auto isOutputPipeliningEnabled = [](ConcreteOp origOp) {
        if (!origOp->hasAttr(outputPipelining)) {
            return false;
        }

        auto outputPipeliningAttr = origOp->getAttr(outputPipelining).template dyn_cast<mlir::BoolAttr>();
        if (outputPipeliningAttr == nullptr) {
            return false;
        }

        return outputPipeliningAttr.getValue();
    };

    if (isOutputPipeliningEnabled(origOp)) {
        return {curTileTypes[0],  curTileTypes[1],  curTileTypes[2],
                nextTileTypes[0], nextTileTypes[1], nextTileTypes[2]};
    }

    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;
    return {curTileTypes[0], curTileTypes[1], curTileTypes[2], isWeightPrefetch ? nextTileTypes[1] : nextTileTypes[0]};
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEConvolutionOp origOp,
                                                                    const OutputTiling& tiling) {
    return getRequiredOperandsForPipeliningConvBased(origOp, tiling);
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEInterpolateOp origOp,
                                                                    const OutputTiling& tiling) {
    return getRequiredOperandsForPipeliningConvBased(origOp, tiling);
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCECompressConvolutionOp origOp,
                                                                    const OutputTiling& tiling) {
    return getRequiredOperandsForPipeliningConvBased(origOp, tiling);
}

template <class ConcreteOp>
int64_t getRequiredChannelSizeForPipeliningConvBased(ConcreteOp origOp, const OutputTiling& tiling) {
    auto curFilterShape = getTileTypes(origOp, tiling[0])[1].getShape();
    auto nextFilterShape = getTileTypes(origOp, tiling[1])[1].getShape();
    return curFilterShape[Dims4D::Filter::OC] + nextFilterShape[Dims4D::Filter::OC];
}

int64_t getRequiredChannelSizeForPipelining(VPU::ConvolutionOp origOp, const OutputTiling& tiling) {
    return getRequiredChannelSizeForPipeliningConvBased(origOp, tiling);
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEConvolutionOp origOp, const OutputTiling& tiling) {
    return getRequiredChannelSizeForPipeliningConvBased(origOp, tiling);
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEInterpolateOp origOp, const OutputTiling& tiling) {
    return getRequiredChannelSizeForPipeliningConvBased(origOp, tiling);
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCECompressConvolutionOp origOp, const OutputTiling& tiling) {
    return getRequiredChannelSizeForPipeliningConvBased(origOp, tiling);
}

template <class ConcreteOp>
mlir::LogicalResult verifyPipeliningCMXConvBased(ConcreteOp origOp, const OutputTiling& tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);
    auto cmxWithFragmentationRatio = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(cmxSize.count()) * FRAGMENTATION_AVOID_RATIO_PIPELINING)));

    Byte requiredCMX = Byte(0);

    requiredCMX = VPU::getRequiredCMXSizeForNCEOps(getRequiredOperandsForPipelining(origOp, tiling),
                                                   getRequiredChannelSizeForPipelining(origOp, tiling));
    if (requiredCMX > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxWithFragmentationRatio, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::ConvolutionOp origOp,
                                                                   const OutputTiling& tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);
    auto cmxWithFragmentationRatio = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(cmxSize.count()) * FRAGMENTATION_AVOID_RATIO_PIPELINING)));

    Byte requiredCMX = Byte(0);

    requiredCMX = VPU::getRequiredCMXSize(getRequiredOperandsForPipelining(origOp, tiling));
    if (requiredCMX > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxWithFragmentationRatio, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEConvolutionOp origOp,
                                                                   const OutputTiling& tiling, Logger log) {
    return verifyPipeliningCMXConvBased(origOp, tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEInterpolateOp origOp,
                                                                   const OutputTiling& tiling, Logger log) {
    return verifyPipeliningCMXConvBased(origOp, tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCECompressConvolutionOp origOp,
                                                                   const OutputTiling& tiling, Logger log) {
    return verifyPipeliningCMXConvBased(origOp, tiling, log);
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::MaxPoolOp origOp, const OutputTiling& tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = VPU::getTileTypes(origOp, curTile);
    const auto& nextTileTypes = VPU::getTileTypes(origOp, nextTile);
    SmallVector<vpux::NDTypeInterface> requiredOperands{curTileTypes[0], curTileTypes[1], nextTileTypes[0]};
    return requiredOperands;
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEMaxPoolOp origOp,
                                                                    const OutputTiling& tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = VPU::getTileTypes(origOp, curTile);
    const auto& nextTileTypes = VPU::getTileTypes(origOp, nextTile);
    return {curTileTypes[0], curTileTypes[1], nextTileTypes[0]};
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEAveragePoolOp origOp,
                                                                    const OutputTiling& tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = VPU::getTileTypes(origOp, curTile);
    const auto& nextTileTypes = VPU::getTileTypes(origOp, nextTile);
    return {curTileTypes[0], curTileTypes[1], nextTileTypes[0]};
}

int64_t getRequiredChannelSizeForPipelining(VPU::MaxPoolOp origOp, const OutputTiling& tiling) {
    auto curInputShape = VPU::getTileTypes(origOp, tiling[0])[0].getShape();
    auto nextInputShape = VPU::getTileTypes(origOp, tiling[1])[0].getShape();
    return curInputShape[Dims4D::Act::C] + nextInputShape[Dims4D::Act::C];
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEMaxPoolOp origOp, const OutputTiling& tiling) {
    auto curInputShape = VPU::getTileTypes(origOp, tiling[0])[0].getShape();
    auto nextInputShape = VPU::getTileTypes(origOp, tiling[1])[0].getShape();
    return curInputShape[Dims4D::Act::C] + nextInputShape[Dims4D::Act::C];
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEAveragePoolOp origOp, const OutputTiling& tiling) {
    auto curInputShape = VPU::getTileTypes(origOp, tiling[0])[0].getShape();
    auto nextInputShape = VPU::getTileTypes(origOp, tiling[1])[0].getShape();
    return curInputShape[Dims4D::Act::C] + nextInputShape[Dims4D::Act::C];
}

Byte getRequiredActWindowForPipelining(VPU::MaxPoolOp origOp) {
    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(origOp.getKernelSizeAttr()));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(origOp.getStridesAttr()));

    //  Consider tiling does not change the element type
    const auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto activationWindowSizePerTile = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::POOL, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inType.getElementType(), 1);
    return Byte(activationWindowSizePerTile * 2);
}

Byte getRequiredActWindowForPipelining(VPU::NCEMaxPoolOp origOp) {
    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(origOp.getKernelSizeAttr()));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(origOp.getStridesAttr()));

    //  Consider tiling does not change the element type
    const auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto activationWindowSizePerTile = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::POOL, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inType.getElementType(), 1);
    return Byte(activationWindowSizePerTile * 2);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::MaxPoolOp origOp, const OutputTiling& tiling,
                                                                   Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);
    auto cmxWithFragmentationRatio = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(cmxSize.count()) * FRAGMENTATION_AVOID_RATIO_PIPELINING)));

    Byte requiredCMX = Byte(0);

    requiredCMX = VPU::getRequiredCMXSize(getRequiredOperandsForPipelining(origOp, tiling));
    if (requiredCMX > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxWithFragmentationRatio, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEMaxPoolOp origOp, const OutputTiling& tiling,
                                                                   Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);
    auto cmxWithFragmentationRatio = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(cmxSize.count()) * FRAGMENTATION_AVOID_RATIO_PIPELINING)));

    Byte requiredCMX = Byte(0);

    requiredCMX = VPU::getRequiredCMXSizeForNCEOps(getRequiredOperandsForPipelining(origOp, tiling),
                                                   getRequiredChannelSizeForPipelining(origOp, tiling));
    if (origOp.getActivationWindow() != nullptr) {
        requiredCMX += getRequiredActWindowForPipelining(origOp);
    }
    if (requiredCMX > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxWithFragmentationRatio, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEAveragePoolOp origOp,
                                                                   const OutputTiling& tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);
    auto cmxWithFragmentationRatio = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(cmxSize.count()) * FRAGMENTATION_AVOID_RATIO_PIPELINING)));

    Byte requiredCMX = Byte(0);
    requiredCMX = VPU::getRequiredCMXSizeForNCEOps(getRequiredOperandsForPipelining(origOp, tiling),
                                                   getRequiredChannelSizeForPipelining(origOp, tiling));
    if (requiredCMX > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxWithFragmentationRatio, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::GroupConvolutionOp origOp,
                                                                    const OutputTiling& tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;

    const auto& curTileTypes = VPU::getTileTypes(origOp, curTile);
    const auto& nextTileTypes = VPU::getTileTypes(origOp, nextTile);
    SmallVector<vpux::NDTypeInterface> requiredOperands{curTileTypes[0], getAlignedFilterType(curTileTypes),
                                                        curTileTypes[2]};
    if (isWeightPrefetch) {
        requiredOperands.push_back(getAlignedFilterType(nextTileTypes));
    } else {
        requiredOperands.push_back(nextTileTypes[0]);
    }
    return requiredOperands;
}

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(VPU::NCEDepthConvolutionOp origOp,
                                                                    const OutputTiling& tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];
    bool isWeightPrefetch = curTile.axis[Dims4D::Act::C] > 1;

    const auto& curTileTypes = VPU::getTileTypes(origOp, curTile);
    const auto& nextTileTypes = VPU::getTileTypes(origOp, nextTile);

    return {curTileTypes[0], curTileTypes[1], curTileTypes[2], isWeightPrefetch ? nextTileTypes[1] : nextTileTypes[0]};
}

int64_t getRequiredChannelSizeForPipelining(VPU::GroupConvolutionOp origOp, const OutputTiling& tiling) {
    auto curFilterShape = VPU::getTileTypes(origOp, tiling[0])[1].getShape();
    auto nextFilterShape = VPU::getTileTypes(origOp, tiling[1])[1].getShape();
    return curFilterShape[Dims4D::Filter::OC] + nextFilterShape[Dims4D::Filter::OC];
}

int64_t getRequiredChannelSizeForPipelining(VPU::NCEDepthConvolutionOp origOp, const OutputTiling& tiling) {
    auto curFilterShape = VPU::getTileTypes(origOp, tiling[0])[1].getShape();
    auto nextFilterShape = VPU::getTileTypes(origOp, tiling[1])[1].getShape();
    return curFilterShape[Dims4D::Filter::OC] + nextFilterShape[Dims4D::Filter::OC];
}

Byte getRequiredActWindowForPipelining(VPU::GroupConvolutionOp origOp) {
    const auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();

    const Shape kernelSizeVals{filterType.getShape()[Dims4D::Filter::KY], filterType.getShape()[Dims4D::Filter::KX]};
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(origOp.getStridesAttr()));

    const auto activationWindowSizePerTile = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::DW_CONV, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inType.getElementType(), 1);

    return Byte(activationWindowSizePerTile * 2);
}

Byte getRequiredActWindowForPipelining(VPU::NCEDepthConvolutionOp origOp) {
    const auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();

    const Shape kernelSizeVals{filterType.getShape()[Dims4D::Filter::KY], filterType.getShape()[Dims4D::Filter::KX]};
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(origOp.getStridesAttr()));

    const auto activationWindowSizePerTile = VPU::NCESparsity::getActivationWindowSize(
            VPU::NCESparsity::Mode::DW_CONV, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
            inType.getElementType(), 1);

    return Byte(activationWindowSizePerTile * 2);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::GroupConvolutionOp origOp,
                                                                   const OutputTiling& tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);
    auto cmxWithFragmentationRatio = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(cmxSize.count()) * FRAGMENTATION_AVOID_RATIO_PIPELINING)));

    Byte requiredCMX = Byte(0);

    requiredCMX = VPU::getRequiredCMXSize(getRequiredOperandsForPipelining(origOp, tiling));
    if (requiredCMX > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxWithFragmentationRatio, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEDepthConvolutionOp origOp,
                                                                   const OutputTiling& tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);
    auto cmxWithFragmentationRatio = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(cmxSize.count()) * FRAGMENTATION_AVOID_RATIO_PIPELINING)));
    Byte requiredCMX = Byte(0);

    requiredCMX = VPU::getRequiredCMXSizeForNCEOps(getRequiredOperandsForPipelining(origOp, tiling),
                                                   getRequiredChannelSizeForPipelining(origOp, tiling));

    if (origOp.getActivationWindow() != nullptr) {
        requiredCMX += getRequiredActWindowForPipelining(origOp);
    }
    if (requiredCMX > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'",
                  origOp->getLoc(), cmxWithFragmentationRatio, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

//
// verifyEltwisePipeliningCMX
//

SmallVector<vpux::NDTypeInterface> getRequiredOperandsForPipelining(mlir::Operation* op, const OutputTiling& tiling) {
    // The tiling strategy follows last-tile-not-biggest
    // So just check the first two tiles are enough to make sure prefetchable
    auto curTile = tiling[0];
    auto nextTile = tiling[1];

    const auto& curTileTypes = VPU::getTileTypes(op, curTile);
    const auto& nextTileTypes = VPU::getTileTypes(op, nextTile);

    return SmallVector<vpux::NDTypeInterface>{curTileTypes[0], curTileTypes[1], curTileTypes[2], nextTileTypes[0],
                                              nextTileTypes[1]};
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyEltwisePipeliningCMX(mlir::Operation* op,
                                                                          const OutputTiling& tiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() <= 1) {
        return mlir::failure();
    }
    if (isNestedTiling(tiling)) {
        return mlir::failure();
    }

    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    Byte requiredCMX = Byte(0);

    requiredCMX = VPU::getRequiredCMXSizeForNCEOps({getRequiredOperandsForPipelining(op, tiling)}, 0);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}'", op->getLoc(),
                  cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::AddOp origOp, const OutputTiling& tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::MultiplyOp origOp, const OutputTiling& tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::SubtractOp origOp, const OutputTiling& tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::AndOp origOp, const OutputTiling& tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(VPU::NCEEltwiseOp origOp, const OutputTiling& tiling,
                                                                   Logger log) {
    return verifyEltwisePipeliningCMX(origOp.getOperation(), tiling, log);
}

//
// verifyPrefetchCMX
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(mlir::Operation* op, const OutputTiling& tiling,
                                                                 mlir::Operation* parentOp,
                                                                 const vpux::OutputTiling& parentTiling, Logger log) {
    log.setName("NCEInvariant");
    if (tiling.size() < 1 || parentTiling.size() < 1) {
        return mlir::failure();
    }
    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = getCMXSizeForTiling(module);

    // Calculate the CMX memory required by the last tile of parent Op
    auto lastParentTile = parentTiling.back();
    auto cmxRequiredByParent = VPU::getRequiredCMX(parentOp, lastParentTile, log);

    // Calculate the CMX memory required by the first tile of current op to prefetch
    auto firstPrefetchTile = tiling.back();
    auto cmxRequiredToPrefetch = VPU::getRequiredCMXForWeight(op, firstPrefetchTile);
    auto cmxWithFragmentationRatio =
            Byte(static_cast<int64_t>(std::ceil(static_cast<double>(cmxSize.count()) * FRAGMENTATION_AVOID_RATIO)));

    if (cmxRequiredByParent + cmxRequiredToPrefetch > cmxWithFragmentationRatio) {
        log.trace("[{0}] CMX memory is not enough for prefetch pipeline, available '{1}', required '{2}', required by "
                  "parent {3}",
                  op->getLoc(), cmxWithFragmentationRatio, cmxRequiredByParent + cmxRequiredToPrefetch,
                  cmxRequiredByParent);
        return mlir::failure();
    }

    return mlir::success();
}

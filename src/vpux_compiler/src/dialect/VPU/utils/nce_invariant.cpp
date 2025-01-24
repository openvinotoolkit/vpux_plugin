//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_padding_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

using namespace vpux;

//
// Precision checks
//

bool vpux::VPU::NCEInvariant::isPrecisionSupported(ArchKind arch, mlir::ValueRange vals, LogCb logCb) {
    for (const auto& val : vals) {
        const auto elemType = val.getType().cast<vpux::NDTypeInterface>().getElementType();

        if (elemType.isBF16() && arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX) {
            logCb(formatv("BF16 is only supported by NPU37XX, NPU40XX"));
            return false;
        }
    }

    return true;
}

//
// Fuse PadOp check
//

bool vpux::VPU::NCEInvariant::verifyPads(int64_t KY, int64_t KX, int64_t padTop, int64_t padBottom, int64_t padLeft,
                                         int64_t padRight, LogCb logCb) {
    if (padTop < 0 || padTop > KY / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padTop, KY / 2));
        return false;
    }
    if (padBottom < 0 || padBottom > KY / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padBottom, KY / 2));
        return false;
    }
    if (padLeft < 0 || padLeft > KX / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padLeft, KX / 2));
        return false;
    }
    if (padRight < 0 || padRight > KX / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padRight, KX / 2));
        return false;
    }

    return true;
}

bool vpux::VPU::NCEInvariant::verifyPads(mlir::ArrayAttr kernelSizeAttr, mlir::ArrayAttr padBeginAttr,
                                         mlir::ArrayAttr padEndAttr, LogCb logCb) {
    const auto kernelSize = parseIntArrayAttr<int64_t>(kernelSizeAttr);
    const auto KY = kernelSize[kernelSize.size() == 4 ? (Dims4D::Filter::KY.ind()) : (Dims4D::Kernel::Y.ind())];
    const auto KX = kernelSize[kernelSize.size() == 4 ? (Dims4D::Filter::KX.ind()) : (Dims4D::Kernel::X.ind())];

    const auto padsBegin = parseIntArrayAttr<int64_t>(padBeginAttr);
    const auto padsEnd = parseIntArrayAttr<int64_t>(padEndAttr);
    const auto padTop = padsBegin[Dims4D::PadsBegin::Top.ind()];
    const auto padLeft = padsBegin[Dims4D::PadsBegin::Left.ind()];
    const auto padBottom = padsEnd[Dims4D::PadsEnd::Bottom.ind()];
    const auto padRight = padsEnd[Dims4D::PadsEnd::Right.ind()];

    return verifyPads(KY, KX, padTop, padBottom, padLeft, padRight, logCb);
}

//
// Attributes checks
//

bool vpux::VPU::NCEInvariant::isAttrsSupported(mlir::Operation* op, int64_t KY, int64_t KX, int64_t SY, int64_t SX,
                                               int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                                               LogCb logCb) {
    const auto arch = VPU::getArch(op);
    if (VPU::hasMaxKernelSize(op)) {
        const auto maxKernelSize = VPU::getMaxKernelSize(op);

        if (KY > maxKernelSize || KY <= 0) {
            logCb(formatv("Unsupported kernel height dimension '{0}', must be in range [1, {1}]", KY, maxKernelSize));
            return false;
        }
        if (KX > maxKernelSize || KX <= 0) {
            logCb(formatv("Unsupported kernel width dimension '{0}', must be in range [1, {1}]", KX, maxKernelSize));
            return false;
        }
    }

    static const int64_t NCE_MAX_STRIDE_SIZE = 8;

    if (SX != SY && arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX) {
        logCb(formatv("Asymmetric strides are not supported"));
        return false;
    }
    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        logCb(formatv("Unsupported stride height dimension '{0}', must be in range [1, {1}]", SY, NCE_MAX_STRIDE_SIZE));
        return false;
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        logCb(formatv("Unsupported stride width dimension '{0}', must be in range [1, {1}]", SX, NCE_MAX_STRIDE_SIZE));
        return false;
    }

    return verifyPads(KY, KX, padTop, padBottom, padLeft, padRight, logCb);
}

//
// Activation type checks
//

bool vpux::VPU::NCEInvariant::isAligned(vpux::NDTypeInterface type, int64_t alignment, ArchKind arch, LogCb logCb) {
    const auto shape = type.getShape();
    const auto order = type.getDimsOrder();
    const auto memShape = order.toMemoryOrder(shape);

    const bool supportsSuperDense = VPU::NCEInvariant::isSuperdenseSupported(arch);
    // In super-dense mode only channels must be aligned.
    const auto channels = type.getRank() == 4 ? shape[Dims4D::Act::C] : shape[DimsGroups5D::Act::C];
    if (supportsSuperDense && channels % alignment == 0) {
        return true;
    }

    const auto innerDim = memShape.back();
    if (innerDim % alignment != 0) {
        logCb(formatv("Activation inner dimension '{0}' is not aligned to '{1}'", innerDim, alignment));
        return false;
    }

    return true;
}

bool is5DAligned(vpux::NDTypeInterface type, int64_t alignment, vpux::VPU::ArchKind arch, LogCb logCb) {
    const auto shape = type.getShape();
    const auto order = type.getDimsOrder();
    const auto memShape = order.toMemoryOrder(shape);

    const bool supportsSuperDense = VPU::NCEInvariant::isSuperdenseSupported(arch);
    // In super-dense mode only channels must be aligned.
    const auto channels = shape[DimsGroups5D::Act::C];
    if (supportsSuperDense && channels % alignment == 0) {
        return true;
    }

    const auto innerDim = memShape.back();
    if (innerDim % alignment != 0) {
        logCb(formatv("Activation inner dimension '{0}' is not aligned to '{1}'", innerDim, alignment));
        return false;
    }

    return true;
}

int64_t vpux::VPU::NCEInvariant::getAlignment(mlir::Type elemType) {
    const Bit typeSizeInBits = getElemTypeSize(elemType);
    return std::max<int64_t>(128 / typeSizeInBits.count(), 16);
}

bool vpux::VPU::NCEInvariant::isOutputActTypeSupported(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb) {
    if (type.getRank() == DimsGroups5D::Act::numDims) {
        const auto channels = type.getShape()[DimsGroups5D::Act::C];
        return channels % alignment == 0;
    }

    if (type.getRank() != 4) {
        logCb(formatv("Ouput activation has unsupported rank: {0}", type.getRank()));
        return false;
    }

    const auto OC = type.getShape()[Dims4D::Act::C];
    if (OC % alignment != 0) {
        logCb(formatv("Output input channels '{0}' are not aligned to '{1}'", OC, alignment));
        return false;
    }

    return true;
}

bool vpux::VPU::NCEInvariant::isInputActTypeSupported(ArchKind arch, vpux::NDTypeInterface type, int64_t alignment,
                                                      bool supportsInputActCompression, LogCb logCb) {
    if (type.getRank() == DimsGroups5D::Act::numDims) {
        return isAligned(type, alignment, arch, logCb);
    }

    if (type.getRank() != 4) {
        logCb(formatv("Input activation has unsupported rank: {0}", type.getRank()));
        return false;
    }

    if (supportsInputActCompression) {
        const auto IC = type.getShape()[Dims4D::Act::C];
        const bool inputChannelsMatch = (IC == VPU_COMPRESSED_INPUT_CHANNEL_NUM);
        if (!inputChannelsMatch) {
            logCb(formatv("Input channels do not match VPU_COMPRESSED_INPUT_CHANNEL_NUM: got {0} expected {1}", IC,
                          VPU_COMPRESSED_INPUT_CHANNEL_NUM));
        }
        return inputChannelsMatch;
    }

    return isAligned(type, alignment, arch, logCb);
}

//
// WeightsTable information
//

Byte vpux::VPU::NCEInvariant::getWeightsTableSize(int64_t OC) {
    return OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;
}

//
// Common utility for AvgPool, MaxPool, Eltwise and DWConv
//

bool vpux::VPU::NCEInvariant::checkLayouts(mlir::TypeRange operandTypes, mlir::TypeRange resultTypes,
                                           const VPU::ArchKind& arch, const unsigned numInputOperands, LogCb logCb) {
    for (unsigned opIdx = 0; opIdx < numInputOperands; opIdx++) {
        const auto actualInLayout = operandTypes[opIdx].cast<vpux::NDTypeInterface>().getDimsOrder();
        const auto& expectedInLayout = DimsOrder::NHWC;
        if (actualInLayout != expectedInLayout) {
            logCb(formatv("Unsupported input layout. Expected: {0}, got: {1}", expectedInLayout, actualInLayout));
            return false;
        }
    }

    for (auto resultType : resultTypes) {
        const auto actualOutLayout = resultType.cast<vpux::NDTypeInterface>().getDimsOrder();
        const auto& expectedOutLayout = DimsOrder::NHWC;
        if (arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX && actualOutLayout != expectedOutLayout) {
            logCb(formatv("Unsupported output layout. Expected: {0}, got: {1}", expectedOutLayout, actualOutLayout));
            return false;
        }
    }

    return true;
}

bool vpux::VPU::NCEInvariant::isSuperdenseSupported(const VPU::ArchKind arch) {
    const llvm::DenseSet<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU37XX,
            VPU::ArchKind::NPU40XX,
    };
    return compatibleTargets.contains(arch);
}

bool vpux::VPU::NCEInvariant::isElementwiseMultiplySupported(const VPU::ArchKind /*arch*/) {
    return false;
}

mlir::LogicalResult vpux::VPU::NCEInvariant::isSupported(mlir::Operation* op, Logger) {
    const bool checkLayout = false;
    const bool checkChannelAlignment = false;
    const bool allowDifferentScales = true;
    const bool allowDifferentZp = true;

    return mlir::success(
            llvm::TypeSwitch<mlir::Operation*, bool>(op)
                    .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp origOp) {
                        return VPU::NCEConvolutionOp::isSupported(origOp, emptyLogCb, checkLayout,
                                                                  checkChannelAlignment);
                    })
                    .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp origOp) {
                        return VPU::NCEMaxPoolOp::isSupported(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::AvgPoolOp>([&](IE::AvgPoolOp origOp) {
                        return VPU::NCEAveragePoolOp::isSupported(origOp, emptyLogCb, checkLayout,
                                                                  checkChannelAlignment);
                    })
                    .Case<IE::AddOp>([&](IE::AddOp origOp) {
                        return VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp,
                                                              emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::MultiplyOp>([&](IE::MultiplyOp origOp) {
                        const auto arch = getArch(origOp);
                        if (!isElementwiseMultiplySupported(arch)) {
                            return false;
                        }
                        return VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp,
                                                              emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::SubtractOp>([&](IE::SubtractOp origOp) {
                        return VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp,
                                                              emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::ReduceMeanOp>([&](IE::ReduceMeanOp origOp) {
                        return VPU::NCEReduceOp::isSupported(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::AndOp>([&](IE::AndOp origOp) {
                        return VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp,
                                                              emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp origOp) {
                        return VPU::NCEDepthConvolutionOp::isSupported(origOp, emptyLogCb, checkLayout,
                                                                       checkChannelAlignment);
                    })
                    .Case<IE::InterpolateOp>([&](IE::InterpolateOp origOp) {
                        return VPU::NCEInterpolateOp::isSupported(origOp, emptyLogCb, checkLayout,
                                                                  checkChannelAlignment, /*checkBatch=*/false);
                    })
                    .Case<IE::TransposedConvolutionOp>([&](IE::TransposedConvolutionOp origOp) {
                        return isSupportedSEPTransposedConv(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::PadOp>([&](IE::PadOp origOp) {
                        return isSupportedSEPPadOp(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::RollOp>([&](IE::RollOp origOp) {
                        return VPU::isSupportedSEPRoll(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Case<IE::MatMulOp>([&](IE::MatMulOp origOp) {
                        return VPU::NCEMatMulOp::isSupported(origOp, emptyLogCb, checkLayout, checkChannelAlignment);
                    })
                    .Default([](mlir::Operation*) -> bool {
                        return false;
                    }));
}

bool vpux::VPU::NCEInvariant::isSmallKernelOptimizationSupported(const VPU::ArchKind arch, mlir::Operation* op) {
    // TODO: E#96201, attach concrete implementation of NCEOpInterface depending on the type of device
    if (arch != VPU::ArchKind::NPU40XX) {
        return false;
    }
    if (!mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
        return false;
    }
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    // Skip Sparse Ops
    if (nceOp->getResult(0).getType().dyn_cast<VPU::SparseTensorType>() != nullptr) {
        return false;
    }

    // L1Opt can be enabled when kernelX = 3 and strideX = 1
    const auto kernelSize = nceOp.getKernelSizeVal();
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

    const auto kernelStride = nceOp.getStridesVal();
    const auto SX = kernelStride[Dims4D::Strides::X.ind()];

    // Get a set containing all the channels from the workloads of the given NCE operations
    mlir::DenseSet<int64_t> workloadsChannels;
    const auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();

    const auto isFp16Input = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType().isF16();

    const auto workloadChannelsMeetRequirement = llvm::all_of(workloads, [&](auto workload) {
        const auto wlSizes = parseIntArrayAttr<int64_t>(workload.getOutSizes());
        return isFp16Input ? wlSizes[Dims4D::Act::C.ind()] == VPU_CHANNEL_SIZE_FOR_L1OPT
                           : wlSizes[Dims4D::Act::C.ind()] % VPU_CHANNEL_SIZE_FOR_L1OPT == 0;
    });

    return KX == 3 && SX == 1 && workloadChannelsMeetRequirement;
}

//
// verifyKernel
//

mlir::LogicalResult vpux::VPU::NCEInvariant::verifyKernel(mlir::Operation* op, int64_t KY, int64_t KX, int64_t SY,
                                                          int64_t SX, int64_t padTop, int64_t padBottom,
                                                          int64_t padLeft, int64_t padRight, Logger log) {
    log.setName("NCEInvariant");
    auto loc = op->getLoc();
    const auto arch = VPU::getArch(op);
    if (VPU::hasMaxKernelSize(op)) {
        const auto maxKernelSize = VPU::getMaxKernelSize(op);

        if (KY > maxKernelSize || KY <= 0) {
            log.trace("[{0}] Unsupported kernel height dimension '{1}', must be in range [1, {2}]", loc, KY,
                      maxKernelSize);
            return mlir::failure();
        }
        if (KX > maxKernelSize || KX <= 0) {
            log.trace("[{0}] Unsupported kernel width dimension '{1}', must be in range [1, {2}]", loc, KX,
                      maxKernelSize);
            return mlir::failure();
        }
    }

    static const int32_t NCE_MAX_STRIDE_SIZE = 8;

    if (SX != SY && arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX) {
        log.trace("[{0}] Asymmetric strides are not supported", loc);
        return mlir::failure();
    }
    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        log.trace("[{0}] Unsupported stride height dimension '{1}', must be in range [1, {2}]", loc, SY,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        log.trace("[{0}] Unsupported stride width dimension '{1}', must be in range [1, {2}]", loc, SX,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }

    if (padTop < 0 || (padTop > 1 && padTop > KY / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padTop, KY / 2);
        return mlir::failure();
    }
    if (padBottom < 0 || (padBottom > 1 && padBottom > KY / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padBottom, KY / 2);
        return mlir::failure();
    }
    if (padLeft < 0 || (padLeft > 1 && padLeft > KX / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padLeft, KX / 2);
        return mlir::failure();
    }
    if (padRight < 0 || (padRight > 1 && padRight > KX / 2)) {
        log.trace("[{0}] Unsupported padding '{1}', must be in range [0, {2}]", loc, padRight, KX / 2);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEInvariant::verifyKernel(mlir::Operation* op, Logger) {
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp origOp) {
                return VPU::NCEConvolutionOp::verifyKernel(origOp);
            })
            .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp origOp) {
                return VPU::NCEMaxPoolOp::verifyKernel(origOp);
            })
            .Case<IE::AvgPoolOp>([&](IE::AvgPoolOp origOp) {
                return VPU::NCEAveragePoolOp::verifyKernel(origOp);
            })
            .Case<IE::AddOp>([&](IE::AddOp origOp) {
                return VPU::NCEEltwiseOp::verifyKernel(origOp);
            })
            .Case<IE::MultiplyOp>([&](IE::MultiplyOp origOp) {
                return VPU::NCEEltwiseOp::verifyKernel(origOp);
            })
            .Case<IE::SubtractOp>([&](IE::SubtractOp origOp) {
                return VPU::NCEEltwiseOp::verifyKernel(origOp);
            })
            .Case<IE::AndOp>([&](IE::AndOp origOp) {
                return VPU::NCEEltwiseOp::verifyKernel(origOp);
            })
            .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp origOp) {
                return VPU::NCEDepthConvolutionOp::verifyKernel(origOp);
            })
            .Case<IE::TransposedConvolutionOp>([&](IE::TransposedConvolutionOp origOp) {
                return VPU::NCEConvolutionOp::verifyKernel(origOp);
            })
            .Default([](mlir::Operation*) -> mlir::LogicalResult {
                return mlir::failure();
            });
}

//
// verifyPoolCMX
//

mlir::LogicalResult vpux::VPU::NCEInvariant::verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module,
                                                           vpux::NDTypeInterface inputType,
                                                           vpux::NDTypeInterface outputType, mlir::ArrayAttr kernelSize,
                                                           mlir::ArrayAttr kernelStrides, Logger log) {
    log.setName("NCEInvariant");

    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: {0}", kernelSize.size());
    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelSize.size());

    const auto outputShape = outputType.getShape();
    const auto OC = outputShape[Dims4D::Act::C];

    const auto requiredCMX = VPU::getRequiredCMXSizeForNCEOps({inputType, outputType}, OC);

    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Pooling, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

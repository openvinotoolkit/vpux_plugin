//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/workload_splitter_base.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <array>

using namespace vpux;
using namespace VPU;

vpux::VPU::WorkloadSplitterBase::WorkloadSplitterBase(mlir::func::FuncOp funcOp,
                                                      ArrayRef<int64_t> supportedChannelsForDW, vpux::Logger log)
        : _funcOp(funcOp),
          _supportedChannelsForDW(supportedChannelsForDW.begin(), supportedChannelsForDW.end()),
          _log(log) {
}

bool vpux::VPU::WorkloadSplitterBase::isDepthwiseOp(mlir::Operation* op) {
    return mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(op);
}

void vpux::VPU::WorkloadSplitterBase::correctInvalidWorkload(const VPU::SparsityConstraint& sparsityConstraint) {
    mlir::DenseSet<mlir::Operation*> handledNCEOps;

    _funcOp.walk([&](VPU::NCEOpInterface nceOp) {
        if (handledNCEOps.contains(nceOp)) {
            return;
        }

        // More than one operation might need to be handled at the same time for some sparse activations,
        // to satisfy the requirements of the consumer ops
        mlir::DenseSet<mlir::Operation*> producerNCEOps{nceOp};
        const auto invalidSparseOps = findInvalidSparseOps(nceOp, sparsityConstraint);
        if (!invalidSparseOps.empty()) {
            producerNCEOps.clear();
            producerNCEOps.insert(invalidSparseOps.begin(), invalidSparseOps.end());
        }

        const auto invalidDepthwiseOps = findInvalidDepthwiseOps(producerNCEOps);
        const auto invalidNCEPermuteOps = findInvalidNCEPermuteOps(producerNCEOps);
        if (invalidSparseOps.empty() && invalidDepthwiseOps.empty() && invalidNCEPermuteOps.empty()) {
            return;
        }

        const auto supportedChannels = getSupportedChannels(producerNCEOps, sparsityConstraint);
        _log.trace("supportedChannels {0} for nceOp {1} to correct workloads", supportedChannels, nceOp->getLoc());
        auto channelPadding = 0;  // used for NCEPermute

        int64_t opIdx = 1;
        for (auto op : producerNCEOps) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
            VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);

            auto isInvalidDepthwise = invalidDepthwiseOps.contains(op);
            auto isInvalidSparsity = invalidSparseOps.contains(op);
            auto isInvalidNCEPermuteOp = invalidNCEPermuteOps.contains(op);
            _log.trace("Correcting workloads for operation '{0}' at '{1}'. Necessary corrections: depthwise "
                       "'{2}', sparsity '{3}' ({4}/{5}), remove padding '{6}'",
                       op->getName(), op->getLoc(), isInvalidDepthwise, isInvalidSparsity, opIdx++,
                       producerNCEOps.size(), isInvalidNCEPermuteOp);

            const auto offsetsCorrectionNeeded = isNCEPermuteOffsetsCorrectionNeeded(nceOp);
            if (isInvalidNCEPermuteOp) {
                channelPadding = op->getResult(0).getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C] -
                                 op->getOperand(0).getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
            }
            auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();
            for (auto workloadOp : llvm::make_early_inc_range(workloads)) {
                const auto wlSizes = parseIntArrayAttr<int64_t>(workloadOp.getOutSizes());
                auto wlChannels = wlSizes[Dims4D::Act::C.ind()];
                if (llvm::find(supportedChannels, wlChannels) != supportedChannels.end()) {
                    continue;
                }

                splitWorkload(workloadOp, supportedChannels, isInvalidNCEPermuteOp, channelPadding,
                              offsetsCorrectionNeeded, _log);
            }

            handledNCEOps.insert(op);
        }
    });
}

SmallVector<Shape> vpux::VPU::WorkloadSplitterBase::getPerClusterShapesWhenSOK(VPU::NCEOpInterface nceOp) {
    SmallVector<Shape> perClusterShapes = {};
    auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value() &&
        clusterOp.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto outputType = clusterOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        auto numClusters = VPU::getOptimalNumClusters(clusterOp, outputType.getShape(),
                                                      VPU::MultiClusterStrategy::SplitOverKernel);
        auto distributedType = getDistributedOutputTypeFromOp(clusterOp, outputType, numClusters,
                                                              VPU::MultiClusterStrategy::SplitOverKernel);
        perClusterShapes = distributedType.getDistributedTypes()
                                   .front()
                                   .cast<VPU::DistributedTensorType>()
                                   .getPerClusterComputeShapes();
    }
    return perClusterShapes;
}

// Get a set containing all the channels from the workloads of the given NCE operations
mlir::DenseSet<int64_t> vpux::VPU::WorkloadSplitterBase::getWorkloadsChannels(
        const mlir::DenseSet<mlir::Operation*>& nceOps, bool skipLastWorkload) {
    SmallVector<VPU::DPUWorkloadOp> allWorkloads;
    mlir::DenseSet<int64_t> workloadsChannels;
    for (auto op : nceOps) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);
        const auto workloads = to_small_vector(nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>());
        allWorkloads.insert(allWorkloads.end(), workloads.begin(), workloads.end());
    }

    if (!allWorkloads.empty()) {
        if (skipLastWorkload) {
            allWorkloads.pop_back();
        }

        auto channels = to_container<mlir::DenseSet<int64_t>>(
                allWorkloads | transformed([](VPU::DPUWorkloadOp workload) -> int64_t {
                    const auto wlSizes = parseIntArrayAttr<int64_t>(workload.getOutSizes());
                    return wlSizes[Dims4D::Act::C.ind()];
                }));
        workloadsChannels.insert(channels.begin(), channels.end());
    } else {
        for (auto op : nceOps) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
            auto perClusterShapes = getPerClusterShapesWhenSOK(nceOp);

            if (!perClusterShapes.empty()) {
                auto channels = to_container<mlir::DenseSet<int64_t>>(perClusterShapes |
                                                                      transformed([](Shape clusterShape) -> int64_t {
                                                                          return clusterShape[Dims4D::Act::C];
                                                                      }));
                workloadsChannels.insert(channels.begin(), channels.end());
            } else {
                const auto outputType = nceOp.getOperation()->getResult(0).getType().cast<NDTypeInterface>();
                const auto OC = outputType.getShape()[vpux::Dims4D::Act::C];
                workloadsChannels.insert(OC);
            }
        }
    }

    return workloadsChannels;
}

// Find the operations which can consume the given value. The value should be of sparse type, therefore the
// consumers can be NCE, Desparsify or Return ops
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitterBase::findConsumerOps(mlir::Value value) {
    mlir::DenseSet<mlir::Operation*> consumerOps;
    for (auto userOp : value.getUsers()) {
        auto taskOp = userOp;
        if (auto nceClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(userOp)) {
            taskOp = nceClusterOp.getInnerTaskOp();
        }

        if (mlir::isa<VPU::NCEOpInterface, VPU::DesparsifyOp, mlir::func::ReturnOp>(taskOp)) {
            consumerOps.insert(userOp);
        } else if (mlir::isa<VPU::CopyOp>(taskOp) || VPU::isPureViewOp(taskOp)) {
            auto ops = findConsumerOps(userOp->getResult(0));
            consumerOps.insert(ops.begin(), ops.end());
        }
    }
    return consumerOps;
}

// Find all the NCE operations that produce the value. Multiple operations can produce a value in case it is
// concatenated or grouped
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitterBase::findProducerNCEOps(mlir::Value value) {
    mlir::DenseSet<mlir::Operation*> producerNCEOps;

    auto producerOp = value.getDefiningOp();
    auto taskOp = producerOp;
    if (auto nceClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(producerOp)) {
        taskOp = nceClusterOp.getInnerTaskOp();
    }

    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(taskOp)) {
        producerNCEOps.insert(nceOp);
    } else if (mlir::isa<VPU::CopyOp>(taskOp)) {
        const auto ops = findProducerNCEOps(producerOp->getOperand(0));
        producerNCEOps.insert(ops.begin(), ops.end());
    } else if (VPU::isPureViewOp(producerOp)) {
        if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(producerOp)) {
            for (const auto& input : concatOp.getInputs()) {
                const auto ops = findProducerNCEOps(input);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        } else if (auto viewOp = mlir::dyn_cast<VPU::ViewLikeOpInterface>(producerOp)) {
            const auto ops = findProducerNCEOps(viewOp->getOperand(0));
            producerNCEOps.insert(ops.begin(), ops.end());
        } else if (auto viewOp = mlir::dyn_cast<MultiViewOpInterface>(producerOp)) {
            if (auto opResult = value.dyn_cast<mlir::OpResult>()) {
                const auto source = viewOp.getViewSource(opResult.getResultNumber());
                const auto ops = findProducerNCEOps(source);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        } else if (auto viewOp = mlir::dyn_cast<GroupedViewOpInterface>(producerOp)) {
            for (const auto& source : viewOp.getViewSources()) {
                const auto ops = findProducerNCEOps(source);
                producerNCEOps.insert(ops.begin(), ops.end());
            }
        }
    }

    return producerNCEOps;
}

// Find all the consumer operations of the value, then find all the producer NCE operations for the input values of
// the consumers. This is then repeated until no new consumers are identified. Chains of operations such as the
// following ensure that all three input Convolutions are returned when the function is called on the value marked
// with '*':
//   Conv   Conv   Conv
//     \*  /    \  /
//     Concat  Concat
//       |       |
//      Conv    Conv
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitterBase::findProducersForConsumers(
        mlir::Value value, mlir::DenseSet<mlir::Operation*> processedConsumerOps) {
    mlir::DenseSet<mlir::Operation*> producerNCEOps;

    auto consumerOps = findConsumerOps(value);
    for (auto consumerOp : consumerOps) {
        if (processedConsumerOps.contains(consumerOp)) {
            continue;
        }

        auto producerOps = findProducerNCEOps(consumerOp->getOperand(0));
        producerNCEOps.insert(producerOps.begin(), producerOps.end());

        if (mlir::isa<VPU::NCEEltwiseOp>(consumerOp)) {
            auto producerOpsInput2 = findProducerNCEOps(consumerOp->getOperand(1));
            producerNCEOps.insert(producerOpsInput2.begin(), producerOpsInput2.end());
        }
    }
    processedConsumerOps.insert(consumerOps.begin(), consumerOps.end());

    mlir::DenseSet<mlir::Operation*> newProducerOps;
    for (auto producerOp : producerNCEOps) {
        if (producerOp->getResult(0) == value) {
            continue;
        }
        auto producerOps = findProducersForConsumers(producerOp->getResult(0), processedConsumerOps);
        newProducerOps.insert(producerOps.begin(), producerOps.end());
    }

    producerNCEOps.insert(newProducerOps.begin(), newProducerOps.end());

    return producerNCEOps;
}

// Invariants that produce sparse activations must satisfy two conditions:
// - all variants must produce the same number of channels
// - the number of channels is a power of two (for NPU37XX)
// Additionally, in case a consumer operation has its input produced by multiple NCE operations,
// all of the producer ops need to have the same number of channels for their variants.
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitterBase::findInvalidSparseOps(
        VPU::NCEOpInterface nceOp, const VPU::SparsityConstraint& sparsityConstraint) {
    mlir::DenseSet<mlir::Operation*> invalidSparseOps;

    if (!nceOp->getResult(0).getType().isa<VPU::SparseTensorType>()) {
        return invalidSparseOps;
    }

    auto result = nceOp->getResult(0);
    if (auto parentOp = nceOp->getParentOfType<VPU::NCEClusterTilingOp>()) {
        result = parentOp->getResult(0);
    }
    auto producerOps = findProducersForConsumers(result);
    auto workloadsChannels = getWorkloadsChannels(producerOps);

    std::optional<int64_t> numChannels = std::nullopt;
    auto invalidWorkloads = llvm::any_of(workloadsChannels, [&](int64_t channels) -> bool {
        if (!numChannels.has_value()) {
            numChannels = channels;
        }
        if (channels != numChannels) {
            return true;
        }
        if (!sparsityConstraint.areChannelsFitForSESize(channels)) {
            return true;
        }
        return false;
    });
    if (invalidWorkloads) {
        invalidSparseOps.insert(producerOps.begin(), producerOps.end());
    }

    return invalidSparseOps;
}

// Depthwise operations must have variants that produce 16, 32 or 64 channels
mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitterBase::findInvalidDepthwiseOps(
        const mlir::DenseSet<mlir::Operation*>& nceOps) {
    mlir::DenseSet<mlir::Operation*> invalidDepthwiseOps;
    for (auto op : nceOps) {
        if (!isDepthwiseOp(op)) {
            continue;
        }
        const auto workloadsChannels = getWorkloadsChannels({op});
        const auto invalidChannels = llvm::any_of(workloadsChannels, [&](const int64_t channels) -> bool {
            return llvm::find(_supportedChannelsForDW, channels) == _supportedChannelsForDW.end();
        });
        if (invalidChannels) {
            invalidDepthwiseOps.insert(op);
        }
    }
    return invalidDepthwiseOps;
}

mlir::DenseSet<mlir::Operation*> vpux::VPU::WorkloadSplitterBase::findInvalidNCEPermuteOps(
        const mlir::DenseSet<mlir::Operation*>& nceOps) {
    mlir::DenseSet<mlir::Operation*> invalidNCEPermuteOps;
    for (auto op : nceOps) {
        if (!mlir::isa<VPU::NCEPermuteOp>(op)) {
            continue;
        }
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Expected NCE op, got '{0}'", op);
        const auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();
        const auto nonZeroPadding = llvm::any_of(workloads, [&](VPU::DPUWorkloadOp workload) -> bool {
            const auto expandChannels = mlir::cast<VPU::NCEPermuteOp>(op).getExpandedChannels();
            const auto origInChannels = op->getOperand(0).getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
            const auto zeroPadding = expandChannels == origInChannels;
            const auto wlOffsets = parseIntArrayAttr<int64_t>(workload.getOutOffsetsAttr());
            const auto isZeroPredicate = [](const int64_t value) -> bool {
                return value == 0;
            };
            const bool zeroOffsets = std::all_of(wlOffsets.begin(), wlOffsets.end(), isZeroPredicate);
            return !zeroPadding || !zeroOffsets;
        });
        if (nonZeroPadding) {
            invalidNCEPermuteOps.insert(op);
        }
    }

    return invalidNCEPermuteOps;
}

/// @brief Get all of the supported channels that can be used to split all of the given workloads, so that the
/// depthwise and sparsity requirements are met.
/// @details For the normal case, workload channel can support [16, 32, ... 8192]. If it's a depthwise op, workload
/// channels only support [16, 32, 64]. That's the first part of this function to generate a pool of
/// supportedChannels that can be used to split all of the given workloads, based on it's depthwise or not. Then if
/// there is output sparsity, we will collect all the workload channels excluding the last one. And we will check
/// each element in supportedChannels. If all workload channels (excluding the last one) are divisible by this
/// element, it's a supported channnel. Otherwise, we remove it from supportedChannels. For example,
/// 1. There's a convolution with 256 OC and output sparsity, split on 6 tiles generates [48, 48, 48, 48, 48, 16].
/// The original supportedChannels are [16, 32, ... 8192]. After filtering out unsupported channels due to sparsity,
/// supportedChannels will be just [16, 48].
/// 2. There's a DW conv with 256 OC and output sparsity, split on 6 tiles generates [48, 48, 48, 48, 48, 16]. The
/// original supportedChannels are [16, 32, 64]. After filtering out unsupported channels, supportedChannels will be
/// just [16].
SmallVector<int64_t> vpux::VPU::WorkloadSplitterBase::getSupportedChannels(
        const mlir::DenseSet<mlir::Operation*>& nceOps, const VPU::SparsityConstraint& sparsityConstraint) {
    SmallVector<int64_t> supportedChannels;

    const auto hasDepthwiseOp = llvm::any_of(nceOps, isDepthwiseOp);
    if (hasDepthwiseOp) {
        supportedChannels.insert(supportedChannels.end(), _supportedChannelsForDW.begin(),
                                 _supportedChannelsForDW.end());
    }

    const auto hasSparseOutput = llvm::any_of(nceOps, [](mlir::Operation* op) {
        return op->getResult(0).getType().isa<VPU::SparseTensorType>();
    });
    if (hasSparseOutput) {
        if (supportedChannels.empty()) {
            for (int64_t channels = VPU::NCEInvariant::VPU_DIMENSION_LIMIT; channels >= 16; channels -= 16) {
                if (!sparsityConstraint.areChannelsFitForSESize(channels)) {
                    continue;
                }
                supportedChannels.push_back(channels);
            }
        }

        auto eraseInvalidChannels = [&](const mlir::DenseSet<int64_t>& workloadsChannels) -> void {
            supportedChannels.erase(std::remove_if(supportedChannels.begin(), supportedChannels.end(),
                                                   [&](const int64_t channels) {
                                                       for (auto wlChannels : workloadsChannels) {
                                                           if (wlChannels % channels != 0) {
                                                               return true;
                                                           }
                                                       }
                                                       return false;
                                                   }),
                                    supportedChannels.end());
        };

        // When we need to check multiple NCE ops with output sparsity, the logic of filtering out unsupported
        // channels become complicated. We need to check for all the workload channels from all NCE ops excluding
        // the last workload from the last NCE op. There's not a simple way to tell which NCE op provides the last
        // workload. It's not safe to rely on IR order because for the senario where multiple NCE ops are connected
        // to the concat's inputs, their execution order are not determined. So if nceOps.size() > 1, just keep the
        // logic as before where we collect all the workload channels from all NCE ops.
        if (nceOps.size() > 1) {
            eraseInvalidChannels(getWorkloadsChannels(nceOps));
        } else if (!nceOps.empty()) {
            eraseInvalidChannels(getWorkloadsChannels(nceOps, true));
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(*(nceOps.begin()));
            const auto outputType = nceOp.getOperation()->getResult(0).getType().cast<NDTypeInterface>();
            auto lastChannel = outputType.getShape()[vpux::Dims4D::Act::C];
            auto workloads = to_small_vector(nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>());

            if (!workloads.empty()) {
                const auto lastWorkloadSizes = parseIntArrayAttr<int64_t>(workloads.back().getOutSizes());
                lastChannel = lastWorkloadSizes[Dims4D::Act::C.ind()];

            } else {
                auto perClusterShapes = getPerClusterShapesWhenSOK(nceOp);
                if (!perClusterShapes.empty()) {
                    lastChannel = perClusterShapes.back()[Dims4D::Act::C];
                }
            }

            // If the last workload's channel can't be supported, we need to make sure it can be represented as
            // a combination of supported channels. For example, A DW Conv which has single workload with
            // channel 96 should be split to {64, 32} A DW Conv which has single workload with channel 112
            // should be split to {32, 32 ,32}
            if (llvm::find(supportedChannels, lastChannel) == supportedChannels.end()) {
                supportedChannels.erase(
                        std::remove_if(supportedChannels.begin(), supportedChannels.end(),
                                       [&](const int64_t channels) {
                                           return (lastChannel % channels) &&
                                                  (llvm::find(supportedChannels, lastChannel % channels) ==
                                                   supportedChannels.end());
                                       }),
                        supportedChannels.end());
            }
        }
    }

    return supportedChannels;
}

// Splits the workload channels so that they are composed out of the values in the `supportedChannels` array, if it
// is provided. Additionally, removes the padding and spatial offsets from the workload based on the `removePadding`
// flag
void vpux::VPU::WorkloadSplitterBase::splitWorkload(VPU::DPUWorkloadOp dpuWorkloadOp,
                                                    ArrayRef<int64_t> supportedChannels,
                                                    const bool isInvalidNCEPermuteOp, int64_t channelPadding,
                                                    bool isNCEPermuteOffsetsCorrectionNeeded, Logger log) {
    auto wlSizes = parseIntArrayAttr<int64_t>(dpuWorkloadOp.getOutSizesAttr());
    auto wlOffsets = parseIntArrayAttr<int64_t>(dpuWorkloadOp.getOutOffsetsAttr());
    auto padsAttr = dpuWorkloadOp.getPad();

    mlir::OpBuilder builder(dpuWorkloadOp);
    if (isInvalidNCEPermuteOp) {
        const auto pads = dpuWorkloadOp.getPad();
        const auto top = pads.getTop().getInt();
        const auto bottom = pads.getBottom().getInt();
        const auto left = pads.getLeft().getInt();
        const auto right = pads.getRight().getInt();
        wlSizes[Dims4D::Act::H.ind()] -= (top + bottom);
        wlSizes[Dims4D::Act::W.ind()] -= (left + right);
        wlSizes[Dims4D::Act::C.ind()] -= channelPadding;

        padsAttr = VPU::getPaddingAttr(pads.getContext(), PadInfo(0, 0, 0, 0));

        if (isNCEPermuteOffsetsCorrectionNeeded) {
            wlOffsets = SmallVector<int64_t>{0, 0, 0, 0};
        }
    }

    SmallVector<int64_t> newWorkloadChannels;
    auto wlChannels = wlSizes[Dims4D::Act::C.ind()];
    if (supportedChannels.empty()) {
        newWorkloadChannels.push_back(wlChannels);
    } else {
        newWorkloadChannels = splitWorkloadChannel(wlChannels, supportedChannels);
        VPUX_THROW_WHEN(newWorkloadChannels.size() == 0,
                        "splitWorkloadChannel failed please check wlChannel - {0}, supportedChannelsDW - {1}",
                        wlChannels, supportedChannels);
    }

    auto channelOffset = wlOffsets[Dims4D::Act::C.ind()];

    for (auto channelSize : newWorkloadChannels) {
        auto sizes = wlSizes;
        sizes[Dims4D::Act::C.ind()] = channelSize;

        auto offsets = wlOffsets;
        offsets[Dims4D::Act::C.ind()] = channelOffset;
        channelOffset += channelSize;

        const auto offsetsAttr = getIntArrayAttr(builder.getContext(), offsets);
        const auto sizesAttr = getIntArrayAttr(builder.getContext(), sizes);

        builder.create<VPU::DPUWorkloadOp>(dpuWorkloadOp.getLoc(), offsetsAttr, sizesAttr, padsAttr,
                                           dpuWorkloadOp.getMpeModeAttr(), dpuWorkloadOp.getClusterIdAttr());
    }

    log.nest().trace("Split workload of size '{0}' into '{1}'", wlSizes[Dims4D::Act::C.ind()], newWorkloadChannels);
    dpuWorkloadOp.erase();
}

//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        return DistributionMode::OVERLAPPED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
               strategy == VPU::MultiClusterStrategy::HKSwitch ||
               strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::InterpolateOp interpolateOp,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    auto inType = interpolateOp.getInput().getType().cast<NDTypeInterface>();
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        return (inType != inputType) ? DistributionMode::DUPLICATED : DistributionMode::OVERLAPPED;
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

VPU::DistributionMode vpux::VPU::getSWInputTensorDistributionMode(mlir::Operation* eltwiseOp,
                                                                  VPU::MultiClusterStrategy strategy,
                                                                  vpux::NDTypeInterface inputType) {
    auto isTileAtBroadCastAxis = [&](vpux::Dim tileAxis) {
        if (!eltwiseOp->hasAttr("auto_broadcast")) {
            return false;
        }
        const auto outputShape = getShape(eltwiseOp->getResult(0));
        const auto inputShape = inputType.getShape();
        VPUX_THROW_UNLESS(inputShape.size() == outputShape.size(),
                          "Input tensor rank {0} is mismatched with Output tensor rank {1}", inputShape.size(),
                          outputShape.size());
        return (outputShape[tileAxis] != inputShape[tileAxis]) && (inputShape[tileAxis] == 1);
    };

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return isTileAtBroadCastAxis(Dims4D::Act::W) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return isTileAtBroadCastAxis(Dims4D::Act::H) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return isTileAtBroadCastAxis(Dims4D::Act::C) ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::PReluOp preluOp, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    auto preluSlopeType = preluOp.getNegativeSlope().getType().cast<NDTypeInterface>();

    // It is possible that two inputs has the same type.
    // For this case, cannot be completely sure if this type is a Slope input.
    // However, this does not conflict with the selection of DistributionMode.
    // The Slope input will only have at most one axis with size greater than 1 at channel
    // and this value equals to the input channel size.
    auto isPotentialSlopeInput = (inputType == preluSlopeType);

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return isPotentialSlopeInput ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return isPotentialSlopeInput ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::AccumulateOp accumulateOp,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    auto outType = accumulateOp.getOutput().getType().cast<NDTypeInterface>();
    // Scale must have 1xCx1x1 shape, which is different from NxCxHxW output shape.
    const auto isScale = (outType != inputType);
    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        // VPU.Accumulate(lhs=[N,C,H,W], rhs=[N,C,H,W], lhsScale=[1,C,1,1], rhsScale=[1,C,1,1])
        // Split over height, where Y = H / clusters
        // VPU.Accumulate(lhs=[N,C,Y,W], rhs=[N,C,Y,W], lhsScale=[1,C,1,1], rhsScale=[1,C,1,1])
        // Split over width, where X = W / clusters
        // VPU.Accumulate(lhs=[N,C,H,X], rhs=[N,C,H,X], lhsScale=[1,C,1,1], rhsScale=[1,C,1,1])
        // lhs and rhs are segmented, scales are duplicated.
        return isScale ? DistributionMode::DUPLICATED : DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        // VPU.Accumulate(lhs=[N,C,H,W], rhs=[N,C,H,W], lhsScale=[1,C,1,1], rhsScale=[1,C,1,1])
        // Split over kernel, where Z = C / clusters
        // VPU.Accumulate(lhs=[N,Z,H,W], rhs=[N,Z,H,W], lhsScale=[1,Z,1,1], rhsScale=[1,Z,1,1])
        // All operands are segmented.
        return DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::DetectionOutputSortOp /*op*/,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface /*inputType*/) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return DistributionMode::SEGMENTED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MatMulOp /*op*/, VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface /*inputType*/) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::MVN1NormalizeOp op,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    auto meanVarType = op.getMeanVar().getType().cast<NDTypeInterface>();
    if (meanVarType == inputType) {
        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            return DistributionMode::SEGMENTED;
        }
        return DistributionMode::DUPLICATED;
    }

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    }
    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
        strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getSWInputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                             VPU::MultiClusterStrategy strategy,
                                                             vpux::NDTypeInterface inputType) {
    return llvm::TypeSwitch<mlir::Operation*, VPU::DistributionMode>(clusteredOp.getOperation())
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolateOp) {
                return getSWInputTensorDistributionMode(interpolateOp, strategy, inputType);
            })
            .Case<VPU::MultiplyOp, VPU::DivideOp, VPU::PowerOp, VPU::MaximumOp, VPU::MinimumOp, VPU::GreaterOp,
                  VPU::LessOp, VPU::SubtractOp, VPU::AddOp, VPU::FloorOp, VPU::FakeQuantizeOp, VPU::SelectOp>(
                    [&](mlir::Operation* eltwiseOp) {
                        return getSWInputTensorDistributionMode(eltwiseOp, strategy, inputType);
                    })
            .Case<VPU::PReluOp>([&](VPU::PReluOp preluOp) {
                return getSWInputTensorDistributionMode(preluOp, strategy, inputType);
            })
            .Case<VPU::AccumulateOp>([&](VPU::AccumulateOp accumulateOp) {
                return getSWInputTensorDistributionMode(accumulateOp, strategy, inputType);
            })
            .Case<VPU::DetectionOutputSortOp>([&](VPU::DetectionOutputSortOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::MatMulOp>([&](VPU::MatMulOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Case<VPU::MVN1NormalizeOp>([&](VPU::MVN1NormalizeOp op) {
                return getSWInputTensorDistributionMode(op, strategy, inputType);
            })
            .Default([&](mlir::Operation*) {
                VPUX_THROW_UNLESS(clusteredOp->getOperands().size() == 1,
                                  "General method only support SW layer with one operand but got '{0}'",
                                  clusteredOp->getOperands().size());
                return getSWInputTensorDistributionMode(strategy);
            });
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy) {
    return getActivationTensorNumTiles(clusteredOp, numClustersAvailableForCompilation, strategy);
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::InterpolateOp interpolateOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(interpolateOp, strategy, inputType);

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::MVN1NormalizeOp normalizeOpOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(normalizeOpOp, strategy, inputType);

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto IC = inputType.getShape()[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = std::min(numClustersAvailableForCompilation, IC);
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersToUseForLayer, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(mlir::Operation* eltwiseOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(eltwiseOp, strategy, inputType);

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto IC = inputType.getShape()[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = std::min(numClustersAvailableForCompilation, IC);
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, numClustersToUseForLayer, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return distributionMode == VPU::DistributionMode::DUPLICATED
                       ? SmallVector<int64_t>{1, 1, 1, 1}
                       : SmallVector<int64_t>{1, 1, 1, numClustersAvailableForCompilation};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::AccumulateOp accumulateOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return {1, 1, 1, 1};
    }
    const auto distributionMode = VPU::getSWInputTensorDistributionMode(accumulateOp, strategy, inputType);
    if (distributionMode == VPU::DistributionMode::DUPLICATED) {
        return {1, 1, 1, 1};
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        const auto height = inputType.getShape()[Dims4D::Act::H];
        const auto clusters = std::min(numClustersAvailableForCompilation, height);
        return {1, 1, clusters, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        const auto width = inputType.getShape()[Dims4D::Act::W];
        const auto clusters = std::min(numClustersAvailableForCompilation, width);
        return {1, 1, 1, clusters};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        const auto channels = inputType.getShape()[Dims4D::Act::C];
        const auto clusters = std::min(numClustersAvailableForCompilation, channels);
        return {1, clusters, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::DetectionOutputSortOp /*op*/,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface /*inputType*/) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return SmallVector<int64_t>{1, 1, numClustersAvailableForCompilation, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::MatMulOp /*op*/,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface /*inputType*/) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return SmallVector<int64_t>{1, numClustersAvailableForCompilation, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy,
                                                         vpux::NDTypeInterface inputType) {
    return llvm::TypeSwitch<mlir::Operation*, SmallVector<int64_t>>(clusteredOp.getOperation())
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolateOp) {
                return getSWInputTensorNumTiles(interpolateOp, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::MultiplyOp, VPU::DivideOp, VPU::PowerOp, VPU::MaximumOp, VPU::MinimumOp, VPU::PReluOp,
                  VPU::GreaterOp, VPU::LessOp, VPU::SubtractOp, VPU::AddOp, VPU::FloorOp, VPU::FakeQuantizeOp,
                  VPU::SelectOp>([&](mlir::Operation* eltwiseOp) {
                return getSWInputTensorNumTiles(eltwiseOp, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::AccumulateOp>([&](VPU::AccumulateOp accumulateOp) {
                return getSWInputTensorNumTiles(accumulateOp, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::DetectionOutputSortOp>([&](VPU::DetectionOutputSortOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::MatMulOp>([&](VPU::MatMulOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Case<VPU::MVN1NormalizeOp>([&](VPU::MVN1NormalizeOp op) {
                return getSWInputTensorNumTiles(op, numClustersAvailableForCompilation, strategy, inputType);
            })
            .Default([&](mlir::Operation*) {
                VPUX_THROW_UNLESS(clusteredOp->getOperands().size() == 1,
                                  "General method only support SW layer with one operand but got '{0}'",
                                  clusteredOp->getOperands().size());
                return getSWInputTensorNumTiles(clusteredOp, numClustersAvailableForCompilation, strategy);
            });
}

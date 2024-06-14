//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

static constexpr auto MODE_AUTO = "auto";
static constexpr auto MODE_TRUE = "true";
static constexpr auto MODE_FALSE = "false";

VPU::EnableActivationSparsityMode VPU::getActSparsityMode(std::string strMode) {
    std::transform(strMode.begin(), strMode.end(), strMode.begin(), ::tolower);

    if (strMode == MODE_AUTO) {
        return VPU::EnableActivationSparsityMode::AUTO;
    } else if (strMode == MODE_TRUE) {
        return VPU::EnableActivationSparsityMode::TRUE;
    } else if (strMode == MODE_FALSE) {
        return VPU::EnableActivationSparsityMode::FALSE;
    }

    VPUX_THROW("Unknown value for the enable activation sparsity option: {0}", strMode);
}

VPU::EnableActivationSparsityMode VPU::getActSparsityMode(const StrOption& enableActivationSparsityOption) {
    auto strOption = convertToOptional(enableActivationSparsityOption);
    if (!strOption.has_value()) {
        return VPU::EnableActivationSparsityMode::AUTO;
    }
    return getActSparsityMode(strOption.value());
}

bool VPU::isActSparsityEnabled(const StrOption& enableActivationSparsityOption) {
    const auto actSparsityMode = getActSparsityMode(enableActivationSparsityOption);
    return actSparsityMode == VPU::EnableActivationSparsityMode::TRUE ||
           actSparsityMode == VPU::EnableActivationSparsityMode::AUTO;
}

// Get the largest storage element size that is compatible with the given number of channels
// The storage element size must be a multiple of 16
// Example: if the number of channels is 48 and the sparsity constraint is for the storage element size to be a power of
// two, the returned value will be 16
int64_t VPU::getSESize(int64_t channels, const VPU::SparsityConstraint& sparsityConstraint) {
    for (int64_t seSize = channels; seSize >= 16; seSize -= 16) {
        if (channels % seSize == 0 && sparsityConstraint.areChannelsFitForSESize(seSize)) {
            return seSize;
        }
    }
    VPUX_THROW("Failed to find se_size for '{0}' channels", channels);
}

//
// shouldRemoveOutputSparsity
//

bool VPU::shouldRemoveOutputSparsity(VPU::NCEOpInterface nceOp) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusteredOp == nullptr || !clusteredOp.getMultiClusterStrategy().has_value()) {
        return false;
    }

    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    if (strategy != VPU::MultiClusterStrategy::SplitOverKernel) {
        return false;
    }

    const auto outputTensorType = clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto sparseOutputType = outputTensorType.dyn_cast<VPU::SparseTensorType>();
    if (sparseOutputType == nullptr) {
        return false;
    }

    VPUX_THROW_UNLESS(sparseOutputType.getSparsityMap() != nullptr, "Missing sparsity map from sparse type {0}",
                      sparseOutputType);
    VPUX_THROW_UNLESS(sparseOutputType.getStorageElementTable() == nullptr,
                      "Dynamically populated storage element table is not supported");

    const auto numClusters =
            VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    const auto distributedDataType = getDistributedOutputTypeFromOp(
            clusteredOp, sparseOutputType.getData(), numClusters,
            /*inputType*/ nullptr, /*tileInfo*/ vpux::TileInfo(ShapeRef()), /*hasExplicitDistributedAttr*/ false);

    // Removes SOK layer's output sparsity if SOK layer has different split sizes on clusters excluding the last
    // one. For example, we need to split OC = 128 on 6 tiles, the tiled size will be {32, 32, 16, 16, 16, 16}.
    // If there's output sparsity, we need to split 32 into two pieces of 16 because we must have the same
    // workload channel excluding the last one. However, two workloads with 16 channels have much worse
    // performance than a workload with 32 channels. If there's no sparsity, we can keep the workload with 32
    // channels.
    const auto distributedTensorType = mlir::cast<vpux::VPU::DistributedTensorType>(distributedDataType);
    if (distributedTensorType.getDistribution().getUniformDistributedSegments() != nullptr) {
        return true;
    }

    // Removes SOK layer's output sparsity if SOK layer's output is used by `VPU.Concat`.
    //
    // Conv1_1 (OC = 256, SOK)  Conv1_2 (OC = 256, SOK)
    //       \                               /
    //                   Concat on C
    //                        |
    //                      Conv2
    //
    // Take above graph as an example, we need to split OC = 256 on 6 tiles, the tiled size will be {48, 48, 48,
    // 48, 48, 16}. After concatenation, the combined workloads will be {48, 48, 48, 48, 48, 16, 48, 48, 48, 48,
    // 48, 16}. If there's output sparsity for Conv1_1 and Conv1_2, we need to split 48 into three pieces of 16
    // because we must have the same workload channel excluding the last one. If there's no sparsity, we can
    // keep the workload with 48 channels.
    auto users = to_small_vector(clusteredOp->getUsers());
    if (llvm::find_if(users, [](const mlir::Operation* op) {
            if (auto concatOp = mlir::dyn_cast_or_null<VPU::ConcatOp>(op)) {
                const auto outputType = concatOp.getOutput().getType().cast<NDTypeInterface>();
                const auto outputShape = outputType.getShape();
                const auto inputDataType = concatOp.getInputs().front().getType().cast<NDTypeInterface>();
                const auto inputShape = inputDataType.getShape();

                if (inputShape[Dims4D::Act::C] != outputShape[Dims4D::Act::C]) {
                    return true;
                }
            }

            return false;
        }) != users.end()) {
        return true;
    }
    return false;
}

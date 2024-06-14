//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>
#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {
namespace VPU {

enum class EnableActivationSparsityMode { AUTO, TRUE, FALSE };

EnableActivationSparsityMode getActSparsityMode(std::string enableActivationSparsityOption);
EnableActivationSparsityMode getActSparsityMode(const StrOption& enableActivationSparsityOption);
bool isActSparsityEnabled(const StrOption& enableActivationSparsityOption);

int64_t getSESize(int64_t channels, const VPU::SparsityConstraint& sparsityConstraint);

/*
    Effective sparse output type is the actual tensor IDU sees at its input after applying SETable over the data.

    For example, for a SEAttr with Interpolate Nearest with 2x2 scales, we'll have the following shapes for the sparse
    type components:
    data: [1, 16, 32, 32]
    sparsity_map: [1, 16, 64, 64]
    storage_element_table [1, 1, 64, 64]

    Effective output type will have shape: [1, 16, 64, 64]
*/
template <typename T, std::enable_if_t<std::disjunction_v<std::is_same<VPU::SparseTensorType, T>,
                                                          std::is_same<VPUIP::SparseBufferType, T>>,
                                       bool> = true>
NDTypeInterface getEffectiveSparseOutputType(T sparseType) {
    auto dataNDType = sparseType.getData().template cast<NDTypeInterface>();
    auto seTableType = sparseType.getStorageElementTable();

    if (seTableType == nullptr) {
        return dataNDType;
    }

    auto seTableNDType = seTableType.template cast<NDTypeInterface>();
    auto outShape = Shape(seTableNDType.getShape().raw());
    outShape[Dims4D::Act::N] = dataNDType.getShape()[Dims4D::Act::N];
    outShape[Dims4D::Act::C] = dataNDType.getShape()[Dims4D::Act::C];

    auto distributedTypeIf = sparseType.template cast<VPU::DistributedTypeInterface>();
    if (!distributedTypeIf.containsDistributedTypes()) {
        return dataNDType.changeShape(outShape);
    }

    auto getDistribution = [](mlir::Type componentType) -> VPU::DistributedTensorAttr {
        if (auto distributedTensor = componentType.dyn_cast<VPU::DistributedTensorType>()) {
            return distributedTensor.getDistribution();
        } else if (auto distributedBuffer = componentType.dyn_cast<VPUIP::DistributedBufferType>()) {
            return distributedBuffer.getDistribution();
        }

        VPUX_THROW("Sparse type's component is not distributed, component type = {0}", componentType);
    };

    auto dataDistribution = getDistribution(dataNDType);
    if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(dataDistribution)) {
        return dataNDType.changeShape(outShape);
    }

    auto distributionForEffectiveType = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
    return dataNDType.template cast<VPU::DistributedTypeInterface>().changeShapeForExplicitDistribution(
            outShape, distributionForEffectiveType);
}

bool shouldRemoveOutputSparsity(VPU::NCEOpInterface nceOp);

}  // namespace VPU
}  // namespace vpux

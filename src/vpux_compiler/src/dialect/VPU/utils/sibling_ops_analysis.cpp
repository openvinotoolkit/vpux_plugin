//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"

namespace vpux::VPU {
std::set<ClusteredOpInterface> SiblingOpsAnalysis::getOrLookupOpSiblings(mlir::Operation* op) {
    if (op == nullptr) {
        return {};
    }
    if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        for (const auto& siblingGroup : _siblingGroups) {
            if (siblingGroup.find(clusteredOp) != siblingGroup.end()) {
                return siblingGroup;
            }
        }
    }

    auto opSiblings = getSiblingOps(op);
    _siblingGroups.emplace_back(opSiblings);
    return opSiblings;
}

std::set<ClusteredOpInterface> SiblingOpsAnalysis::getSiblings(ClusteredOpInterface clusteredOp) {
    return getOrLookupOpSiblings(clusteredOp);
}

std::set<ClusteredOpInterface> SiblingOpsAnalysis::getConsumers(ClusteredOpInterface clusteredOp) {
    mlir::Operation* consumerOp = nullptr;
    if (isPassthroughOp(clusteredOp.getOperation())) {
        // For passthrough ops, ensure input and output tensors use the same pool of ops to
        // determine the distribution
        consumerOp = clusteredOp.getOperation();
    } else {
        for (const auto& consumer : clusteredOp->getUsers()) {
            // find first valid consumer and use it to get all its clustered siblings
            if (mlir::isa<ClusteredOpInterface>(consumer) || isPassthroughOp(consumer)) {
                consumerOp = consumer;
                break;
            }
        }
    }

    return getOrLookupOpSiblings(consumerOp);
}
}  // namespace vpux::VPU

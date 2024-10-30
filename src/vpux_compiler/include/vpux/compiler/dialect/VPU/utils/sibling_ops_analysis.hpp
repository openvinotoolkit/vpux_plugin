//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <set>
#include <vector>

namespace vpux::VPU {

class ClusteredOpInterface;

// Analysis which finds clustered op siblings and consumers set.
// Siblings and consumers are computed lazily and cached in _siblingGroups.
// Be careful not to introduce or remove clustered ops into IR when using this class, if
// siblings were already cached it might lead to missed ops or to invalid ops being returned.
class SiblingOpsAnalysis {
public:
    SiblingOpsAnalysis(mlir::Operation*){};

    // Get clustered ops that are siblings to the passed clustered op.
    std::set<ClusteredOpInterface> getSiblings(ClusteredOpInterface);

    // Get clustered ops that consume result of the passed clustered op.
    // Clustered op is considered to be a consumer if it consumes result directly
    // or if it consumes a result of the view-like op which in turn consumes result
    // of the passed op like in following pattern:
    // ClusuteredOp(producer) -> View-like op -> ClusteredOp(consumer)
    std::set<ClusteredOpInterface> getConsumers(ClusteredOpInterface);

private:
    std::set<ClusteredOpInterface> getOrLookupOpSiblings(mlir::Operation* op);

    std::vector<std::set<ClusteredOpInterface>> _siblingGroups{};
};
}  // namespace vpux::VPU

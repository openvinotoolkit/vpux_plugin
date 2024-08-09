//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux::VPU::arch37xx {

/*
   Class for getting MakeOpsWithDistributedTensorStrategy patterns for NPU37XX
*/
class MakeOpsWithDistributedTensorStrategy : public IGreedilyPassStrategy {
public:
    MakeOpsWithDistributedTensorStrategy(llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                                         bool enableExplicitDistributedTensorAttr)
            : _overlapParamsLookup(overlapParamsLookup),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr) {
    }
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;

private:
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
    bool _enableExplicitDistributedTensorAttr = false;
};

}  // namespace vpux::VPU::arch37xx

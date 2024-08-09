//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class ViewTaskRangeRewriter : public VPUASMSymbolizationPattern<VPURegMapped::ViewTaskRangeOp> {
public:
    using Base::Base;
    mlir::LogicalResult symbolize(VPURegMapped::ViewTaskRangeOp op, SymbolMapper& mapper,
                                  mlir::ConversionPatternRewriter& rewriter) const override;
    llvm::SmallVector<mlir::FlatSymbolRefAttr> getSymbolicNames(VPURegMapped::ViewTaskRangeOp op,
                                                                size_t counter) override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux

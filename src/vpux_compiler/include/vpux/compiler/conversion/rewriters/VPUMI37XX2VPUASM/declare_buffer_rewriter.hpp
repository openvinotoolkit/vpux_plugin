//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_pattern.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

class DeclareBufferRewriter : public VPUASMSymbolizationPattern<VPURT::DeclareBufferOp> {
public:
    using Base::Base;
    mlir::LogicalResult symbolize(VPURT::DeclareBufferOp op, SymbolMapper& mapper,
                                  mlir::ConversionPatternRewriter& rewriter) const override;
    llvm::SmallVector<mlir::FlatSymbolRefAttr> getSymbolicNames(VPURT::DeclareBufferOp op, size_t counter) override;
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux

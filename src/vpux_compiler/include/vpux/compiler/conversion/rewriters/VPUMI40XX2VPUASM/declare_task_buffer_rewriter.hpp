//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class DeclareTaskBufferRewriter : public VPUASMSymbolizationPattern<VPURegMapped::DeclareTaskBufferOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(VPURegMapped::DeclareTaskBufferOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
    llvm::SmallVector<mlir::FlatSymbolRefAttr> getSymbolicNames(VPURegMapped::DeclareTaskBufferOp op,
                                                                size_t counter) override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux

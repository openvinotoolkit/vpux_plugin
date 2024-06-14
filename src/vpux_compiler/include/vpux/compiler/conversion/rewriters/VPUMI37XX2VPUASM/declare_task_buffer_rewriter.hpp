//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

class DeclareTaskBufferRewriter : public VPUASMSymbolizationPattern<VPURegMapped::DeclareTaskBufferOp> {
public:
    using Base::Base;
    mlir::LogicalResult symbolize(VPURegMapped::DeclareTaskBufferOp op, SymbolMapper& mapper,
                                  mlir::ConversionPatternRewriter& rewriter) const override;
    mlir::FlatSymbolRefAttr getSymbolicName(VPURegMapped::DeclareTaskBufferOp op, size_t counter) override;
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

class KernelInvocationRewriter : public VPUASMSymbolizationPattern<VPUMI37XX::ActKernelInvocationOp> {
public:
    using Base::Base;
    mlir::LogicalResult symbolize(VPUMI37XX::ActKernelInvocationOp op, SymbolMapper& mapper,
                                  mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux

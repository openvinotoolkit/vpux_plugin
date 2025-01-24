//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

class KernelParamsRewriter : public VPUASMSymbolizationPattern<VPUMI37XX::KernelParamsOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(VPUMI37XX::KernelParamsOp op, SymbolMapper& mappper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux

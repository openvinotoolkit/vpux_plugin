//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class DPUInvariantRewriter : public VPUASMSymbolizationPattern<VPUMI40XX::DPUInvariantOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(VPUMI40XX::DPUInvariantOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux

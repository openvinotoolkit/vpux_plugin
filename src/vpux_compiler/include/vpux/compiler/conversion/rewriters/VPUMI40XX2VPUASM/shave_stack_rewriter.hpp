//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class ShaveStackRewriter : public VPUASMSymbolizationPattern<VPUMI40XX::ShaveStackFrameOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(VPUMI40XX::ShaveStackFrameOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class BarrierRewriter : public VPUASMSymbolizationPattern<VPUMI40XX::ConfigureBarrierOp> {
public:
    using Base::Base;
    mlir::LogicalResult symbolize(VPUMI40XX::ConfigureBarrierOp op, SymbolMapper& mapper,
                                  mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux

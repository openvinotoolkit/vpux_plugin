//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class NNDMARewriter : public VPUASMSymbolizationPattern<VPUMI40XX::NNDMAOp> {
public:
    using Base::Base;
    mlir::LogicalResult symbolize(VPUMI40XX::NNDMAOp op, SymbolMapper& mapper,
                                  mlir::ConversionPatternRewriter& rewriter) const override;
    mlir::FlatSymbolRefAttr getSymbolicName(VPUMI40XX::NNDMAOp op, size_t) override;

private:
    VPUIP::DMADescriptorAttr getDmaTransactionTraits(VPUMI40XX::NNDMAOp op, mlir::MLIRContext* ctx) const;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux

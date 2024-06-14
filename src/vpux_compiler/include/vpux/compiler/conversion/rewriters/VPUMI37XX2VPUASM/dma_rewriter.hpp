//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

class NNDMARewriter : public VPUASMSymbolizationPattern<VPUMI37XX::NNDMAOp> {
public:
    using Base::Base;
    mlir::LogicalResult symbolize(VPUMI37XX::NNDMAOp op, SymbolMapper& mapper,
                                  mlir::ConversionPatternRewriter& rewriter) const override;
    mlir::FlatSymbolRefAttr getSymbolicName(VPUMI37XX::NNDMAOp op, size_t) override;

private:
    // E#36225: can we have this reduction at a pass at memref level? Need to place
    // some conditions on the DMA, and in some scenarios, may have to do 1*DMA -> n*DMA
    //      transaction rewrites.
    // Please refer to the ticket E#36225.
    static llvm::SmallVector<std::pair<uint32_t, int32_t>> reduce_dims_for_dma(mlir::Value val);
    VPUIP::DMADescriptorAttr getDmaTransactionTraits(VPUMI37XX::NNDMAOp op, mlir::MLIRContext* ctx) const;
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux

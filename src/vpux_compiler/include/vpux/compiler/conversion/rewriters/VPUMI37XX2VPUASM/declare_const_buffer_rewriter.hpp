//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

class DeclareConstBufferRewriter : public VPUASMSymbolizationPattern<Const::DeclareOp> {
public:
    using Base::Base;
    mlir::LogicalResult symbolize(Const::DeclareOp op, SymbolMapper& mapper,
                                  mlir::ConversionPatternRewriter& rewriter) const override;
    mlir::FlatSymbolRefAttr getSymbolicName(Const::DeclareOp op, size_t counter) override;
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux

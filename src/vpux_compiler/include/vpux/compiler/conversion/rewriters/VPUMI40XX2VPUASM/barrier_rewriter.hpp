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

    BarrierRewriter(mlir::func::FuncOp netFunc, SymbolizationTypeConverter& typeConverter, SymbolMapper& mapper,
                    SectionMapper& sectionMap, mlir::MLIRContext* ctx, Logger log, bool enablePWLM)
            : VPUASMSymbolizationPattern<VPUMI40XX::ConfigureBarrierOp>(netFunc, typeConverter, mapper, sectionMap, ctx,
                                                                        log),
              _enablePWLM(enablePWLM) {
    }

    mlir::FailureOr<SymbolizationResult> symbolize(VPUMI40XX::ConfigureBarrierOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;

private:
    bool _enablePWLM;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux

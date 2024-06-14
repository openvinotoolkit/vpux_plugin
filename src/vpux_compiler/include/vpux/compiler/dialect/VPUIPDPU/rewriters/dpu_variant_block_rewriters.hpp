//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

namespace vpux {
namespace VPUIPDPU {

class DPUVariantBlockRewriter {
public:
    DPUVariantBlockRewriter(VPUASM::DPUVariantOp origVarOp, mlir::PatternRewriter& rewriter, const Logger& log);

protected:
    VPUASM::DPUVariantOp _origVarOp;
    mlir::PatternRewriter& _rewriter;
    const Logger _log;
};

class DPUVariantIDURewriter : public DPUVariantBlockRewriter {
public:
    DPUVariantIDURewriter(VPUASM::DPUVariantOp origVarOp, mlir::PatternRewriter& rewriter, const Logger& log);

    mlir::LogicalResult rewrite(ELF::SymbolReferenceMap& symRefMap);
};

class DPUVariantPPERewriter : public DPUVariantBlockRewriter {
public:
    DPUVariantPPERewriter(VPUASM::DPUVariantOp origVarOp, mlir::PatternRewriter& rewriter, const Logger& log);

    mlir::LogicalResult rewrite();
};

class DPUVariantODURewriter : public DPUVariantBlockRewriter {
public:
    DPUVariantODURewriter(VPUASM::DPUVariantOp origVarOp, mlir::PatternRewriter& rewriter, const Logger& log);

    mlir::LogicalResult rewrite(ELF::SymbolReferenceMap& symRefMap);
};

}  // namespace VPUIPDPU
}  // namespace vpux

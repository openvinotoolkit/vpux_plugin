//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

namespace vpux {
namespace VPUIPDPU {

class DPUVariantRewriter final : public mlir::OpRewritePattern<VPUASM::DPUVariantOp> {
public:
    DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap);

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::DPUVariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

}  // namespace VPUIPDPU
}  // namespace vpux

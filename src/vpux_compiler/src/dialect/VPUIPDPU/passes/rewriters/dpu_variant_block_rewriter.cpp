//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_variant_block_rewriters.hpp"

namespace vpux {
namespace VPUIPDPU {

DPUVariantBlockRewriter::DPUVariantBlockRewriter(VPUASM::DPUVariantOp origVarOp, mlir::PatternRewriter& rewriter,
                                                 const Logger& log)
        : _origVarOp(origVarOp), _rewriter(rewriter), _log(log) {
}

}  // namespace VPUIPDPU
}  // namespace vpux

//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

#include <functional>

using namespace vpux;
using namespace vpux::VPUIPDPU;
using namespace mlir;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.cpp.inc>

//
// Custom
//

namespace vpux {
namespace VPUIPDPU {

mlir::LogicalResult ODUWriteCombineBufferOp::verify() {
    auto sparsityModeExists = getSparsityMode().has_value();

    if (!sparsityModeExists) {
        return ::mlir::success();
    }

    auto& parentEntryBlock = getOperation()->getParentOp()->getRegion(0).getBlocks().front();
    auto sparsityOp = parentEntryBlock.op_begin<ODUSparsityOp>();
    if (sparsityOp != parentEntryBlock.op_end<ODUSparsityOp>()) {
        auto sparsityMapExists = ((*sparsityOp).getSparsityMap() != nullptr);

        if (sparsityMapExists) {
            return ::mlir::success();
        }
    }

    return errorAt(getLoc(),
                   "Operation {0}: sparsity_mode should only exist in case a sibbling ODUSparsityOp exists with "
                   "sparsity_map param set",
                   getOperationName());
}

}  // namespace VPUIPDPU
}  // namespace vpux

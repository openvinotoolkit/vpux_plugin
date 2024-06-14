//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/attributes.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/traits_utils.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
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
#include <vpux/compiler/dialect/VPUIPDPU/ops.cpp.inc>

//
// Custom
//

namespace vpux {
namespace VPUIPDPU {

mlir::LogicalResult DPUInvariantOp::verify() {
    if (!hasMandatorySingleInstanceChildren<DPUInvariantOp, IDUCfgOp, PPECfgOp, ODUCfgOp>(*this)) {
        return errorAt(getLoc(), "Operation {0}: missing mandatory child ops", getOperationName());
    }
    if (!hasOptionalSingleInstanceChildren<DPUInvariantOp, MPECfgOp>(*this)) {
        return errorAt(getLoc(), "Operation {0}: too many optional child ops", getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult DPUVariantOp::verify() {
    if (!hasMandatorySingleInstanceChildren<DPUVariantOp, ODUOutSubtensorOp>(*this)) {
        return errorAt(getLoc(), "Operation {0}: missing mandatory child ops", getOperationName());
    }

    // NPU40XX supports up to 5 halo regions
    if (getEntryBlockSize<ODUHaloRegionOp>(getOperation()) > 5) {
        return errorAt(getLoc(), "Operation {0}: too many halo regions defined", getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult ODUCfgOp::verify() {
    if (!hasMandatorySingleInstanceChildren<ODUCfgOp, ODUOutTensorSizeOp, ODUOutActivationsOp>(*this)) {
        return errorAt(getLoc(), "Operation {0}: missing mandatory child ops", getOperationName());
    }

    if (!hasOptionalSingleInstanceChildren<ODUCfgOp, ODUDataReuseOp, ODUPermuteDataOp, ODUSparsityOp, ODUSwizzleDataOp,
                                           ODUMemoryModeOp, ODUCmxPortsOp, ODUWriteCombineBufferOp>(*this)) {
        return errorAt(getLoc(), "Operation {0}: too many optional child ops", getOperationName());
    }

    // NPU37XX  supports up to 3 cast instances
    if (getEntryBlockSize<ODUCastOp>(getOperation()) > 3) {
        return errorAt(getLoc(), "Operation {0}: too many cast instances defined", getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult MPECfgOp::verify() {
    if (!hasOptionalSingleInstanceChildren<MPECfgOp, MPEDenormalOperandsFTZOp, MPEActivationBiasOp, MPEWeightsBiasOp>(
                *this)) {
        return errorAt(getLoc(), "Operation {0}: too many optional child ops", getOperationName());
    }

    return ::mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux

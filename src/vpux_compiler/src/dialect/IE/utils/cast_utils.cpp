//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/cast_utils.hpp"

namespace vpux {

mlir::LogicalResult isQuantizeCastValid(mlir::Location loc, mlir::Type srcType, mlir::Type dstType) {
    const auto srcBitSize = vpux::getElemTypeSize(srcType).count();
    const auto dstBitSize = vpux::getElemTypeSize(dstType).count();

    if (srcBitSize != dstBitSize) {
        return errorAt(loc, "Not matching bit size of src type {0} and dst type {1}", srcBitSize, dstBitSize);
    }

    // Admitting all cases except I1, as currently we're treating it as pure I1 data type
    // not requiring any quant cast
    if (srcBitSize < 2 || !vpux::isPowerOfTwo(srcBitSize)) {
        return errorAt(loc, "Type to be casted size in bits is not a power of two: {0}", srcBitSize);
    }

    if (!(mlir::isa<mlir::quant::QuantizedType>(srcType) || mlir::isa<mlir::quant::QuantizedType>(dstType))) {
        return errorAt(loc, "At least one of src and dst should be quantized type");
    }

    if (srcType.isa<mlir::FloatType>() || dstType.isa<mlir::FloatType>()) {
        return errorAt(loc, "Don't admit float operand and results");
    }

    return mlir::success();
}

}  // namespace vpux

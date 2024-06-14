//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::DistributedCastOp::getViewSource() {
    return getInput();
}

//
// fold
//

mlir::OpFoldResult VPUIP::DistributedCastOp::fold(FoldAdaptor) {
    return getInput().getType() == getOutput().getType() ? getInput() : mlir::TypedValue<mlir::MemRefType>{nullptr};
}

//
// verify
//

mlir::LogicalResult vpux::VPUIP::DistributedCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    if (auto sparseBufferInput = getInput().getType().dyn_cast<VPUIP::SparseBufferType>()) {
        if (auto sparseBufferOutput = getOutput().getType().dyn_cast<VPUIP::SparseBufferType>()) {
            const auto inputData = sparseBufferInput.getData().cast<VPUIP::DistributedBufferType>();
            const auto outputData = sparseBufferOutput.getData().cast<VPUIP::DistributedBufferType>();
            return VPU::isDistributedCastCompatible(inputData, outputData, logCb);
        }

        logCb(formatv("Mismatch between types for input and output. "
                      "If input is SparseBufferType then output must be of same type."));
        return mlir::failure();
    }

    const auto outType = getOutput().getType();
    if (mlir::isa<VPUIP::SparseBufferType>(outType)) {
        logCb(formatv("Mismatch between types for input and output. "
                      "If output is SparseBufferType then input must be of same type."));
        return mlir::failure();
    }

    const auto inDistributedType = getInput().getType().cast<VPUIP::DistributedBufferType>();
    const auto outDistributedType = getOutput().getType().cast<VPUIP::DistributedBufferType>();

    return VPU::isDistributedCastCompatible(inDistributedType, outDistributedType, logCb);
}

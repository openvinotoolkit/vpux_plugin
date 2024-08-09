//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/VPUMI40XX/ops.hpp>
#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>

using namespace vpux;
using namespace VPUMI40XX;

//
// NNDMAOp
//

void NNDMAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, VPURegMapped::IndexType index,
                    mlir::Value taskLocation, mlir::Value input, mlir::Value output_buff, mlir::Value previousDma,
                    mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                    VPUMI40XX::DMATransactionAttr dma_transaction, VPUIP::DMADescriptorAttr dma_descriptor) {
    build(odsBuilder, odsState, index, taskLocation, input, mlir::ValueRange(output_buff), previousDma, waitBarriers,
          updateBarriers, 0, 0, false, false, false, 0, VPUIP::DMAAccMode::DISABLE,
          /*act_compression_size_entry*/ nullptr, /*act_compression_sparsity_map*/ nullptr, dma_transaction,
          dma_descriptor, 0, nullptr, 0, nullptr);
}

mlir::LogicalResult NNDMAOp::verify() {
    if (!isConfigureBarrierOpType(getWaitBarriers())) {
        return errorAt(getLoc(), "Operation {0}: waitBarriers should be of type ConfigureBarrier ", getOperationName());
    }

    if (!isConfigureBarrierOpType(getUpdateBarriers())) {
        return errorAt(getLoc(), "Operation {0}: updateBarriers should be of type ConfigureBarrier ",
                       getOperationName());
    }

    const auto currentDMAIdx = getIndex().getType().cast<VPURegMapped::IndexType>();
    const auto prevDMAIdx = VPUMI40XX::NNDMAOp::getPreviousTask()
                                    ? VPUMI40XX::NNDMAOp::getPreviousTask().getType().cast<VPURegMapped::IndexType>()
                                    : nullptr;
    if (prevDMAIdx) {
        if (prevDMAIdx.getTileIdx() != currentDMAIdx.getTileIdx()) {
            return errorAt(getLoc(), "Operation {0}: tileIndex for previousDMA {1} and currentDma {2} do not match ",
                           getOperationName(), prevDMAIdx.getTileIdx(), currentDMAIdx.getTileIdx());
        }
        if (prevDMAIdx.getListIdx() != currentDMAIdx.getListIdx()) {
            return errorAt(getLoc(), "Operation {0}: listIndex for previousDMA {1} and currentDma {2} do not match ",
                           getOperationName(), prevDMAIdx.getListIdx(), currentDMAIdx.getListIdx());
        }
        if (prevDMAIdx.getValue() != currentDMAIdx.getValue() - 1) {
            return errorAt(getLoc(), "Operation {0}: index for previousDMA {1} and currentDma {2} are not consecutive ",
                           getOperationName(), prevDMAIdx.getValue(), currentDMAIdx.getValue());
        }
    }

    auto inputShape = getShape(getInput());
    auto outputShape = getShape(getOutputBuffs()[0]);

    if (!getAllowDifferentInOutShapes() && (inputShape != outputShape)) {
        return errorAt(getLoc(),
                       "Input and output shapes are not equal: {0} != {1}. Not equal shapes allowed in case "
                       "getAccelerationMode() and getAllowDifferentInOutShapes() enabled only.",
                       inputShape, outputShape);
    }
    return ::mlir::success();
}

//
// Dot Printer
//

DotNodeColor NNDMAOp::getNodeColor() {
    return DotNodeColor::BLUE;
}

bool NNDMAOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head, llvm::StringRef middle,
                              llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    return true;
}

DOT::EdgeDir NNDMAOp::getEdgeDirection(mlir::Operation* source) {
    if (checkBarrierProductionRelationship(source, mlir::cast<ExecutableTaskOpInterface>(getOperation()))) {
        return DOT::EdgeDir::EDGE_REVERSE;
    }
    return DOT::EdgeDir::EDGE_NORMAL;
}

bool NNDMAOp::supportsHardLink() {
    return true;
}

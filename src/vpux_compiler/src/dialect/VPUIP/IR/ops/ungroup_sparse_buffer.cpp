//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

//
// build
//

void VPUIP::UngroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type data,
                                         mlir::Type sparsityMap, mlir::Type storageElementTable, mlir::Value input) {
    state.addOperands(input);
    state.addTypes(data);
    if (sparsityMap) {
        state.addTypes(sparsityMap);
    }
    if (storageElementTable) {
        state.addTypes(storageElementTable);
    }
    state.addAttribute(getResultSegmentSizesAttrName(state.name),
                       builder.getDenseI32ArrayAttr({1, (sparsityMap ? 1 : 0), (storageElementTable ? 1 : 0)}));
}

void VPUIP::UngroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input) {
    state.addOperands(input);

    SmallVector<mlir::Type, 2> inferredReturnTypes;
    if (mlir::succeeded(UngroupSparseBufferOp::inferReturnTypes(builder.getContext(), state.location, state.operands,
                                                                state.attributes.getDictionary(state.getContext()),
                                                                state.getRawProperties(),
                                                                /*regions=*/{}, inferredReturnTypes))) {
        state.addTypes(inferredReturnTypes);
    } else {
        VPUX_THROW("Failed to infer result types for {0}", state.name);
    }

    const auto sparseInput = input.getType().dyn_cast<VPUIP::SparseBufferType>();
    VPUX_THROW_WHEN(sparseInput == nullptr, "Input type ({0}) is not a sparse buffer", input.getType());
    const int32_t sparsityMapAttrVal = (sparseInput.getSparsityMap() != nullptr) ? 1 : 0;
    const int32_t seTableAttrVal = (sparseInput.getStorageElementTable() != nullptr) ? 1 : 0;
    state.addAttribute(getResultSegmentSizesAttrName(state.name),
                       builder.getDenseI32ArrayAttr({1, sparsityMapAttrVal, seTableAttrVal}));
}

//
// getViewSource
//

mlir::Value VPUIP::UngroupSparseBufferOp::getViewSource(ptrdiff_t /*resultInd*/) {
    return getInput();
}

//
// inferReturnTypes
//

mlir::LogicalResult VPUIP::UngroupSparseBufferOp::inferReturnTypes(mlir::MLIRContext*, std::optional<mlir::Location>,
                                                                   mlir::ValueRange operands,
                                                                   mlir::DictionaryAttr /*attrs*/,
                                                                   mlir::OpaqueProperties, mlir::RegionRange /*ranges*/,
                                                                   SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPUX_THROW_UNLESS(operands[0].getType().isa<VPUIP::SparseBufferType>(),
                      "Operand of type {0} is not a sparse buffer", operands[0].getType());
    const auto sparseBufferType = operands[0].getType().cast<VPUIP::SparseBufferType>();

    inferredReturnTypes.push_back(sparseBufferType.getData());
    if (sparseBufferType.getSparsityMap() != nullptr) {
        inferredReturnTypes.push_back(sparseBufferType.getSparsityMap());
    }
    if (sparseBufferType.getStorageElementTable() != nullptr) {
        inferredReturnTypes.push_back(sparseBufferType.getStorageElementTable());
    }

    return mlir::success();
}

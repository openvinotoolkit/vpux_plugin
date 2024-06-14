//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

#include <mlir/IR/BuiltinTypes.h>

vpux::VPUIP::BlobWriter::SpecificTask vpux::VPUIP::LSTMSequenceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::LSTMCellParamsBuilder builder(writer);
    builder.add_RNNForward(getDirection() == IE::RNNSequenceDirection::FORWARD ? 1 : 0);
    builder.add_nCells(checked_cast<int32_t>(getSequenceLength()));
    const auto inputDataShape = getInputData().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    VPUX_THROW_UNLESS(inputDataShape.size() == 3, "LSTMSequenceUPAOp inputData shape must be 3D");
    const auto batchSize = inputDataShape[0];
    builder.add_nBatches(checked_cast<int32_t>(batchSize));
    builder.add_useCellState(1);
    builder.add_outputsNumber(3);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_LSTMCellParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseLSTMSequence(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 5, "LSTMSequence supports only 5 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 3, "LSTMSequence supports only 3 output, got {0}", outputs.size());

    const auto params = task->softLayerParams_as_LSTMCellParams();

    const auto direction = params->RNNForward() ? IE::RNNSequenceDirection::FORWARD : IE::RNNSequenceDirection::REVERSE;
    const auto directionAttr = IE::RNNSequenceDirectionAttr::get(_ctx, direction);
    const auto seqLengthAttr = getIntAttr(_ctx, params->nCells());

    return builder.create<VPUIP::LSTMSequenceUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2],
                                                    inputs[3], inputs[4], outputs[0], outputs[1], outputs[2],
                                                    seqLengthAttr, directionAttr);
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

vpux::VPUIP::BlobWriter::SpecificTask vpux::VPUIP::LSTMCellUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto inputDataShape = getInputData().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    VPUX_THROW_UNLESS(inputDataShape.size() == 2, "LSTMCellUPAOp inputData shape must be 2D");

    const auto batchSize = inputDataShape[0];

    MVCNN::LSTMCellParamsBuilder builder(writer);
    builder.add_RNNForward(1);
    builder.add_nCells(1);
    builder.add_nBatches(static_cast<int32_t>(batchSize));
    builder.add_useCellState(1);
    builder.add_outputsNumber(2);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_LSTMCellParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseLSTMCell(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                        ArrayRef<mlir::Value> outputs,
                                                        const MVCNN::UPALayerTask* /*task*/) {
    VPUX_THROW_UNLESS(inputs.size() == 5, "LSTMCell supports only 5 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 2, "LSTMCell supports only 2 output, got {0}", outputs.size());

    return builder.create<VPUIP::LSTMCellUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2], inputs[3],
                                                inputs[4], outputs[1], outputs[2]);
}

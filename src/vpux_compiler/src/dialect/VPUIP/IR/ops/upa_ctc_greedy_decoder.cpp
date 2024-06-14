//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::CTCGreedyDecoderUPAOp::verify() {
    const auto op = getOperation();
    const auto inShape = getShape(getInput());

    if (inShape.size() != 3) {
        return errorAt(op, "Input shape should have 3 dimensions");
    }

    if (inShape[Dim(1)] != 1) {
        return errorAt(op, "Input tensor [T N C] = [{0} {1} {2}] has unsupported dimension size N != 1",
                       inShape[Dim(0)], inShape[Dim(1)], inShape[Dim(2)]);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CTCGreedyDecoderUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CTCDecoderParamsBuilder builder(writer);
    builder.add_ctc_merge_repeated(getMergeRepeated());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CTCDecoderParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseCTCGreedyDecoder(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                                ArrayRef<mlir::Value> outputs,
                                                                const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPACTCGreedyDecoder supports only 2 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPACTCGreedyDecoder supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_CTCDecoderParams();
    return builder.create<VPUIP::CTCGreedyDecoderUPAOp>(
            mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
            params->ctc_merge_repeated() ? mlir::UnitAttr::get(_ctx) : nullptr);
}

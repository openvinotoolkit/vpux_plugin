//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::ReverseSequenceUPAOp::verify() {
    const auto seqShape = getShape(getSeqLength());

    if (seqShape.size() != 1) {
        return errorAt(*this, "ReverseSequence second input should be 1D Tensor");
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReverseSequenceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ReversesequenceParamsBuilder builder(writer);
    builder.add_seq_axis(checked_cast<int32_t>(getSeqAxis()));
    builder.add_batch_axis(checked_cast<int32_t>(getBatchAxis()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReversesequenceParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseReverseSequence(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                               ArrayRef<mlir::Value> outputs,
                                                               const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPAReverseSequence supports only 2 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAReverseSequence supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_ReversesequenceParams();
    const auto seq_axis = getIntAttr(_ctx, params->seq_axis());
    const auto batch_axis = getIntAttr(_ctx, params->batch_axis());
    return builder.create<VPUIP::ReverseSequenceUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                       seq_axis, batch_axis);
}

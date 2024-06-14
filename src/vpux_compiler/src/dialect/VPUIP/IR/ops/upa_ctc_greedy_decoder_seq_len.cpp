//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CTCGreedyDecoderSeqLenUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CTCGreedyDecoderSeqLenParamsBuilder builder(writer);
    builder.add_mergeRepeated(getMergeRepeated());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this,
                                     {paramsOff.Union(), MVCNN::SoftwareLayerParams_CTCGreedyDecoderSeqLenParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseCTCGreedyDecoderSeqLen(mlir::OpBuilder& builder,
                                                                      ArrayRef<mlir::Value> inputs,
                                                                      ArrayRef<mlir::Value> outputs,
                                                                      const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 3, "UPACTCGreedyDecoderSeqLen supports only 3 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 2, "UPACTCGreedyDecoderSeqLen supports only 2 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_CTCGreedyDecoderSeqLenParams();
    return builder.create<VPUIP::CTCGreedyDecoderSeqLenUPAOp>(
            mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2], outputs[0], outputs[1],
            params->mergeRepeated() ? mlir::UnitAttr::get(_ctx) : nullptr);
}

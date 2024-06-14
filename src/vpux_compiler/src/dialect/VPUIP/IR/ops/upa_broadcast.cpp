//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BroadcastUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::BroadcastParamsBuilder builder(writer);

    MVCNN::BroadcastMode mvcnn_mode;

    if (this->getMode() == IE::BroadcastType::NUMPY) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_NUMPY;
    } else if (this->getMode() == IE::BroadcastType::BIDIRECTIONAL) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_BIDIRECTIONAL;
    } else if (this->getMode() == IE::BroadcastType::EXPLICIT) {
        mvcnn_mode = MVCNN::BroadcastMode::BroadcastMode_EXPLICIT;
    } else {
        VPUX_THROW("Unsupported broadcast mode {0}", this->getMode());
    }

    builder.add_mode(mvcnn_mode);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_BroadcastParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseBroadcast(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                         ArrayRef<mlir::Value> outputs,
                                                         const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3, "BroadcastUPA supports only 2 or 3 inputs, got {0}",
                      inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "BroadcastUPA supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_BroadcastParams();
    const auto mvcnn_mode = params->mode();

    IE::BroadcastType mode;
    if (mvcnn_mode == MVCNN::BroadcastMode::BroadcastMode_NUMPY) {
        mode = IE::BroadcastType::NUMPY;
    } else if (mvcnn_mode == MVCNN::BroadcastMode::BroadcastMode_BIDIRECTIONAL) {
        mode = IE::BroadcastType::BIDIRECTIONAL;
    } else if (mvcnn_mode == MVCNN::BroadcastMode::BroadcastMode_EXPLICIT) {
        mode = IE::BroadcastType::EXPLICIT;
    } else {
        VPUX_THROW("Unsupported broadcast mode {0}", mvcnn_mode);
    }

    return builder.create<VPUIP::BroadcastUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1],
                                                 inputs.size() == 3 ? inputs[2] : nullptr, outputs[0],
                                                 IE::BroadcastTypeAttr::get(_ctx, mode));
}

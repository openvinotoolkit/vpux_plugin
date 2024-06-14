//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReorgYoloUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ReorgYOLOParamsBuilder builder(writer);
    builder.add_stride(checked_cast<uint32_t>(getStride()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReorgYOLOParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseReorgYolo(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                         ArrayRef<mlir::Value> outputs,
                                                         const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "ReorgYoloUPA supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "ReorgYoloUPA supports only 1 output, got {0}", outputs.size());

    const auto params = task->softLayerParams_as_ReorgYOLOParams();
    const auto stride = getIntAttr(_ctx, params->stride());
    return builder.create<VPUIP::ReorgYoloUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], stride);
}

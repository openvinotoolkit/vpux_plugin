//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::GatherNDUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::GatherNDParamsBuilder builder(writer);
    builder.add_batch_dims(checked_cast<uint32_t>(getBatchDims()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GatherNDParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseGatherND(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                        ArrayRef<mlir::Value> outputs,
                                                        const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "GatherNDUPA supports only 2 inputs", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "GatherNDUPA supports only 1 output", outputs.size());

    const auto params = task->softLayerParams_as_GatherNDParams();
    const auto batch_dims = getIntAttr(_ctx, params->batch_dims());
    return builder.create<VPUIP::GatherNDUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                batch_dims);
}

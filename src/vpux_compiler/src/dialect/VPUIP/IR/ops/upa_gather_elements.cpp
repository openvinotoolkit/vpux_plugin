//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::GatherElementsUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::GatherElementsParamsBuilder builder(writer);
    builder.add_axis(checked_cast<int32_t>(getAxis()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GatherElementsParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseGatherElements(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                              ArrayRef<mlir::Value> outputs,
                                                              const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "GatherElementsUPA supports only 2 inputs", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "GatherElementsUPA supports only 1 output", outputs.size());

    const auto params = task->softLayerParams_as_GatherElementsParams();
    const auto axis = getIntAttr(_ctx, params->axis());
    return builder.create<VPUIP::GatherElementsUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                      axis);
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::GatherUPAOp::verify() {
    // Axis should not exceed input rank
    const auto axisNo = getAxis();
    const auto inShape = getShape(getInput());
    if (checked_cast<size_t>(axisNo) >= inShape.size()) {
        return errorAt(*this, "Gather axis '{0}' is out of range [0,{1}]", axisNo, inShape.size() - 1);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::GatherUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::GatherParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(getAxis()));
    builder.add_batch_dims(checked_cast<uint32_t>(getBatchDims()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GatherParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseGather(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                      ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "GatherUPA supports only 2 inputs", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "GatherUPA supports only 1 output", outputs.size());

    const auto params = task->softLayerParams_as_GatherParams();
    const auto axis = getIntAttr(_ctx, params->axis());
    const auto batch_dims = getIntAttr(_ctx, params->batch_dims());
    return builder.create<VPUIP::GatherUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0], axis,
                                              batch_dims);
}

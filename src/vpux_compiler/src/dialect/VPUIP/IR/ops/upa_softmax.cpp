//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::SoftMaxUPAOp::verify() {
    const auto op = getOperation();
    const auto inShape = getShape(getInput());
    const auto axis = Dim(getAxisInd());

    if (inShape[axis] == 1) {
        return errorAt(op, "Softmax on 1 element doesn't make sense (dim along the 'axis' equal 1)");
    }

    const auto cmxSizeLimit = Byte(SHAVE_LIB_DATA_SIZE) - Byte(8 * FP16_SIZE);
    if (Byte(inShape[axis] * FP16_SIZE) > cmxSizeLimit) {
        return errorAt(op, "Axis '{0}' dimension '{1}' exceeds local CMX buffer limitation '{2}'", axis, inShape[axis],
                       cmxSizeLimit);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SoftMaxUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::SoftmaxParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(getAxisInd()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SoftmaxParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseSoftmax(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPASoftMax supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPASoftMax supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_SoftmaxParams();
    const auto axis = getIntAttr(_ctx, params->axis());
    return builder.create<VPUIP::SoftMaxUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], axis);
}

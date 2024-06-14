//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NegativeUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto negative = MVCNN::CreatePowerParams(writer, 0.0f, -1.0f, 1.0f);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_PowerParams);
    builder.add_nested_params(negative.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseNegative(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                        ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask*) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPANegative supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPANegative supports only 1 output, got {0}", outputs.size());
    return builder.create<VPUIP::NegativeUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0]);
}

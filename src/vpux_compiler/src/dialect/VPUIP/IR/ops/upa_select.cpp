//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0.
//

//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SelectUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;
    type = writer.createString("select");
    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
}

mlir::LogicalResult vpux::VPUIP::SelectUPAOp::verify() {
    if (!((getInput2().getType() == getInput3().getType()) && (getInput3().getType() == getOutputBuff().getType()))) {
        return errorAt(*this, "Input 2, 3 & output_buff have different type");
    }
    return mlir::success();
}

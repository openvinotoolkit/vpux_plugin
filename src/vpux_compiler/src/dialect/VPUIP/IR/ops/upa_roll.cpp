//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::RollUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::RollParamsBuilder builder(writer);

    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_RollParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseRoll(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                    ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask*) {
    VPUX_THROW_UNLESS(inputs.size() == 3, "RollUPA supports only 3 inputs", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "RollUPA supports only 1 output", outputs.size());

    return builder.create<VPUIP::RollUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2], outputs[0]);
}

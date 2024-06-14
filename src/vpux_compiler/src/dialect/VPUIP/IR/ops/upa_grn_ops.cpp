//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::GRNUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::GRNParamsBuilder builder(writer);
    builder.add_bias(static_cast<float>(getBias().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GRNParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseGRN(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                   ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAGRN supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAGRN supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_GRNParams();
    const auto bias = getFPAttr(_ctx, params->bias());

    return builder.create<VPUIP::GRNUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], bias);
}

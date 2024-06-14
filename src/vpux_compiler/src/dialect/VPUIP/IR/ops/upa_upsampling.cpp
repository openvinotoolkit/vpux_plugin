//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::UpsamplingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    SmallVector<int32_t> pad_x = {
            checked_cast<int32_t>(getPadAttr().getPadsWidth()[0].cast<mlir::IntegerAttr>().getInt()),
            checked_cast<int32_t>(getPadAttr().getPadsWidth()[1].cast<mlir::IntegerAttr>().getInt())};
    auto pad_x_vector = writer.createVector(pad_x);

    SmallVector<int32_t> pad_y = {
            checked_cast<int32_t>(getPadAttr().getPadsHeight()[0].cast<mlir::IntegerAttr>().getInt()),
            checked_cast<int32_t>(getPadAttr().getPadsHeight()[1].cast<mlir::IntegerAttr>().getInt())};
    auto pad_y_vector = writer.createVector(pad_y);

    SmallVector<int32_t> pad_z = {
            checked_cast<int32_t>(getPadAttr().getPadsChannel()[0].cast<mlir::IntegerAttr>().getInt()),
            checked_cast<int32_t>(getPadAttr().getPadsChannel()[1].cast<mlir::IntegerAttr>().getInt())};
    auto pad_z_vector = writer.createVector(pad_z);

    MVCNN::UpsamplingParamsBuilder builder(writer);
    builder.add_upsampling_factor_x(checked_cast<int32_t>(getUpsamplingFactor()[0].cast<mlir::IntegerAttr>().getInt()));
    builder.add_upsampling_factor_y(checked_cast<int32_t>(getUpsamplingFactor()[1].cast<mlir::IntegerAttr>().getInt()));
    builder.add_upsampling_factor_z(checked_cast<int32_t>(getUpsamplingFactor()[2].cast<mlir::IntegerAttr>().getInt()));
    builder.add_pad_x(pad_x_vector);
    builder.add_pad_y(pad_y_vector);
    builder.add_pad_z(pad_z_vector);

    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_UpsamplingParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseUpsampling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                          ArrayRef<mlir::Value> outputs,
                                                          const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAUpsampling supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAUpsampling supports only 1 output, got {0}", outputs.size());

    const auto* params = task->softLayerParams_as_UpsamplingParams();
    const SmallVector<int32_t, 3> upsampling_factor = {params->upsampling_factor_x(), params->upsampling_factor_y(),
                                                       params->upsampling_factor_z()};
    auto padChannelAttr = getIntArrayAttr(_ctx, SmallVector<int64_t>{params->pad_z()->Get(0), params->pad_z()->Get(1)});
    auto padHeightAttr = getIntArrayAttr(_ctx, SmallVector<int64_t>{params->pad_y()->Get(0), params->pad_y()->Get(1)});
    auto padWidthAttr = getIntArrayAttr(_ctx, SmallVector<int64_t>{params->pad_x()->Get(0), params->pad_x()->Get(1)});
    auto padAttr = IE::UpsamplingPadAttr::get(_ctx, padChannelAttr, padHeightAttr, padWidthAttr);

    return builder.create<VPUIP::UpsamplingUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                  getIntArrayAttr(_ctx, upsampling_factor), padAttr);
}

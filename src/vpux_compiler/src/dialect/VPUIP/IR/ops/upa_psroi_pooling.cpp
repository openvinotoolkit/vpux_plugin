// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

IE::PSROIPoolingMode convertPSROIPoolingModeToIE(MVCNN::PSROIPoolingMode mode) {
    switch (mode) {
    case MVCNN::PSROIPoolingMode::PSROIPoolingMode_AVERAGE:
        return IE::PSROIPoolingMode::AVERAGE;
    case MVCNN::PSROIPoolingMode::PSROIPoolingMode_BILINEAR:
        return IE::PSROIPoolingMode::BILINEAR;
    default:
        VPUX_THROW("Unknown PSROIPoolingMode. Got {0} mode", mode);
    }
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::PSROIPoolingUPAOp::verify() {
    const auto outputDim = getOutputDim();

    if (outputDim <= 0) {
        return errorAt(*this, "Attribute outputDim should be positive. Got {0} value", outputDim);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PSROIPoolingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto spatialBinsX = getSpatialBinsX().has_value() ? getSpatialBinsX().value() : 1;
    const auto spatialBinsY = getSpatialBinsY().has_value() ? getSpatialBinsY().value() : 1;
    const auto psROIPoolingMode = getMode().has_value() ? getMode().value() : IE::PSROIPoolingMode::AVERAGE;

    MVCNN::PSROIPoolingParamsBuilder builder(writer);

    builder.add_output_dim(checked_cast<uint32_t>(getOutputDim()));
    builder.add_spatial_scale(checked_cast<float>(getSpatialScaleAttr().getValueAsDouble()));
    builder.add_group_size(checked_cast<uint32_t>(getGroupSize()));
    builder.add_pooled_w(checked_cast<uint32_t>(getGroupSize()));
    builder.add_pooled_h(checked_cast<uint32_t>(getGroupSize()));
    builder.add_spatial_bin_x(checked_cast<uint32_t>(spatialBinsX));
    builder.add_spatial_bin_y(checked_cast<uint32_t>(spatialBinsY));
    builder.add_mode(convertVPUXPSROIPoolingModeToMVNCNN(psROIPoolingMode));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PSROIPoolingParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parsePSROIPooling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPAPSROIPooling supports only 2 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAPSROIPooling supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_PSROIPoolingParams();
    const auto outputDim = getIntAttr(_ctx, params->output_dim());
    const auto spatialScale = getFPAttr(_ctx, params->spatial_scale());
    const auto groupSize = getIntAttr(_ctx, params->group_size());
    const auto spatialBinsX = getIntAttr(_ctx, params->spatial_bin_x());
    const auto spatialBinsY = getIntAttr(_ctx, params->spatial_bin_y());
    IE::PSROIPoolingMode mode = convertPSROIPoolingModeToIE(params->mode());

    return builder.create<VPUIP::PSROIPoolingUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                    outputDim, spatialScale, groupSize, spatialBinsX, spatialBinsY,
                                                    IE::PSROIPoolingModeAttr::get(_ctx, mode));
}

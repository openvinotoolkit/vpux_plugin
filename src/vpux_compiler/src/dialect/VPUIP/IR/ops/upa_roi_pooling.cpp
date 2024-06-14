//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::ROIPoolingUPAOp::verify() {
    const auto op = getOperation();
    const auto inShapeFeatureMap = getShape(getInput());
    const auto inShapeCoord = getShape(getCoords());

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(op, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(op, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    const auto outputSize = parseIntArrayAttr<int64_t>(getOutputSize());
    if (outputSize.size() != 2) {
        return errorAt(op, "Dimension of pooled size is expected to be equal to 2. Got {0}", outputSize.size());
    }

    if (outputSize[0] <= 0 || outputSize[1] <= 0) {
        return errorAt(op, "Pooled size attributes pooled_h and pooled_w should be positive.");
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ROIPoolingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float spatial_scale_val = static_cast<float>(getSpatialScale().convertToDouble());
    uint32_t num_rois = checked_cast<uint32_t>(getCoords().getType().cast<vpux::NDTypeInterface>().getShape().raw()[0]);
    const auto output_size = parseIntArrayAttr<int64_t>(getOutputSizeAttr());

    MVCNN::ROIPoolingParamsBuilder builder(writer);
    builder.add_spatial_scale(spatial_scale_val);
    builder.add_roi_pooling_method(convertVPUXROIPoolingMethod2Int32(getMethod()));
    builder.add_num_rois(num_rois);
    builder.add_pooled_h(checked_cast<uint32_t>(output_size[0]));
    builder.add_pooled_w(checked_cast<uint32_t>(output_size[1]));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ROIPoolingParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseROIPooling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                          ArrayRef<mlir::Value> outputs,
                                                          const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPAROIPooling supports only 2 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAROIPooling supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_ROIPoolingParams();
    const auto outputSize = getIntArrayAttr(_ctx, SmallVector<uint32_t>{params->pooled_h(), params->pooled_w()});
    const auto spatialScale = getFPAttr(_ctx, params->spatial_scale());
    IE::ROIPoolingMethod method;
    switch (params->roi_pooling_method()) {
    case 0:
        method = IE::ROIPoolingMethod::MAX;
        break;
    case 1:
        method = IE::ROIPoolingMethod::BILINEAR;
        break;
    default:
        VPUX_THROW("Unknown ROIPoolingMethod. MAX and BILINEAR methods are supported only");
    }

    return builder.create<VPUIP::ROIPoolingUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                  outputSize, spatialScale,
                                                  IE::ROIPoolingMethodAttr::get(_ctx, method));
}

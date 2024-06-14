//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DetectionOutputUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto detectionOutputAttr = getAttr();

    const auto codeType = detectionOutputAttr.getCodeType().getValue();
    const auto codeTypeString = stringifyDetectionOutputCodeType(codeType);
    const auto fbCodeType = writer.createString(codeTypeString);

    MVCNN::DetectionOutputParamsBuilder builder(writer);
    builder.add_num_classes(checked_cast<int32_t>(detectionOutputAttr.getNumClasses().getInt()));
    builder.add_keep_top_k(
            checked_cast<int32_t>(detectionOutputAttr.getKeepTopK()[0].cast<mlir::IntegerAttr>().getInt()));
    builder.add_nms_threshold(static_cast<float>(detectionOutputAttr.getNmsThreshold().getValue().convertToDouble()));
    builder.add_background_label_id(checked_cast<int32_t>(detectionOutputAttr.getBackgroundLabelId().getInt()));
    builder.add_top_k(checked_cast<int32_t>(detectionOutputAttr.getTopK().getInt()));
    builder.add_variance_encoded_in_target(detectionOutputAttr.getVarianceEncodedInTarget().getValue());
    builder.add_code_type(fbCodeType);
    builder.add_share_location(detectionOutputAttr.getShareLocation().getValue());
    builder.add_confidence_threshold(
            static_cast<float>(detectionOutputAttr.getConfidenceThreshold().getValue().convertToDouble()));
    builder.add_clip_before_nms(detectionOutputAttr.getClipBeforeNms().getValue());
    builder.add_clip_after_nms(detectionOutputAttr.getClipAfterNms().getValue());
    builder.add_decrease_label_id(detectionOutputAttr.getDecreaseLabelId().getValue());
    builder.add_normalized(detectionOutputAttr.getNormalized().getValue());
    builder.add_input_height(checked_cast<uint32_t>(detectionOutputAttr.getInputHeight().getValue().getSExtValue()));
    builder.add_input_width(checked_cast<uint32_t>(detectionOutputAttr.getInputWidth().getValue().getSExtValue()));
    builder.add_objectness_score(
            static_cast<float>(detectionOutputAttr.getObjectnessScore().getValue().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DetectionOutputParams});
}

IE::DetectionOutputCodeType convertMvcnnCodeTypeToIE(const std::string& upaCodeType) {
    if (upaCodeType == "CENTER_SIZE") {
        return IE::DetectionOutputCodeType::CENTER_SIZE;
    } else if (upaCodeType == "CORNER_SIZE") {
        return IE::DetectionOutputCodeType::CORNER_SIZE;
    } else if (upaCodeType == "CORNER") {
        return IE::DetectionOutputCodeType::CORNER;
    }

    VPUX_THROW("Unsupported DetectionOutput upaCodeType: {0}", upaCodeType);
}

mlir::Operation* vpux::VPUIP::BlobReader::parseDetectionOutput(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                               ArrayRef<mlir::Value> outputs,
                                                               const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 5, "UPADetectionOutput supports only 5 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPADetectionOutput supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_DetectionOutputParams();
    const auto numClasses = getIntAttr(_ctx, params->num_classes());
    const auto backgroundLabelId = getIntAttr(_ctx, params->background_label_id());
    const auto topK = getIntAttr(_ctx, params->top_k());
    const auto varianceEncodedInTarget = mlir::BoolAttr::get(_ctx, params->variance_encoded_in_target());
    const auto keepTopK = getIntArrayAttr(_ctx, SmallVector<int32_t>{params->keep_top_k()});
    const auto codeType =
            IE::DetectionOutputCodeTypeAttr::get(_ctx, convertMvcnnCodeTypeToIE(params->code_type()->str()));
    const auto shareLocation = mlir::BoolAttr::get(_ctx, params->share_location());
    const auto nmsThreshold = getFPAttr(_ctx, params->nms_threshold());
    const auto confidenceThreshold = getFPAttr(_ctx, params->confidence_threshold());
    const auto clipAfterNms = mlir::BoolAttr::get(_ctx, params->clip_after_nms());
    const auto clipBeforeNms = mlir::BoolAttr::get(_ctx, params->clip_before_nms());
    const auto decreaseLabelId = mlir::BoolAttr::get(_ctx, params->decrease_label_id());
    const auto normalized = mlir::BoolAttr::get(_ctx, params->normalized());
    const auto inputHeight = getIntAttr(_ctx, params->input_height());
    const auto inputWidth = getIntAttr(_ctx, params->input_width());
    const auto objectnessScore = getFPAttr(_ctx, params->objectness_score());

    const auto detectionOutputAttr = IE::DetectionOutputAttr::get(
            _ctx, numClasses, backgroundLabelId, topK, varianceEncodedInTarget, keepTopK, codeType, shareLocation,
            nmsThreshold, confidenceThreshold, clipAfterNms, clipBeforeNms, decreaseLabelId, normalized, inputHeight,
            inputWidth, objectnessScore);
    return builder.create<VPUIP::DetectionOutputUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2],
                                                       inputs[3], inputs[4], outputs[0], detectionOutputAttr);
}

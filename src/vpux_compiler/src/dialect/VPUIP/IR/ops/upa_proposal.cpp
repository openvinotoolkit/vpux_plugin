//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::ProposalUPAOp::verify() {
    IE::ProposalAttr attr = getProposalAttrs();

    auto frameworkValue = attr.getFramework().getValue();

    if (frameworkValue != "" && frameworkValue != "tensorflow") {
        return errorAt(*this, "Unsupported framework attr {0}", frameworkValue.str());
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ProposalUPAOp::serialize(VPUIP::BlobWriter& writer) {
    IE::ProposalAttr attr = getProposalAttrs();
    VPUIP::BlobWriter::String framework = writer.createString(attr.getFramework().getValue());

    auto ratio_fb = writer.createVector(parseFPArrayAttr<float>(attr.getRatio()));
    auto scale_fb = writer.createVector(parseFPArrayAttr<float>(attr.getScale()));

    MVCNN::ProposalParamsBuilder builder(writer);
    builder.add_ratio(ratio_fb);
    builder.add_scale(scale_fb);

    builder.add_min_size(checked_cast<uint32_t>(attr.getMinSize().getValue().getSExtValue()));
    builder.add_base_size(checked_cast<uint32_t>(attr.getBaseSize().getValue().getSExtValue()));
    builder.add_framework(framework);
    builder.add_normalize(attr.getNormalize().getValue());
    builder.add_nms_thresh(static_cast<float>(attr.getNmsThresh().getValueAsDouble()));
    builder.add_feat_stride(checked_cast<uint32_t>(attr.getFeatStride().getValue().getSExtValue()));
    builder.add_pre_nms_topn(checked_cast<uint32_t>(attr.getPreNmsTopN().getValue().getSExtValue()));
    builder.add_post_nms_topn(checked_cast<uint32_t>(attr.getPostNmsTopN().getValue().getSExtValue()));
    builder.add_box_size_scale(static_cast<float>(attr.getBoxSizeScale().getValueAsDouble()));
    builder.add_clip_after_nms(attr.getClipAfterNms().getValue());
    builder.add_for_deformable(false);  // ov doesn't have this parameter
    builder.add_pre_nms_thresh(0.0);    // ov doesn't have this parameter
    builder.add_clip_before_nms(attr.getClipBeforeNms().getValue());
    builder.add_box_coordinate_scale(static_cast<float>(attr.getBoxCoordinateScale().getValueAsDouble()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ProposalParams});
}

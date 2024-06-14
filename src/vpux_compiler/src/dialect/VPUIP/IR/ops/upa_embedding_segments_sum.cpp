//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EmbeddingSegmentsSumUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto indices = writer.createVector(parseIntArrayAttr<int32_t>(getIndicesValue()));
    const auto segmentIds = writer.createVector(parseIntArrayAttr<int32_t>(getSegmentIdsValue()));

    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = vpux::type::float16(val);
        return valFP16.to_bits();
    };
    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    VPUIP::BlobWriter::Vector<uint16_t> serializedWeights;
    const auto weightsArr = parseFPArrayAttr<double>(getPerSampleWeightsValue());
    serializedWeights = getVecFP16(weightsArr);

    MVCNN::EmbeddingSegmentsSumParamsBuilder builder(writer);

    builder.add_indices(indices);
    builder.add_segment_ids(segmentIds);
    builder.add_num_segments(checked_cast<int32_t>(getNumSegmentsValue()));
    builder.add_default_index(checked_cast<int32_t>(getDefaultIndexValue()));
    builder.add_per_sample_weights(serializedWeights);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EmbeddingSegmentsSumParams});
}

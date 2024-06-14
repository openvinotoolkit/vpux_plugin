//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::StridedSliceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    auto attrToVector = [&](mlir::ArrayAttr attr) {
        return to_std_vector(parseIntArrayAttr<uint32_t>(attr));
    };

    const auto beginsVec = attrToVector(getBegins());
    const auto endsVec = attrToVector(getEnds());
    const auto stridesVec = attrToVector(getStrides());

    const auto paramsOff = MVCNN::CreateStridedSliceParamsDirect(writer, &beginsVec, &endsVec, &stridesVec);

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_StridedSliceParams});
}

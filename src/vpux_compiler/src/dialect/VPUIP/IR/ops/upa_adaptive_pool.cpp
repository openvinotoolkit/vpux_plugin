//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AdaptiveAvgPoolUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::AdaptivePoolParamsBuilder builder(writer);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_AdaptivePoolParams});
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AdaptiveMaxPoolUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::AdaptivePoolParamsBuilder builder(writer);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_AdaptivePoolParams});
}

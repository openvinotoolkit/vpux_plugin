//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::MVNUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::MVNParamsBuilder builder(writer);
    builder.add_across_channels(getAcrossChannels().value_or(false));
    builder.add_normalize_variance(getNormalizeVariance().value_or(false));
    builder.add_eps(static_cast<float>(getEps().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_MVNParams});
}

mlir::LogicalResult vpux::VPUIP::MVNUPAOp::verify() {
    const auto inShape = getShape(getInput());

    if (inShape.size() != 3 && inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(*this, "Input shape should have 3, 4 or 5 dimensions");
    }

    return mlir::success();
}

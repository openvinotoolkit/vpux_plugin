//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ScatterNDUpdateUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ScatterNDUpdateParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ScatterNDUpdateParams});
}

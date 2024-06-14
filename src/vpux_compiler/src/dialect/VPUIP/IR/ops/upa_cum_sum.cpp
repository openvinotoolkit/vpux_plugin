//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CumSumUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CumSumParamsBuilder builder(writer);
    builder.add_exclusive(checked_cast<bool>(getExclusive()));
    builder.add_reverse(checked_cast<bool>(getReverse()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CumSumParams});
}

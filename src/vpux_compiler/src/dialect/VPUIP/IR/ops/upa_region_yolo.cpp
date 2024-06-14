//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::RegionYoloUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::Vector<int32_t> serializedMask;
    if (getMask().has_value()) {
        serializedMask = writer.createVector(parseIntArrayAttr<int32_t>(getMask().value()));
    }

    MVCNN::RegionYOLOParamsBuilder builder(writer);
    builder.add_coords(checked_cast<int32_t>(getCoords()));
    builder.add_classes(checked_cast<int32_t>(getClasses()));
    builder.add_num(checked_cast<int32_t>(getNumRegions()));
    builder.add_do_softmax(getDoSoftmax().value_or(false));
    if (getMask().has_value()) {
        builder.add_mask(serializedMask);
    }

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_RegionYOLOParams});
}

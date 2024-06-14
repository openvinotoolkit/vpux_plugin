//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/utils/core/enums.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::InterpolateUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::InterpolateParamsBuilder builder(writer);

    const auto interpolateModeIter = VPUIP::supportedInterpModeMap.find(getMode());
    VPUX_THROW_UNLESS(interpolateModeIter != VPUIP::supportedInterpModeMap.end(), "Unsupported interpolate mode {0}",
                      getMode());
    builder.add_interpolationMode(interpolateModeIter->second);

    const auto coordModeIter = VPUIP::coordTransformModeMap.find(getCoordMode());
    VPUX_THROW_UNLESS(coordModeIter != VPUIP::coordTransformModeMap.end(),
                      "Unsupported coordinate transformation mode {0}", getCoordMode());
    builder.add_coordTransformMode(coordModeIter->second);

    const auto nearestModeIter = VPUIP::nearestModeMap.find(getNearestMode());
    VPUX_THROW_UNLESS(nearestModeIter != VPUIP::nearestModeMap.end(), "Unsupported nearest mode {0}", getNearestMode());
    builder.add_nearestMode(nearestModeIter->second);

    builder.add_align_corners(getCoordMode() == IE::InterpolateCoordMode::ALIGN_CORNERS);
    builder.add_antialias(getAntialias());

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_InterpolateParams});
}

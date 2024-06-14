//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::SpaceToDepthUPAOp::verify() {
    if (getBlockSize() <= 0) {
        return errorAt(*this, "block_size should be greater than zero");
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SpaceToDepthUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::SpaceToDepthParamsBuilder builder(writer);

    const auto blockSize = checked_cast<int32_t>(getBlockSize());
    builder.add_blockSize(blockSize);

    const auto spdMode = VPUIP::convertVPUXSpaceToDepthMode2MVCNN(getMode());
    builder.add_mode(spdMode);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SpaceToDepthParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseSpaceToDepth(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPASpaceToDepth supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPASpaceToDepth supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_SpaceToDepthParams();
    const auto block_size = getIntAttr(_ctx, params->blockSize());

    IE::SpaceToDepthMode spdMode;
    switch (params->mode()) {
    case MVCNN::SpaceToDepthMode::SpaceToDepthMode_BLOCKS_FIRST:
        spdMode = IE::SpaceToDepthMode::BLOCKS_FIRST;
        break;
    case MVCNN::SpaceToDepthMode::SpaceToDepthMode_DEPTH_FIRST:
        spdMode = IE::SpaceToDepthMode::DEPTH_FIRST;
        break;
    default:
        VPUX_THROW("Unsupported SpaceToDepthMode {0}", params->mode());
    }

    return builder.create<VPUIP::SpaceToDepthUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], block_size,
                                                    IE::SpaceToDepthModeAttr::get(_ctx, spdMode));
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::DepthToSpaceUPAOp::verify() {
    const auto op = getOperation();
    if (getBlockSize() <= 0) {
        return errorAt(op, "Block size should be greater than 0. Got {0}", getBlockSize());
    }

    if (getMode() != vpux::IE::DepthToSpaceMode::BLOCKS_FIRST && getMode() != vpux::IE::DepthToSpaceMode::DEPTH_FIRST) {
        return errorAt(op, "Unknown DepthToSpaceMode. Blocks_FIRST and DEPTH_FIRST methods are supported only");
    }

    return mlir::success();
}

void vpux::VPUIP::DepthToSpaceUPAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                                           mlir::Value input, mlir::Value output_buff, mlir::IntegerAttr block_size,
                                           vpux::IE::DepthToSpaceModeAttr mode) {
    build(odsBuilder, odsState, input, output_buff, block_size, mode, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DepthToSpaceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::DepthToSpaceParamsBuilder builder(writer);

    const auto blockSize = checked_cast<int32_t>(getBlockSize());
    builder.add_blockSize(blockSize);

    builder.add_mode(vpux::VPUIP::convertVPUXDepthToSpaceMode2MVCNN(getMode()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DepthToSpaceParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseDepthToSpace(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPADepthToSpace supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPADepthToSpace supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_DepthToSpaceParams();
    const auto block_size = getIntAttr(_ctx, params->blockSize());
    IE::DepthToSpaceMode mode;
    switch (params->mode()) {
    case MVCNN::DepthToSpaceMode_BLOCKS_FIRST:
        mode = IE::DepthToSpaceMode::BLOCKS_FIRST;
        break;
    case MVCNN::DepthToSpaceMode_DEPTH_FIRST:
        mode = IE::DepthToSpaceMode::DEPTH_FIRST;
        break;
    default:
        VPUX_THROW("Unknown DepthToSpaceMode. BLOCKS_FIRST and DEPTH_FIRST methods are supported only");
    }
    return builder.create<VPUIP::DepthToSpaceUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], block_size,
                                                    IE::DepthToSpaceModeAttr::get(_ctx, mode));
}

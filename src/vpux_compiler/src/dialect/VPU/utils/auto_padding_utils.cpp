//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

bool hasAutoPadding(mlir::ModuleOp module, StringRef paddingMode) {
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch auto padding mode");

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(paddingMode);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find {0} IE.OptionOp attribute", paddingMode);
    return static_cast<bool>(attrValue.getOptionValue());
}

bool VPU::hasAutoPaddingIDU(mlir::ModuleOp module) {
    return hasAutoPadding(module, AUTO_PADDING_IDU);
}

bool VPU::hasAutoPaddingODU(mlir::ModuleOp module) {
    return hasAutoPadding(module, AUTO_PADDING_ODU);
}

bool VPU::inputCompatibleWithAutoPad(vpux::NDTypeInterface type) {
    const auto inShape = type.getShape();
    VPUX_THROW_WHEN(inShape.size() != 4, "Unsupported shape rank: {0}", inShape.size());
    const auto inputC = inShape[Dims4D::Act::C];
    const auto elemTypeBitWidth = type.getElemTypeSize().count();
    using vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;

    return (elemTypeBitWidth >= CHAR_BIT && ((elemTypeBitWidth < FP16_WIDTH && inputC < VPU_CHANNEL_ALIGNMENT) ||
                                             (elemTypeBitWidth >= FP16_WIDTH && inputC < WIDTH16_CHANNEL_LIMIT)));
}

bool VPU::hasOnlyOutPadding(mlir::ModuleOp module) {
    return VPU::hasAutoPaddingODU(module) && !VPU::hasAutoPaddingIDU(module);
}

bool VPU::hasOnlyInPadding(mlir::ModuleOp module) {
    return !VPU::hasAutoPaddingODU(module) && VPU::hasAutoPaddingIDU(module);
}

//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool hasAutoPadding(mlir::ModuleOp module, StringRef paddingMode) {
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find PipelineOptions to fetch auto padding mode");
        return false;
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(paddingMode);
    if (attrValue == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find IE.OptionOp to fetch auto padding mode");
        return false;
    }
    return static_cast<bool>(attrValue.getOptionValue());
}

bool VPU::hasAutoPaddingIDU(mlir::ModuleOp module) {
    return hasAutoPadding(module, AUTO_PADDING_IDU);
}

bool VPU::hasAutoPaddingODU(mlir::ModuleOp module) {
    return hasAutoPadding(module, AUTO_PADDING_ODU);
}

bool VPU::outputCompatibleWithAutoPad(vpux::NDTypeInterface type) {
    const auto outShape = type.getShape();
    const auto outputC = outShape[Dims4D::Act::C];
    const auto elemTypeBitWidth = type.getElemTypeSize().count();
    using vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;

    return (outputC < VPU_CHANNEL_ALIGNMENT && elemTypeBitWidth >= CHAR_BIT);
}

bool VPU::inputCompatibleWithAutoPad(vpux::NDTypeInterface type) {
    const auto inShape = type.getShape();
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

bool VPU::canAutopadOutput(mlir::Operation* op) {
    auto type = op->getResult(0).getType().template cast<vpux::NDTypeInterface>();
    auto isODUSupported = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op) != nullptr ? true : false;
    return isODUSupported && outputCompatibleWithAutoPad(type) && hasAutoPaddingODU(getModuleOp(op));
}

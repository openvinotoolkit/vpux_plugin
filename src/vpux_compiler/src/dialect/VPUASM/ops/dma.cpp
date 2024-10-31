//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// NNDMAOp
//

void vpux::VPUASM::NNDMAOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for NNDMAOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
    return;
}

size_t vpux::VPUASM::NNDMAOp::getBinarySize() {
    return sizeof(nn_public::VpuDMATask);
}

size_t vpux::VPUASM::NNDMAOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDMATask);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::NNDMAOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::NNDMAOp::getMemoryAccessingProc() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::NNDMAOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "dma", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::NNDMAOp::hasMemoryFootprint() {
    return true;
}

mlir::LogicalResult vpux::VPUASM::NNDMAOp::verify() {
    // TODO: E#82441
    // act_compression_size_entryAttr not required for compressed weight case but it's necessary for activation
    // spilling. We need to distinguish these two cases and require act_compression_size_entryAttr be set if
    // acceleration_mode is decompression for activation spilling but current API doesn't allow us to do that.
    // Same for act_compression_sparsity_map
    bool isConfigurationSupported = false;
    if (getActCompressionSizeEntryAttr() || getActCompressionSparsityMapAttr()) {
        isConfigurationSupported = getAccelerationMode() == VPUIP::DMAAccMode::COMPRESSION ||
                                   getAccelerationMode() == VPUIP::DMAAccMode::DECOMPRESSION;
    } else {
        isConfigurationSupported = getAccelerationMode() == VPUIP::DMAAccMode::DISABLE ||
                                   getAccelerationMode() == VPUIP::DMAAccMode::DECOMPRESSION;
    }

    if (!isConfigurationSupported) {
        return errorAt(getLoc(),
                       "Operation {0}: unsupported configuration: "
                       "act_compression_size_entryAttr={1}/act_compression_sparsity_mapAttr={2} and "
                       "acceleration_mode={3}",
                       getOperationName(), getActCompressionSizeEntryAttr(), getActCompressionSparsityMapAttr(),
                       getAccelerationMode());
    }

    const auto symName = getSymName();
    llvm::SmallVector<llvm::StringRef> subStrings;
    symName.split(subStrings, '_');
    VPUX_THROW_UNLESS(subStrings.size() == 4,
                      "symName for NNDMAOp {0} does not respect the format @NNDMAOpName_tileIdx_listIdx_value",
                      getOperationName());
    const auto tileIdx = std::stoul(subStrings[1].str());
    const auto listIdx = std::stoul(subStrings[2].str());
    const auto value = std::stoul(subStrings[3].str());
    const auto taskIndex = getTaskIndex();
    if (tileIdx != taskIndex.getTileIdx() || listIdx > taskIndex.getListIdx() || value != taskIndex.getValue()) {
        return errorAt(getLoc(), "Operation {0}: symbolName {1} and taskIndex {2} are not in sync", getOperationName(),
                       symName, taskIndex);
    }

    return mlir::success();
}

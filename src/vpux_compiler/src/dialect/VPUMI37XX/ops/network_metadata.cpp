//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_headers/serial_metadata.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux_headers/serial_metadata.hpp"

using namespace vpux;

//
//  NetworkMetadataOp
//

void vpux::VPUMI37XX::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                   elf::NetworkMetadata& metadata) {
    auto operation = getOperation();
    auto mainModule = operation->getParentOfType<mlir::ModuleOp>();

    auto nBarrs = VPUIP::getNumAvailableBarriers(operation);
    metadata.mResourceRequirements.nn_barriers_ = checked_cast<uint8_t>(nBarrs);
    metadata.mResourceRequirements.nn_slice_count_ = checked_cast<uint8_t>(VPUIP::getNumTilesUsed(mainModule));

    metadata.mResourceRequirements.ddr_scratch_length_ =
            checked_cast<uint32_t>(IE::getAvailableMemory(mainModule, vpux::VPU::MemoryKind::DDR).getByteSize());

    metadata.mResourceRequirements.nn_slice_length_ =
            checked_cast<uint32_t>(IE::getAvailableMemory(mainModule, vpux::VPU::MemoryKind::CMX_NN).getByteSize());

    auto serializedMetadata = elf::MetadataSerialization::serialize(metadata);
    binDataSection.appendData(&serializedMetadata[0], serializedMetadata.size());
}

void vpux::VPUMI37XX::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VPUX_UNUSED(binDataSection);
    VPUX_THROW("ERROR");
}

size_t vpux::VPUMI37XX::NetworkMetadataOp::getBinarySize() {
    return sizeof(elf::NetworkMetadata);
}

size_t vpux::VPUMI37XX::NetworkMetadataOp::getAlignmentRequirements() {
    return alignof(elf::NetworkMetadata);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::NetworkMetadataOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::NetworkMetadataOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::NetworkMetadataOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

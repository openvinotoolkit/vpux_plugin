//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

#include "schema/profiling_generated.h"

using namespace vpux;

//
//  ProfilingMetadataOp
//

void vpux::VPUMI37XX::ProfilingMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto denseMetaAttr = getMetadata().dyn_cast<mlir::DenseElementsAttr>();
    VPUX_THROW_UNLESS(denseMetaAttr != nullptr, "ProfilingMetadata's data is NULL");

    auto buf = denseMetaAttr.getRawData();
    binDataSection.appendData(reinterpret_cast<const uint8_t*>(buf.data()), buf.size());
}

size_t vpux::VPUMI37XX::ProfilingMetadataOp::getBinarySize() {
    return sizeof(ProfilingFB::ProfilingMeta);
}

size_t vpux::VPUMI37XX::ProfilingMetadataOp::getAlignmentRequirements() {
    return alignof(ProfilingFB::ProfilingMeta);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ProfilingMetadataOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::ProfilingMetadataOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ProfilingMetadataOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

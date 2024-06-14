//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

void vpux::ELFNPU37XX::PadOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto padSize = getPaddingSize();

    auto padValue = getPaddingValue().value_or(0);

    SmallVector<uint8_t> padding(padSize, padValue);

    binDataSection.appendData(padding.data(), padSize);
}

size_t vpux::ELFNPU37XX::PadOp::getBinarySize() {
    return getPaddingSize();
}

size_t vpux::ELFNPU37XX::PadOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_NO_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::ELFNPU37XX::PadOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::ELFNPU37XX::PadOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::ELFNPU37XX::PadOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

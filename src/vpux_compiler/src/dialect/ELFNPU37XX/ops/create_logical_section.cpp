//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

void vpux::ELFNPU37XX::CreateLogicalSectionOp::serialize(elf::Writer& writer,
                                                         vpux::ELFNPU37XX::SectionMapType& sectionMap,
                                                         vpux::ELFNPU37XX::SymbolMapType& symbolMap) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(sectionMap);
    VPUX_UNUSED(symbolMap);
}

void vpux::ELFNPU37XX::CreateLogicalSectionOp::preserialize(elf::Writer& writer,
                                                            vpux::ELFNPU37XX::SectionMapType& sectionMap) {
    const auto name = getSecName().str();
    auto section = writer.addEmptySection(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    auto block = getBody();
    auto ops = block->getOps<ELFNPU37XX::PutOpInSectionOp>();

    size_t nonBufferAllocs = 0;
    size_t maxOffset = 0;
    for (auto op : ops) {
        auto pointingOp = op.getInputArg().getDefiningOp();

        auto binOp = mlir::dyn_cast<ELFNPU37XX::BinaryOpInterface>(pointingOp);
        VPUX_THROW_UNLESS(binOp, "Operation included in section is not a BinaryOp");

        if (auto bufferOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(pointingOp)) {
            auto currentOffset = bufferOp.getByteOffset() + bufferOp.getBinarySize();
            if (currentOffset > maxOffset) {
                maxOffset = currentOffset;
            }
        } else {
            nonBufferAllocs += binOp.getBinarySize();
        }
    }
    section->setSize(maxOffset + nonBufferAllocs);

    sectionMap[getOperation()] = section;
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"

void vpux::ELFNPU37XX::CreateSectionOp::serialize(elf::Writer& writer, vpux::ELFNPU37XX::SectionMapType& sectionMap,
                                                  vpux::ELFNPU37XX::SymbolMapType& symbolMap) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(symbolMap);

    const auto section = sectionMap.find(getOperation());
    VPUX_THROW_WHEN(section == sectionMap.end(), "ELF section not found: {0}", getSecName().str());
    auto binDataSection = dynamic_cast<elf::writer::BinaryDataSection<uint8_t>*>(section->second);
    VPUX_THROW_WHEN(binDataSection == nullptr, "Invalid binary section in ELF writer");

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<vpux::ELFNPU37XX::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<vpux::ELFNPU37XX::BinaryOpInterface>(op);
            binaryOp.serialize(*binDataSection);
        }
    }
}

void vpux::ELFNPU37XX::CreateSectionOp::preserialize(elf::Writer& writer,
                                                     vpux::ELFNPU37XX::SectionMapType& sectionMap) {
    const auto name = getSecName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(name, static_cast<uint32_t>(getSecType()));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    size_t sectionSize = 0;
    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<vpux::ELFNPU37XX::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<vpux::ELFNPU37XX::BinaryOpInterface>(op);
            sectionSize += binaryOp.getBinarySize();
        }
    }
    section->setSize(sectionSize);

    sectionMap[getOperation()] = section;
}

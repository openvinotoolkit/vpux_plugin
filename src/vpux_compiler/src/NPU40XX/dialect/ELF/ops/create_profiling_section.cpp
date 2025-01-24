//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/attributes.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

void vpux::ELF::CreateProfilingSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                    vpux::ELF::SymbolMapType& symbolMap) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(symbolMap);

    const auto log = vpux::Logger::global();
    log.trace("Serializing ELF::CreateProfilingSectionOp");

    const auto section = sectionMap.find(getOperation());
    VPUX_THROW_WHEN(section == sectionMap.end(), "ELF section not found: {0}", getSymName().str());
    auto binDataSection = dynamic_cast<elf::writer::BinaryDataSection<uint8_t>*>(section->second);
    VPUX_THROW_WHEN(binDataSection == nullptr, "Invalid binary section in ELF writer");

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (auto metadataOp = mlir::dyn_cast<VPUASM::ProfilingMetadataOp>(op)) {
            metadataOp.serialize(*binDataSection);
        }
    }
}

void vpux::ELF::CreateProfilingSectionOp::preserialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                       vpux::ELF::SymbolReferenceMap&) {
    const auto name = getSymName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(
            name, static_cast<elf::Elf_Word>(vpux::ELF::SectionTypeAttr::VPU_SHT_PROF));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    auto block = getBody();
    size_t sectionSize = 0;
    for (auto& op : block->getOperations()) {
        if (auto metadataOp = mlir::dyn_cast<VPUASM::ProfilingMetadataOp>(op)) {
            sectionSize = metadataOp.getBinarySize(VPU::ArchKind::UNKNOWN);
        }
    }

    section->setSize(sectionSize);
    sectionMap[getOperation()] = section;
}

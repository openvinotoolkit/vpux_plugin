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
    VPUX_UNUSED(symbolMap);

    const auto log = vpux::Logger::global();
    log.trace("Serializing ELF::CreateProfilingSectionOp");

    const auto name = getSymName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(
            name, static_cast<elf::Elf_Word>(vpux::ELF::SectionTypeAttr::VPU_SHT_PROF));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (auto metadataOp = mlir::dyn_cast<VPUASM::ProfilingMetadataOp>(op)) {
            metadataOp.serialize(*section);
        }
    }

    sectionMap[getOperation()] = section;
}

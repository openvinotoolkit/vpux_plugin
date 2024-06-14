//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/attributes.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

void vpux::ELF::CreateMetadataSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                   vpux::ELF::SymbolMapType& symbolMap,
                                                   elf::NetworkMetadata& metadata) {
    VPUX_UNUSED(symbolMap);
    const auto name = getSymName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(
            name, static_cast<elf::Elf_Word>(vpux::ELF::SectionTypeAttr::VPU_SHT_NETDESC));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    bool isMetadataSerialized = false;
    auto block = getBody();
    for (auto& op : block->getOperations()) {
        VPUX_THROW_UNLESS(!isMetadataSerialized, "There should be only 1 metadata op in an ELF metadata section");
        if (auto metadata_op = mlir::dyn_cast<vpux::VPUASM::NetworkMetadataOp>(op)) {
            isMetadataSerialized = true;
            metadata_op.serialize(*section, metadata);
        }
    }
    VPUX_THROW_UNLESS(isMetadataSerialized, "No metadata defined in the ELF metadata section");

    sectionMap[getOperation()] = section;
}

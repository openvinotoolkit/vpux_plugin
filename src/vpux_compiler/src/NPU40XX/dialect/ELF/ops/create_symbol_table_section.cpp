//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"

using namespace vpux;

void ELF::CreateSymbolTableSectionOp::serialize(elf::Writer& writer, ELF::SectionMapType& sectionMap,
                                                ELF::SymbolMapType& symbolMap, ELF::SymbolReferenceMap& symRefMap) {
    VPUX_UNUSED(symRefMap);
    const auto name = getSymName().str();
    auto section = writer.addSymbolSection(name);

    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));

    auto& operations = getBody()->getOperations();
    for (auto& op : operations) {
        auto symbol = section->addSymbolEntry();
        auto symOp = llvm::dyn_cast<ELF::SymbolOp>(op);

        VPUX_THROW_UNLESS(symOp, "Symbol table section op is expected to contain only SymbolOps. Got {0}", op);
        symOp.serialize(symbol, sectionMap);
        symbolMap[symOp.getOperation()] = symbol;
    }

    // since we only currently issue Symbols with STB_LOCAL binding, we just set the info to the number of symbolOps
    // in the block

    section->setInfo(static_cast<uint32_t>(operations.size() + 1));

    sectionMap[getOperation()] = section;
}

void vpux::ELF::CreateSymbolTableSectionOp::preserialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                         vpux::ELF::SymbolReferenceMap&) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(sectionMap);

    // don't implement and go into elf::Writer internal state first
    // as there're symbol entries to be updated first
    // E#136375
}

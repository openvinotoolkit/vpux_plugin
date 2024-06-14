//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"

void vpux::ELF::DataSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                         vpux::ELF::SymbolMapType& symbolMap,
                                         vpux::ELF::SymbolReferenceMap& symRefMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = getSymName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(name, static_cast<uint32_t>(getSecType()));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<vpux::ELF::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<vpux::ELF::BinaryOpInterface>(op);

            binaryOp.serializeCached(*section, symRefMap);
        }
    }

    sectionMap[getOperation()] = section;
}

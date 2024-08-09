//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"

using namespace vpux;

void ELF::DataSectionOp::serialize(elf::Writer& writer, ELF::SectionMapType& sectionMap, ELF::SymbolMapType& symbolMap,
                                   ELF::SymbolReferenceMap& symRefMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = getSymName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(name, static_cast<uint32_t>(getSecType()));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<ELF::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<ELF::BinaryOpInterface>(op);

            binaryOp.serializeCached(*section, symRefMap);
        }
    }

    sectionMap[getOperation()] = section;
}

ELF::SymbolSignature ELF::DataSectionOp::getSymbolSignature() {
    auto symName = ELF::SymbolOp::getDefaultNamePrefix() + getSymName();
    return {mlir::SymbolRefAttr::get(getSymNameAttr()), symName.str(), ELF::SymbolType::STT_SECTION};
}

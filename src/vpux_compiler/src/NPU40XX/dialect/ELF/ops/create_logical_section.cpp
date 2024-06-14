//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"

void vpux::ELF::LogicalSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                            vpux::ELF::SymbolMapType& symbolMap,
                                            vpux::ELF::SymbolReferenceMap& symRefMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = getSymName().str();
    auto section = writer.addEmptySection(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());
    section->setType(static_cast<elf::Elf_Word>(getSecType()));

    size_t totalSize = 0;
    auto calcSpan = [&totalSize, &symRefMap](mlir::Operation& op) {
        auto binaryOp = mlir::dyn_cast<vpux::ELF::BinaryOpInterface>(&op);
        auto wrappableOp = mlir::dyn_cast<vpux::ELF::WrappableOpInterface>(&op);

        if (binaryOp && wrappableOp) {
            auto size = binaryOp.getBinarySizeCached(symRefMap);
            auto offset = wrappableOp.getMemoryOffset();
            auto span = size + offset;

            totalSize = std::max(totalSize, span);
        }
    };

    llvm::for_each(getBody()->getOperations(), calcSpan);

    section->setSize(totalSize);

    sectionMap[getOperation()] = section;
}

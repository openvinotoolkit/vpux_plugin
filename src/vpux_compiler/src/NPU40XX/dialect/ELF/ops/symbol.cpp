//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"

#include <mlir/IR/SymbolTable.h>

void vpux::ELF::SymbolOp::serialize(elf::writer::Symbol* symbol, vpux::ELF::SectionMapType& sectionMap) {
    auto symName = getSymName();
    auto symType = getType();
    auto symSize = getSize();
    auto symVal = getValue();

    /* From the serialization perspective the symbols can be of 5 types:
        - Section symbols: in this case the parentSection is the defining op itself;
        - Generic symbols: Symbols representing an OP inside the IR. In this case we need the parent section of either
       the OP or its placeholder;
        - Standalone symbols: symbols that do not relate to any entity inside the IR (nor the ELF itself).
      The ticket E#29144 plans to handle Standalone symbols.
    */

    auto referenceOp = mlir::SymbolTable::lookupNearestSymbolFrom(getOperation()->getParentOp(), getReferenceAttr());
    auto parentSection = referenceOp;
    if (!mlir::isa<ELF::ElfSectionInterface>(referenceOp)) {
        parentSection = referenceOp->getParentOp();
        VPUX_THROW_UNLESS(mlir::isa<ELF::ElfSectionInterface>(parentSection),
                          "Symbol op referencing and OP not in a section {0}", this);
    }

    symbol->setName(symName.str());
    symbol->setType(static_cast<elf::Elf_Word>(symType));
    symbol->setSize(symSize);
    symbol->setValue(symVal);

    auto sectionMapEntry = sectionMap.find(parentSection);
    VPUX_THROW_UNLESS(sectionMapEntry != sectionMap.end(), "Unable to find section entry for SymbolOp");
    auto sectionEntry = sectionMapEntry->second;

    symbol->setRelatedSection(sectionEntry);
}

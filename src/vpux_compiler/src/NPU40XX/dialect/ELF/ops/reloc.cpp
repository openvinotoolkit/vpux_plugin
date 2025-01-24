//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"

using namespace vpux;

void vpux::ELF::RelocOp::serialize(elf::writer::Relocation* relocation, vpux::ELF::SymbolMapType& symbolMap) {
    auto symbolRef = mlir::SymbolTable::lookupNearestSymbolFrom(getOperation(), getSourceSymbolAttr());
    auto symbolOp = mlir::dyn_cast_or_null<ELF::SymbolOp>(symbolRef);

    VPUX_THROW_UNLESS(symbolOp, "Reloc op expecting valid source symbol op {0}", this);

    auto symbolMapEntry = symbolMap.find(symbolOp);
    VPUX_THROW_UNLESS(symbolMapEntry != symbolMap.end(), "Unable to locate symbol entry for relocation");
    auto symbolEntry = symbolMapEntry->second;
    relocation->setSymbol(symbolEntry);

    auto relocType = getRelocationType();
    auto relocAddend = getAddend();

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(getOffset());
    relocation->setAddend(relocAddend);
}

void vpux::ELF::RelocOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, int64_t offset,
                               ::mlir::SymbolRefAttr sourceSymbol, vpux::ELF::RelocationType relocationType,
                               int64_t addend, llvm::StringRef description) {
    build(odsBuilder, odsState, offset, sourceSymbol, relocationType, addend, odsBuilder.getStringAttr(description));
}

void vpux::ELF::RelocOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, int64_t offset,
                               ::mlir::SymbolRefAttr sourceSymbol, vpux::ELF::RelocationType relocationType,
                               int64_t addend) {
    build(odsBuilder, odsState, offset, sourceSymbol, relocationType, addend, "");
}

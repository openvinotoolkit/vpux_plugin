//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

vpux::VPURT::BufferSection vpux::VPUASM::getBufferLocation(mlir::Operation* symTableOp, mlir::SymbolRefAttr symRef,
                                                           Logger log) {
    VPUX_THROW_UNLESS(symTableOp->hasTrait<mlir::OpTrait::SymbolTable>(),
                      "The symTableOp parameter must have the SymbolTable trait");
    auto symTable = mlir::SymbolTable(symTableOp);

    auto referencedOp = symTable.lookupSymbolIn(symTableOp, symRef);

    if (auto bufferOp = mlir::dyn_cast<VPUASM::DeclareBufferOp>(referencedOp)) {
        return bufferOp.getBufferType().getLocation().getSection();
    } else if (auto constantOp = mlir::dyn_cast<VPUASM::ConstBufferOp>(referencedOp)) {
        return constantOp.getBufferType().getLocation().getSection();
    } else if (auto constantOp = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(referencedOp)) {
        return vpux::VPURT::BufferSection::CMX_NN;
    } else {
        // TODO: E#98637
        // Until SymRef lookup & interpretation is fixed
        log.trace("Potentially wrong buffer location for {0}", symRef);
        return VPURT::BufferSection::DDR;
    }
}

vpux::VPURT::BufferSection vpux::VPUASM::getBufferLocation(ELF::SymbolReferenceMap& symRefMap,
                                                           mlir::SymbolRefAttr symRef, Logger log) {
    auto referencedOp = symRefMap.lookupSymbol(symRef);

    if (auto bufferOp = mlir::dyn_cast<VPUASM::DeclareBufferOp>(referencedOp)) {
        return bufferOp.getBufferType().getLocation().getSection();
    } else if (auto constantOp = mlir::dyn_cast<VPUASM::ConstBufferOp>(referencedOp)) {
        return constantOp.getBufferType().getLocation().getSection();
    } else if (auto constantOp = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(referencedOp)) {
        return vpux::VPURT::BufferSection::CMX_NN;
    } else {
        // TODO: E#98637
        // Until SymRef lookup & interpretation is fixed
        log.trace("Potentially wrong buffer location for {0}", symRef);
        return VPURT::BufferSection::DDR;
    }
}

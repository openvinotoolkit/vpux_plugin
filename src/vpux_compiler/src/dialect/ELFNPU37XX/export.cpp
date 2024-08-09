//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"

using namespace vpux;

std::vector<uint8_t> vpux::ELFNPU37XX::exportToELF(mlir::ModuleOp module,
                                                   const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                                   const std::vector<std::shared_ptr<const ov::Node>>& results,
                                                   Logger log) {
    log.setName("ELF BackEnd");

    log.trace("Extract '{0}' from Module (ELF File)", IE::CNNNetworkOp::getOperationName());

    elf::Writer elfWriter;
    // Associate the respective mlir::Operation* of
    //   CreateSectionOp/CreateLogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    // Normally the ELF serialization process requires raw data to be serialized first,
    // then symbols, then relocations, simply because of data dependency.
    // However ticket #29166 plans to introduce ordering constraints on IR validity,
    // which would allow us to serialize all data in one function and just use
    // a SectionInterface rather than ops themselves.

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    log.trace("Serializing '{0}' ops", ELFNPU37XX::CreateMetadataSectionOp::getOperationName());
    auto createMetadataSectionOps = netFunc.getOps<ELFNPU37XX::CreateMetadataSectionOp>();
    for (auto createMetadataSectionOp : createMetadataSectionOps) {
        auto metadataPtr = vpux::ELFNPU37XX::constructMetadata(module, netOp, netFunc, parameters, results, log.nest());
        auto& metadata = *metadataPtr.get();
        createMetadataSectionOp.serialize(elfWriter, sectionMap, symbolMap, metadata);
    }

    auto createProfilingSectionOps = netFunc.getOps<ELFNPU37XX::CreateProfilingSectionOp>();
    if (!createProfilingSectionOps.empty()) {
        log.trace("Serializing '{0}' ops", ELFNPU37XX::CreateProfilingSectionOp::getOperationName());
        for (auto createProfilingSectionOp : createProfilingSectionOps) {
            createProfilingSectionOp.serialize(elfWriter, sectionMap, symbolMap);
        }
    }

    log.trace("Serializing '{0}' ops", ELFNPU37XX::CreateSectionOp::getOperationName());
    auto createSectionOps = netFunc.getOps<ELFNPU37XX::CreateSectionOp>();
    for (auto createSectionOp : createSectionOps) {
        createSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", ELFNPU37XX::CreateLogicalSectionOp::getOperationName());
    auto createLogicalSectionOps = netFunc.getOps<ELFNPU37XX::CreateLogicalSectionOp>();
    for (auto createLogicalSectionOp : createLogicalSectionOps) {
        createLogicalSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", ELFNPU37XX::CreateSymbolTableSectionOp::getOperationName());
    auto createSymTabOps = netFunc.getOps<ELFNPU37XX::CreateSymbolTableSectionOp>();
    for (auto createSymTabOp : createSymTabOps) {
        createSymTabOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", ELFNPU37XX::CreateRelocationSectionOp::getOperationName());
    auto createRelocSectionOps = netFunc.getOps<ELFNPU37XX::CreateRelocationSectionOp>();
    for (auto createRelocSection : createRelocSectionOps) {
        createRelocSection.serialize(elfWriter, sectionMap, symbolMap);
    }

    return elfWriter.generateELF();
}

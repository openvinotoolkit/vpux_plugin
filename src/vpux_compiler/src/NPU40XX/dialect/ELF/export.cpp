//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/export.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/metadata.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <vpux_elf/writer.hpp>

using namespace vpux;

std::vector<uint8_t> vpux::ELF::exportToELF(mlir::ModuleOp module,
                                            const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                            const std::vector<std::shared_ptr<const ov::Node>>& results, Logger log) {
    log.setName("ELF BackEnd");

    elf::Writer elfWriter;
    // Associate the respective mlir::Operation* of
    //   DataSectionOp/LogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    vpux::ELF::SymbolReferenceMap symRefMap(elfMain, true);

    log.trace("Serializing '{0}' ops", ELF::CreateMetadataSectionOp::getOperationName());
    auto createMetadataSectionOps = elfMain.getOps<ELF::CreateMetadataSectionOp>();
    for (auto createMetadataSectionOp : createMetadataSectionOps) {
        auto metadataPtr = vpux::ELFNPU37XX::constructMetadata(module, netOp, netFunc, parameters, results);
        auto& metadata = *metadataPtr.get();
        createMetadataSectionOp.serialize(elfWriter, sectionMap, symbolMap, metadata);
    }

    auto createProfSectionOps = to_small_vector(elfMain.getOps<ELF::CreateProfilingSectionOp>());
    if (!createProfSectionOps.empty()) {
        VPUX_THROW_UNLESS(createProfSectionOps.size() == 1, "Expected exactly one CreateProfilingSectionOp. Got {0}",
                          createProfSectionOps.size());
        log.trace("Serializing '{0}' ops", ELF::CreateProfilingSectionOp::getOperationName());
        auto createProfSectionOp = createProfSectionOps[0];
        createProfSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", ELF::DataSectionOp::getOperationName());
    auto dataSectionOps = elfMain.getOps<ELF::DataSectionOp>();
    for (auto dataSectionOp : dataSectionOps) {
        dataSectionOp.serialize(elfWriter, sectionMap, symbolMap, symRefMap);
    }

    log.trace("Serializing '{0}' ops", ELF::LogicalSectionOp::getOperationName());
    auto logicalSectionOps = elfMain.getOps<ELF::LogicalSectionOp>();
    for (auto logicalSectionOp : logicalSectionOps) {
        logicalSectionOp.serialize(elfWriter, sectionMap, symbolMap, symRefMap);
    }

    log.trace("Serializing '{0}' ops", ELF::CreateSymbolTableSectionOp::getOperationName());
    auto symTabOps = elfMain.getOps<ELF::CreateSymbolTableSectionOp>();
    for (auto symTabOp : symTabOps) {
        symTabOp.serialize(elfWriter, sectionMap, symbolMap, symRefMap);
    }

    log.trace("Serializing '{0}' ops", ELF::CreateRelocationSectionOp::getOperationName());
    auto relocSectionOps = elfMain.getOps<ELF::CreateRelocationSectionOp>();
    for (auto relocSectionOp : relocSectionOps) {
        relocSectionOp.serialize(elfWriter, sectionMap, symbolMap, symRefMap);
    }

    return elfWriter.generateELF();
}

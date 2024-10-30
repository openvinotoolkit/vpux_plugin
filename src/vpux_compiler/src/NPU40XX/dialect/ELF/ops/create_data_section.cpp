//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/loop.hpp"

using namespace vpux;

void ELF::DataSectionOp::serialize(elf::Writer& writer, ELF::SectionMapType& sectionMap, ELF::SymbolMapType& symbolMap,
                                   ELF::SymbolReferenceMap& symRefMap) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(symbolMap);

    const auto section = sectionMap.find(getOperation());
    VPUX_THROW_WHEN(section == sectionMap.end(), "ELF section not found: {0}", getSymName().str());
    auto binDataSection = dynamic_cast<elf::writer::BinaryDataSection<uint8_t>*>(section->second);
    VPUX_THROW_WHEN(binDataSection == nullptr, "Invalid binary section in ELF writer");

    auto& blockOps = getBody()->getOperations();
    if (blockOps.empty()) {
        return;
    }

    // All operations are checked since there could be Pad operations in the section along constants
    // E#136963: this could be refactored to check for the section type/flags/name instead
    const auto containsConstants = llvm::any_of(blockOps, [](mlir::Operation& op) {
        return mlir::isa<VPUASM::ConstBufferOp>(op);
    });
    if (containsConstants) {
        std::vector<mlir::Operation*> ops(blockOps.size());
        llvm::transform(blockOps, ops.begin(), [&](mlir::Operation& op) {
            return &op;
        });

        loop_1d(LoopExecPolicy::Parallel, getContext(), static_cast<int64_t>(ops.size()), [&](int64_t opIdx) {
            auto* op = ops[opIdx];
            // Pad operations are skipped, as constants have the memory offset pre-calculated to include this extra
            // padding (see SetOpOffsetsPass)
            if (mlir::isa<ELF::PadOp>(op)) {
                return;
            }
            VPUX_THROW_WHEN(!mlir::isa<VPUASM::ConstBufferOp>(op), "Unexpected operation {0} in data section",
                            op->getName());
            auto constOp = mlir::cast<VPUASM::ConstBufferOp>(op);
            constOp.serializeCached(*binDataSection, symRefMap);
            // Note: after serialization, the constant data is no longer needed,
            // so clean it up - if the data buffer is created by the compiler,
            // it should get deallocated here.
            constOp.getProperties().setContent({});
        });
    } else {
        for (auto& op : blockOps) {
            if (auto binaryOp = llvm::dyn_cast<ELF::BinaryOpInterface>(op)) {
                binaryOp.serializeCached(*binDataSection, symRefMap);
            }
        }
    }
}

void vpux::ELF::DataSectionOp::preserialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                            vpux::ELF::SymbolReferenceMap& symRefMap) {
    const auto name = getSymName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(name, static_cast<uint32_t>(getSecType()));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    size_t sectionSize = 0;
    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<ELF::BinaryOpInterface::Trait>()) {
            auto binaryOp = mlir::cast<ELF::BinaryOpInterface>(op);
            sectionSize += binaryOp.getBinarySizeCached(symRefMap);
        }
    }
    section->setSize(sectionSize);

    sectionMap[getOperation()] = section;
}

ELF::SymbolSignature ELF::DataSectionOp::getSymbolSignature() {
    auto symName = ELF::SymbolOp::getDefaultNamePrefix() + getSymName();
    return {mlir::SymbolRefAttr::get(getSymNameAttr()), symName.str(), ELF::SymbolType::STT_SECTION};
}

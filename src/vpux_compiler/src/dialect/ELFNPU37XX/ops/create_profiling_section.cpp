//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/attributes.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

void vpux::ELFNPU37XX::CreateProfilingSectionOp::serialize(elf::Writer& writer,
                                                           vpux::ELFNPU37XX::SectionMapType& sectionMap,
                                                           vpux::ELFNPU37XX::SymbolMapType& symbolMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = getSecName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(
            name, static_cast<elf::Elf_Word>(vpux::ELFNPU37XX::SectionTypeAttr::VPU_SHT_PROF));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (auto metadataOp = mlir::dyn_cast<vpux::VPUMI37XX::ProfilingMetadataOp>(op)) {
            metadataOp.serialize(*section);
        }
    }

    sectionMap[getOperation()] = section;
}

mlir::LogicalResult vpux::ELFNPU37XX::CreateProfilingSectionOp::verify() {
    auto loc = getLoc();

    bool seenMetadataOp = false;
    for (auto& op : getBody()->getOperations()) {
        if (auto metadataOp = mlir::dyn_cast<vpux::VPUMI37XX::ProfilingMetadataOp>(op)) {
            if (seenMetadataOp) {
                return errorAt(loc,
                               "There should be only 1 ProfilingMetadata op in CreateProfilingSection, found multiple");
            }
            seenMetadataOp = true;
        }
    }

    return mlir::success();
}

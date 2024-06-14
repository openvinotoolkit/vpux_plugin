//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"

void vpux::ELFNPU37XX::CreateSectionOp::serialize(elf::Writer& writer, vpux::ELFNPU37XX::SectionMapType& sectionMap,
                                                  vpux::ELFNPU37XX::SymbolMapType& symbolMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = getSecName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(name, static_cast<uint32_t>(getSecType()));
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<vpux::ELFNPU37XX::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<vpux::ELFNPU37XX::BinaryOpInterface>(op);

            binaryOp.serialize(*section);
        }
    }

    sectionMap[getOperation()] = section;
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

//
// ConstBufferOp
//

void VPUASM::ConstBufferOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    vpux::Const::Content cnt = getContent();

    const auto size = cnt.getType().getTotalAllocSize().count();
    auto tmpBuf = std::make_unique<char[]>(size);
    MutableArrayRef<char> buf(tmpBuf.get(), size);
    cnt.copyTo(buf);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(tmpBuf.get());
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t VPUASM::ConstBufferOp::getBinarySize() {
    auto content = getContentAttr();

    return content.getType().getTotalAllocSize().count();
}

size_t VPUASM::ConstBufferOp::getAlignmentRequirements() {
    // TODO: E#59169 measure if weights alignment has any impact on performance.
    return ELF::VPUX_DEFAULT_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr VPUASM::ConstBufferOp::getAccessingProcs(mlir::SymbolUserMap& symbolUserMap) {
    auto flags = ELF::SectionFlagsAttr::SHF_NONE;

    const auto users = symbolUserMap.getUsers(getOperation());
    VPUX_THROW_WHEN(users.empty(), "It's unexpected for ConstBufferOp to don't have users: {0}", *this);
    for (auto user : users) {
        flags = flags | mlir::cast<ELF::WrappableOpInterface>(user).getUserProcs();
    }

    return flags;
}

vpux::ELF::SectionFlagsAttr VPUASM::ConstBufferOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ConstBufferOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("buffer", getBufferType(), "constant"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ConstBufferOp::hasMemoryFootprint() {
    return true;
}

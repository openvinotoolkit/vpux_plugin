//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <cstring>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;
using MIVersionNote = elf::elf_note::VersionNote;

void vpux::VPUASM::MappedInferenceVersionOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for
    // VPUASM::MappedInferenceVersionOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage ", getOperationName());
#endif
}

size_t vpux::VPUASM::MappedInferenceVersionOp::getBinarySize() {
    return sizeof(MIVersionNote);
}

size_t vpux::VPUASM::MappedInferenceVersionOp::getAlignmentRequirements() {
    return alignof(MIVersionNote);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::MappedInferenceVersionOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("note", "MappedInferenceVersion"),
                                 ELF::SectionFlagsAttr::SHF_NONE, ELF::SectionTypeAttr::SHT_NOTE);
}

bool vpux::VPUASM::MappedInferenceVersionOp::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::MappedInferenceVersionOp::build(mlir::OpBuilder& builder, mlir::OperationState& state) {
    build(builder, state, "MappedInferenceVersion");
}

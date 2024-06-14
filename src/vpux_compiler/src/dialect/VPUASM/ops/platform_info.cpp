//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <cstring>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux_headers/platform.hpp"

using namespace vpux;

void vpux::VPUASM::PlatformInfoOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    elf::platform::PlatformInfo platformInfo;

    platformInfo.mArchKind = ELF::mapVpuArchKindToElfArchKind(getArchKind());

    auto serializedPlatformInfo = elf::platform::PlatformInfoSerialization::serialize(platformInfo);
    binDataSection.appendData(&serializedPlatformInfo[0], serializedPlatformInfo.size());
}

size_t vpux::VPUASM::PlatformInfoOp::getBinarySize() {
    return sizeof(elf::platform::PlatformInfo);
}

size_t vpux::VPUASM::PlatformInfoOp::getAlignmentRequirements() {
    return alignof(elf::platform::PlatformInfo);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::PlatformInfoOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::PlatformInfoOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::PlatformInfoOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("meta", "PlatformInfo"), ELF::SectionFlagsAttr::SHF_NONE,
                                 ELF::SectionTypeAttr::VPU_SHT_PLATFORM_INFO);
}

bool vpux::VPUASM::PlatformInfoOp::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::PlatformInfoOp::build(mlir::OpBuilder& builder, mlir::OperationState& state) {
    build(builder, state, "PlatformInfo", VPU::ArchKind::UNKNOWN);
}

void vpux::VPUASM::PlatformInfoOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                         VPU::ArchKind archKind) {
    build(builder, state, "PlatformInfo", archKind);
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;

//
// MappedInferenceOp
//

void vpux::VPUASM::MappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for ActKernelRangeOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::MappedInferenceOp::getBinarySize(VPU::ArchKind /*arch*/) {
    return sizeof(npu40xx::nn_public::VpuMappedInference);
}

size_t vpux::VPUASM::MappedInferenceOp::getAlignmentRequirements(VPU::ArchKind /*arch*/) {
    return alignof(npu40xx::nn_public::VpuMappedInference);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::MappedInferenceOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::MappedInferenceOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "mapped_inference"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::MappedInferenceOp::hasMemoryFootprint() {
    return true;
}

//
// MappedInferenceOp_37XX
//

void vpux::VPUASM::MappedInferenceOp_37XX::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for ActKernelRangeOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
}

size_t vpux::VPUASM::MappedInferenceOp_37XX::getBinarySize(VPU::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuMappedInference);
}

size_t vpux::VPUASM::MappedInferenceOp_37XX::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(npu40xx::nn_public::VpuMappedInference);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::MappedInferenceOp_37XX::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::MappedInferenceOp_37XX::getSectionSignature() {
    return ELF::SectionSignature("text.mappedInference", ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::MappedInferenceOp_37XX::hasMemoryFootprint() {
    return true;
}

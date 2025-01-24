//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;

void vpux::VPUASM::DPUInvariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for DPUInvariantOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
    return;
}

size_t vpux::VPUASM::DPUInvariantOp::getBinarySize(VPU::ArchKind /*arch*/) {
    return sizeof(npu40xx::nn_public::VpuDPUInvariant);
}

size_t vpux::VPUASM::DPUInvariantOp::getAlignmentRequirements(VPU::ArchKind /*arch*/) {
    return alignof(npu40xx::nn_public::VpuDPUInvariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUInvariantOp::getPredefinedMemoryAccessors() {
    // DPU can't access DDR, therefore DPU descriptors are copied from DDR to metadata in CMX by DMA
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUInvariantOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "dpu", "invariant", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUInvariantOp::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::DPUVariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for Variant
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
    return;
}

size_t vpux::VPUASM::DPUVariantOp::getBinarySize(VPU::ArchKind /*arch*/) {
    return sizeof(npu40xx::nn_public::VpuDPUVariant);
}

size_t vpux::VPUASM::DPUVariantOp::getAlignmentRequirements(VPU::ArchKind /*arch*/) {
    return alignof(npu40xx::nn_public::VpuDPUVariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUVariantOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUVariantOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "dpu", "variant", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUVariantOp::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::DPUInvariantOp_37XX::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for DPUInvariantOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
    return;
}

size_t vpux::VPUASM::DPUInvariantOp_37XX::getBinarySize(VPU::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuDPUInvariant);
}

size_t vpux::VPUASM::DPUInvariantOp_37XX::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(npu40xx::nn_public::VpuDPUInvariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUInvariantOp_37XX::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUInvariantOp_37XX::getSectionSignature() {
    return ELF::SectionSignature("text.invariants", ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUInvariantOp_37XX::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::DPUVariantOp_37XX::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for DPUVariantOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
    return;
}

size_t vpux::VPUASM::DPUVariantOp_37XX::getBinarySize(VPU::ArchKind) {
    return sizeof(npu40xx::nn_public::VpuDPUVariant);
}

size_t vpux::VPUASM::DPUVariantOp_37XX::getAlignmentRequirements(VPU::ArchKind) {
    return alignof(npu40xx::nn_public::VpuDPUVariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUVariantOp_37XX::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUVariantOp_37XX::getSectionSignature() {
    return ELF::SectionSignature("text.variants", ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUVariantOp_37XX::hasMemoryFootprint() {
    return true;
}

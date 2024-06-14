//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/compiler/utils/traits_utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

void vpux::VPUASM::DPUInvariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>&) {
    // TODO: E#80148 after interface refactoring should we not require serialization for DPUInvariantOp
#ifdef VPUX_DEVELOPER_BUILD
    auto logger = Logger::global();
    logger.warning("Serializing {0} op, which may mean invalid usage");
#endif
    return;
}

size_t vpux::VPUASM::DPUInvariantOp::getBinarySize() {
    return sizeof(nn_public::VpuDPUInvariant);
}

size_t vpux::VPUASM::DPUInvariantOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUInvariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUInvariantOp::getAccessingProcs(mlir::SymbolUserMap&) {
    // DPU can't access DDR, therefore DPU descriptors are copied from DDR to metadata in CMX by DMA
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUInvariantOp::getUserProcs() {
    // DPU can access only CMX
    return ELF::SectionFlagsAttr::SHF_NONE;
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

size_t vpux::VPUASM::DPUVariantOp::getBinarySize() {
    return sizeof(nn_public::VpuDPUVariant);
}

size_t vpux::VPUASM::DPUVariantOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUVariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUVariantOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUVariantOp::getUserProcs() {
    // DPU can access only CMX
    return ELF::SectionFlagsAttr::SHF_NONE;
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

size_t vpux::VPUASM::DPUInvariantOp_37XX::getBinarySize() {
    return sizeof(nn_public::VpuDPUInvariant);
}

size_t vpux::VPUASM::DPUInvariantOp_37XX::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUInvariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUInvariantOp_37XX::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUInvariantOp_37XX::getUserProcs() {
    // DPU can access only CMX
    return ELF::SectionFlagsAttr::SHF_NONE;
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

size_t vpux::VPUASM::DPUVariantOp_37XX::getBinarySize() {
    return sizeof(nn_public::VpuDPUVariant);
}

size_t vpux::VPUASM::DPUVariantOp_37XX::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUVariant);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUVariantOp_37XX::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::DPUVariantOp_37XX::getUserProcs() {
    // DPU can access only CMX
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::DPUVariantOp_37XX::getSectionSignature() {
    return ELF::SectionSignature("text.variants", ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::DPUVariantOp_37XX::hasMemoryFootprint() {
    return true;
}

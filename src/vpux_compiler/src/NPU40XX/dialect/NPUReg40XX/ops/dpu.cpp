//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;

namespace vpux {
namespace NPUReg40XX {

void DPUInvariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto invariantDesc = getDpuInvariantDescriptor().getRegMapped();

    VPUX_THROW_UNLESS(Byte(sizeof(nn_public::VpuDPUInvariant)) == invariantDesc.getWidth(),
                      "HW VpuDPUInvariant size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuDPUInvariant), invariantDesc.getWidth());

    auto serializedInvariantDesc = invariantDesc.serialize();
    binDataSection.appendData(serializedInvariantDesc.data(), serializedInvariantDesc.size());
}

size_t DPUInvariantOp::getBinarySize() {
    return sizeof(nn_public::VpuDPUInvariant);
}

std::vector<ELF::RelocationInfo> DPUInvariantOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto regsOffset = offsetof(nn_public::VpuDPUInvariant, registers_);
    auto opType = getNceTaskType();

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    //
    // Input Relocs
    //
    {
        auto inputSymRef = getInput();
        auto addend = ELF::getOffsetOfSymRef(symRefMap, inputSymRef);
        // SOH disabled for NPUReg40XX

        // input address gets relocated 3/4 times and gets masked with CMX_BASE_ADDR_MSK = 0x001F'FFFF
        if (opType != VPUIP::NCETaskType::ELTWISE) {
            relocs.push_back(ELF::RelocationInfo(
                    inputSymRef, targetSection,
                    regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[0]),
                    ELF::RelocationType::R_VPU_LO_21, addend, "Input (act_offset[0]) in DPU invariant reloc"));
        } else {
            relocs.push_back(
                    ELF::RelocationInfo(inputSymRef, targetSection,
                                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, tensor_start),
                                        ELF::RelocationType::R_VPU_LO_21_RSHIFT_4, addend,
                                        "tensor_start (for ELTWISE) in DPU invariant reloc"));
        }
        relocs.push_back(ELF::RelocationInfo(
                inputSymRef, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[1]),
                ELF::RelocationType::R_VPU_LO_21, addend, "Input (act_offset[1]) in DPU invariant reloc"));
        relocs.push_back(ELF::RelocationInfo(
                inputSymRef, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[2]),
                ELF::RelocationType::R_VPU_LO_21, addend, "Input (act_offset[2]) in DPU invariant reloc"));
        relocs.push_back(ELF::RelocationInfo(
                inputSymRef, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[3]),
                ELF::RelocationType::R_VPU_LO_21, addend, "Input (act_offset[3]) in DPU invariant reloc"));
    }

    //
    // Input Sparsity Map Relocs
    //
    if (auto inputSparsityMap = getInputSparsityMap().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, inputSparsityMap);
        relocs.push_back(ELF::RelocationInfo(inputSparsityMap, targetSection,
                                             regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, sparsity_addr),
                                             ELF::RelocationType::R_VPU_LO_21, addend,
                                             "Input sparsity map in DPU invariant reloc"));

        if (opType == VPUIP::NCETaskType::ELTWISE) {
            auto elop_addend = addend;
            if (auto tensorBSparsityMapSymRef = getWeightsSparsityMap().value_or(nullptr)) {
                elop_addend = ELF::getOffsetOfSymRef(symRefMap, tensorBSparsityMapSymRef);
            }
            relocs.push_back(
                    ELF::RelocationInfo(inputSparsityMap, targetSection,
                                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, elops_sparsity_addr),
                                        ELF::RelocationType::R_VPU_LO_21, elop_addend,
                                        "Input sparsity map (for ELTWISE) in DPU invariant reloc"));
        }
    }

    //
    // Input SE Table Relocs
    //
    if (auto inputSETable = getInputStorageElementTable().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, inputSETable);
        relocs.push_back(ELF::RelocationInfo(
                inputSETable, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_addr),
                ELF::RelocationType::R_VPU_LO_21, addend, "Input SE table (se_addr) in DPU invariant reloc"));
    }

    //
    // Weights Relocs
    //
    // wt_offset needs to be set even if there is no weights operand in MAXPOOL & shouldn't be set for
    // ELTWISE
    if ((getWeights().value_or(nullptr) && opType != VPUIP::NCETaskType::ELTWISE) ||
        opType == VPUIP::NCETaskType::MAXPOOL) {
        // Relocation needs to setup wt_offset to the base CMX addr
        auto addend = 0;
        auto newInput = getInput();

        relocs.push_back(ELF::RelocationInfo(
                newInput, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, wt_offset),
                ELF::RelocationType::R_VPU_LO_21, addend,
                "Weights (wt_offset) in DPU invariant reloc"));  // Using input as source just as a placeholder
    }

    //
    // Tensor2 Relocs
    //
    // tensor2_start needs to be set for dual elops (ELTWISE with 2 input tensors) in case of per output channel
    // scaling, i.e. when weightTable is provided
    if (getWeightTable().has_value() && opType == VPUIP::NCETaskType::ELTWISE) {
        if (auto weights = getWeights().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, weights);
            relocs.push_back(ELF::RelocationInfo(weights, targetSection,
                                                 regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, reserved1),
                                                 ELF::RelocationType::R_VPU_LO_21_RSHIFT_4, addend,
                                                 "Weights (for ELTWISE, reserved1) in DPU invariant reloc"));
        }
    }

    //
    // sprLookupTable Relocs
    //
    if (auto sprLookupTable = getSprLookupTable().value_or(nullptr)) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, sprLookupTable);
        relocs.push_back(ELF::RelocationInfo(sprLookupTable, targetSection,
                                             regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, reserved2),
                                             ELF::RelocationType::R_VPU_16_LSB_17_RSHIFT_5, addend,
                                             "Spr lookup table (reserved2) in DPU invariant reloc"));
    }

    //
    // Output Relocs
    //
    // no output in case of continued conv
    if (!getIsContinued() && getOutput().has_value()) {
        auto outputSymRef = getOutput().value().cast<mlir::SymbolRefAttr>();
        auto addend = ELF::getOffsetOfSymRef(symRefMap, outputSymRef);
        relocs.push_back(ELF::RelocationInfo(
                outputSymRef, targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, odu_ac_base),
                ELF::RelocationType::R_VPU_LO_21, addend, "Output (odu_ac_base) in DPU invariant reloc"));

        if (auto outputSparsityMap = getOutputSparsityMap().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, outputSparsityMap);
            relocs.push_back(ELF::RelocationInfo(outputSparsityMap, targetSection,
                                                 regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, sp_base),
                                                 ELF::RelocationType::R_VPU_LO_21_MULTICAST_BASE, addend,
                                                 "Output sparsity map (sp_base) in DPU invariant reloc"));
        }
    }

    //
    // HWP Relocs
    //
    if (getProfilingData().has_value()) {
        // Relocation needs to setup hwp_cmx_mem_addr to the base CMX addr
        auto addend = 0;
        relocs.push_back(ELF::RelocationInfo(
                getInput(), targetSection, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, hwp_cmx_mem_addr),
                ELF::RelocationType::R_VPU_CMX_LOCAL_RSHIFT_5, addend,
                "HWP (hwp_cmx_mem_addr) in DPU invariant reloc"));  // Using input as source just as a placeholder
    }

    // set invariant pointer. workaround preemtion: #E-97614
    auto addend = ELF::getOffsetOfSymRef(symRefMap, getTaskLocation().value());
    relocs.push_back(ELF::RelocationInfo(getTaskLocation().value(), targetSection,
                                         offsetof(nn_public::VpuDPUInvariant, registers_.tensor_mode),
                                         ELF::RelocationType::R_VPU_16_LSB_17_RSHIFT_5_LSHIFT_16, addend,
                                         "Invariant pointer in DPU invariant reloc"));

    return relocs;
}

size_t DPUInvariantOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUInvariant);
}

std::optional<ELF::SectionSignature> DPUInvariantOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "dpu", "invariant", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool DPUInvariantOp::hasMemoryFootprint() {
    return true;
}

void DPUVariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto variantDesc = getDpuVariantDescriptor().getRegMapped();

    VPUX_THROW_UNLESS(Byte(sizeof(nn_public::VpuDPUVariant)) == variantDesc.getWidth(),
                      "HW VpuDPUVariant size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuDPUVariant), variantDesc.getWidth());

    auto serializedVariantDesc = variantDesc.serialize();
    binDataSection.appendData(serializedVariantDesc.data(), serializedVariantDesc.size());
}

size_t DPUVariantOp::getBinarySize() {
    return sizeof(nn_public::VpuDPUVariant);
}

std::vector<ELF::RelocationInfo> DPUVariantOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto regsOffset = offsetof(nn_public::VpuDPUVariant, registers_);
    auto opType = getNceTaskType();

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    // Important!
    // weight_start needs to be preset at serialization to the value of variant.weight_table_offset_
    if (auto weightTable = getWeightTable().value_or(nullptr)) {
        // weight_start reloc R_32_SUM with weight_table sym
        auto addend = ELF::getOffsetOfSymRef(symRefMap, weightTable);
        relocs.push_back(ELF::RelocationInfo(
                weightTable, targetSection, regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, weight_start),
                ELF::RelocationType::R_VPU_LO_21_SUM, addend, "Weights (weight_start) in DPU variant reloc"));
    } else if (opType == VPUIP::NCETaskType::ELTWISE) {
        if (auto weights = getWeights().value_or(nullptr)) {
            auto addend = ELF::getOffsetOfSymRef(symRefMap, weights);
            relocs.push_back(ELF::RelocationInfo(weights, targetSection,
                                                 regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, weight_start),
                                                 ELF::RelocationType::R_VPU_LO_21_RSHIFT_4, addend,
                                                 "Weights (for ELTWISE, weight_start) in DPU variant reloc"));
        }
    }

    // Set the additional invariant pointer in the VpuDPUVariant struct
    if (auto linkedInvariant = getInvariantTaskLocation()) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, linkedInvariant);
        relocs.push_back(ELF::RelocationInfo(linkedInvariant, targetSection,
                                             offsetof(nn_public::VpuDPUVariant, invariant_) +
                                                     offsetof(nn_public::VpuPtr<nn_public::VpuDPUInvariant>, ptr),
                                             ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET, addend,
                                             "Invariant pointer in DPU variant reloc"));

        relocs.push_back(ELF::RelocationInfo(linkedInvariant, targetSection,
                                             regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, invar_ptr),
                                             ELF::RelocationType::R_VPU_16_LSB_17_RSHIFT_5, addend,
                                             "Invariant pointer register in DPU variant registers reloc"));
    }

    // set variant pointer. workaround preemtion: #E-97614
    auto addend = ELF::getOffsetOfSymRef(symRefMap, getTaskLocation().value());
    relocs.push_back(ELF::RelocationInfo(getTaskLocation().value(), targetSection,
                                         offsetof(nn_public::VpuDPUVariant, registers_.dpu_cfg),
                                         ELF::RelocationType::R_VPU_16_LSB_17_RSHIFT_5_LSHIFT_CUSTOM, addend,
                                         "Variant pointer in DPU variant reloc"));

    if (auto nextLink = getNextLinkAttr()) {
        auto addend = ELF::getOffsetOfSymRef(symRefMap, nextLink);
        relocs.push_back(ELF::RelocationInfo(nextLink, targetSection,
                                             regsOffset + offsetof(nn_public::VpuDPUVariantRegisters, var_cfg),
                                             ELF::RelocationType::R_VPU_16_LSB_17_RSHIFT_5_LSHIFT_16, addend,
                                             "Next link (var_cfg) in DPU variant reloc"));
    }

    return relocs;
}

size_t DPUVariantOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUVariant);
}

std::optional<ELF::SectionSignature> DPUVariantOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("task", "dpu", "variant", getTaskIndex()),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool DPUVariantOp::hasMemoryFootprint() {
    return true;
}

}  // namespace NPUReg40XX
}  // namespace vpux

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

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

size_t vpux::VPUASM::MappedInferenceOp::getBinarySize() {
    return sizeof(nn_public::VpuMappedInference);
}

size_t vpux::VPUASM::MappedInferenceOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuMappedInference);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::MappedInferenceOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::MappedInferenceOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::MappedInferenceOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "mapped_inference"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::MappedInferenceOp::hasMemoryFootprint() {
    return true;
}

namespace {
size_t getSymRefOffsetForReloc(VPUASM::MappedInferenceOp op, mlir::SymbolRefAttr ref) {
    // To be removed after E#90466
    auto dmaTaskReferenceOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuDMATask>, address);
    for (size_t dmaEngine = 0; dmaEngine < op.getDmaTasksAttr().size(); dmaEngine++) {
        auto dmaGroup = op.getDmaTasksAttr()[dmaEngine].cast<mlir::ArrayAttr>();
        auto dmaCounts = op.getDmaCount();
        auto dmaCountList = parseIntArrayAttr<int64_t>(dmaCounts[dmaEngine].cast<mlir::ArrayAttr>());
        auto dmaTaskIdxOffset = (dmaEngine * sizeof(nn_public::VpuTaskReference<nn_public::VpuDMATask>));

        // by default we expect that getDmaTasksAttr
        // will return a list that looks like
        // index 0 -> DDR dmas
        // index 1 -> CMX dmas
        auto cmxIdx = 1;
        if (dmaCountList[0] != 0) {
            auto dmaDDR = dmaGroup[0].cast<mlir::SymbolRefAttr>();
            if (ref == dmaDDR) {
                return offsetof(nn_public::VpuMappedInference, dma_tasks_ddr_[0]) + dmaTaskIdxOffset +
                       dmaTaskReferenceOffset;
            }
        } else {
            cmxIdx = 0;
        }

        // could be cmx dma
        // if there are no DDR dmas for the current task it means that cmx index would be 0
        if (dmaCountList[1] != 0) {
            auto dmaCMX = dmaGroup[cmxIdx].cast<mlir::SymbolRefAttr>();
            if (ref == dmaCMX) {
                return offsetof(nn_public::VpuMappedInference, dma_tasks_cmx_[0]) + dmaTaskIdxOffset +
                       dmaTaskReferenceOffset;
            }
        }
    }
    if (ref == op.getBarrierTasks()) {
        return offsetof(nn_public::VpuMappedInference, barrier_configs) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuBarrierCountConfig>, address);
    }

    if (ref == op.getMediaTasks()) {
        return offsetof(nn_public::VpuMappedInference, media_tasks) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuMediaTask>, address);
    }

    if (ref == op.getActShaveRt()) {
        return offsetof(nn_public::VpuMappedInference, shv_rt_configs) +
               offsetof(nn_public::VpuNNShaveRuntimeConfigs, act_rt_window_base);
    }

    if (ref == op.getDmaHwpBase()) {
        return offsetof(nn_public::VpuMappedInference, logaddr_dma_hwp_);
    }

    if (ref == op.getHwpWorkpointCfg()) {
        return offsetof(nn_public::VpuMappedInference, hwp_workpoint_cfg_addr);
    }

    if (op.getActShaveStacksAttr()) {
        auto shaveStacksRef = llvm::find(op.getActShaveStacksAttr(), ref);
        if (shaveStacksRef != op.getActShaveStacksAttr().end()) {
            auto index = shaveStacksRef - op.getActShaveStacksAttr().begin();
            auto shaveRtConfigOffset = offsetof(nn_public::VpuMappedInference, shv_rt_configs);
            auto stackFramesOffset = offsetof(nn_public::VpuNNShaveRuntimeConfigs, stack_frames);
            auto arrayIdxOffset = index * sizeof(nn_public::VpuNNShaveRuntimeConfigs::stack_frames[0]);
            return shaveRtConfigOffset + stackFramesOffset + arrayIdxOffset;
        }
    }

    auto getTileIdx = [](auto arrayIdx, mlir::ArrayAttr countAttr) -> size_t {
        auto count = parseIntArrayAttr<int64_t>(countAttr);

        auto usedTilesNum = arrayIdx + 1;
        for (size_t countIdx = 0; countIdx < count.size() && usedTilesNum; ++countIdx) {
            if (count[countIdx]) {
                --usedTilesNum;
            }
            if (!usedTilesNum) {
                return countIdx;
            }
        }

        VPUX_THROW_WHEN(usedTilesNum, "Cannot identify the tile corresponding for task of index {0}", arrayIdx);

        return 0;
    };

    if (op.getActKernelInvocationsAttr()) {
        auto shaveInvoRef = llvm::find(op.getActKernelInvocationsAttr(), ref);
        if (shaveInvoRef != op.getActKernelInvocationsAttr().end()) {
            auto index = getTileIdx(shaveInvoRef - op.getActKernelInvocationsAttr().begin(),
                                    op.getActKernelInvocationsCount());
            auto invosOffset = offsetof(nn_public::VpuMappedInference, act_kernel_invocations);
            auto arrayIdxOffset = (index * sizeof(nn_public::VpuTaskReference<nn_public::VpuActKernelInvocation>));
            auto vpuPtrOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuActKernelInvocation>, address);
            return invosOffset + arrayIdxOffset + vpuPtrOffset;
        }
    }

    if (op.getActKernelRangesAttr()) {
        auto shaveRangeRef = llvm::find(op.getActKernelRangesAttr(), ref);
        if (shaveRangeRef != op.getActKernelRangesAttr().end()) {
            auto index = getTileIdx(shaveRangeRef - op.getActKernelRangesAttr().begin(), op.getActKernelRangesCount());
            auto rangesOffset = offsetof(nn_public::VpuMappedInference, act_kernel_ranges);
            auto arrayIdxOffset = (index * sizeof(nn_public::VpuTaskReference<nn_public::VpuActKernelRange>));
            auto vpuPtrOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuActKernelRange>, address);
            return rangesOffset + arrayIdxOffset + vpuPtrOffset;
        }
    }

    if (op.getVariantTasks()) {
        auto variantsRef = llvm::find(op.getVariantTasksAttr(), ref);
        if (variantsRef != op.getVariantTasksAttr().end()) {
            auto index = getTileIdx(variantsRef - op.getVariantTasksAttr().begin(), op.getVariantCount());
            auto variantsOffset = offsetof(nn_public::VpuMappedInference, variants);
            auto arrayIdxOffset = (index * sizeof(nn_public::VpuTaskReference<nn_public::VpuDPUVariant>));
            auto vpuPtrOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuDPUVariant>, address);
            return variantsOffset + arrayIdxOffset + vpuPtrOffset;
        }
    }

    if (op.getInvariantTasks()) {
        auto invariantsRef = llvm::find(op.getInvariantTasksAttr(), ref);
        if (invariantsRef != op.getInvariantTasksAttr().end()) {
            auto index = getTileIdx(invariantsRef - op.getInvariantTasksAttr().begin(), op.getInvariantCount());
            auto invariantsOffset = offsetof(nn_public::VpuMappedInference, invariants);
            auto arrayIdxOffset = (index * sizeof(nn_public::VpuTaskReference<nn_public::VpuDPUInvariant>));
            auto vpuPtrOffset = offsetof(nn_public::VpuTaskReference<nn_public::VpuDPUInvariant>, address);
            return invariantsOffset + arrayIdxOffset + vpuPtrOffset;
        }
    }

    if (op.getManagedMappedInferenceAttr()) {
        if (ref == op.getManagedMappedInferenceAttr()) {
            return offsetof(nn_public::VpuMappedInference, managed_inference) +
                   offsetof(nn_public::VpuTaskReference<uint8_t>, address);
        }
    }

    VPUX_THROW("Provided SymbolRefAttr is not linked to the MappedInference Op or getSymRefOffsetForReloc does not "
               "support it");
}
}  // namespace

std::vector<ELF::RelocationInfo> vpux::VPUASM::MappedInferenceOp::getRelocationInfo(
        ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    auto thisMI = *(this);
    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    if (auto dmaTasks = getDmaTasks().value_or(nullptr)) {
        for (auto dmaList : dmaTasks) {
            auto dmaListArrayAttr = dmaList;
            dmaListArrayAttr.walkImmediateSubElements(
                    [&](mlir::Attribute attr) {
                        if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                            relocs.push_back(ELF::RelocationInfo(
                                    symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                    ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef)));
                        }
                    },
                    [](mlir::Type) {});
        }
    }

    if (auto invariantTasks = getInvariantTasks().value_or(nullptr)) {
        auto invTasksSubElemIf = invariantTasks;
        invTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.push_back(ELF::RelocationInfo(
                                symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef)));
                    }
                },
                [](mlir::Type) {});
    }

    if (auto variantTasks = getVariantTasks().value_or(nullptr)) {
        auto varTasksSubElemIf = variantTasks;
        varTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.push_back(ELF::RelocationInfo(
                                symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef)));
                    }
                },
                [](mlir::Type) {});
    }

    if (auto actKernelRanges = getActKernelRanges().value_or(nullptr)) {
        auto akrTasksSubElemIf = actKernelRanges;
        akrTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.push_back(ELF::RelocationInfo(
                                symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef)));
                    }
                },
                [](mlir::Type) {});
    }

    if (auto actKernelInvos = getActKernelInvocations().value_or(nullptr)) {
        auto akiTasksSubElemIf = actKernelInvos;
        akiTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.push_back(ELF::RelocationInfo(
                                symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, symRef)));
                    }
                },
                [](mlir::Type) {});
    }

    if (auto mediaTasks = getMediaTasks().value_or(nullptr)) {
        relocs.push_back(ELF::RelocationInfo(mediaTasks, targetSection, getSymRefOffsetForReloc(thisMI, mediaTasks),
                                             ELF::RelocationType::R_VPU_64,
                                             ELF::getOffsetOfSymRef(symRefMap, mediaTasks)));
    }

    if (auto barrierTasks = getBarrierTasks().value_or(nullptr)) {
        relocs.push_back(ELF::RelocationInfo(barrierTasks, targetSection, getSymRefOffsetForReloc(thisMI, barrierTasks),
                                             ELF::RelocationType::R_VPU_64,
                                             ELF::getOffsetOfSymRef(symRefMap, barrierTasks)));
    }

    if (auto actShaveRt = getActShaveRt().value_or(nullptr)) {
        relocs.push_back(ELF::RelocationInfo(actShaveRt, targetSection, getSymRefOffsetForReloc(thisMI, actShaveRt),
                                             ELF::RelocationType::R_VPU_64,
                                             ELF::getOffsetOfSymRef(symRefMap, actShaveRt)));
    }

    if (auto actShaveStacks = getActShaveStacks().value_or(nullptr)) {
        auto shvStacksTasksSubElemIf = actShaveStacks;
        shvStacksTasksSubElemIf.walkImmediateSubElements(
                [&](mlir::Attribute attr) {
                    if (auto symRef = attr.dyn_cast<mlir::SymbolRefAttr>()) {
                        relocs.push_back(ELF::RelocationInfo(
                                symRef, targetSection, getSymRefOffsetForReloc(thisMI, symRef),
                                ELF::RelocationType::R_VPU_32, ELF::getOffsetOfSymRef(symRefMap, symRef)));
                    }
                },
                [](mlir::Type) {});
    }

    if (auto dmaHwpBase = getDmaHwpBase().value_or(nullptr)) {
        relocs.push_back(ELF::RelocationInfo(dmaHwpBase, targetSection, getSymRefOffsetForReloc(thisMI, dmaHwpBase),
                                             ELF::RelocationType::R_VPU_64,
                                             ELF::getOffsetOfSymRef(symRefMap, dmaHwpBase)));
    }

    if (auto hwpWorkpointCfg = getHwpWorkpointCfg().value_or(nullptr)) {
        relocs.push_back(
                ELF::RelocationInfo(hwpWorkpointCfg, targetSection, getSymRefOffsetForReloc(thisMI, hwpWorkpointCfg),
                                    ELF::RelocationType::R_VPU_64, ELF::getOffsetOfSymRef(symRefMap, hwpWorkpointCfg)));
    }

    if (auto managedMPI = getManagedMappedInference().value_or(nullptr)) {
        relocs.push_back(ELF::RelocationInfo(managedMPI, targetSection, getSymRefOffsetForReloc(thisMI, managedMPI),
                                             ELF::RelocationType::R_VPU_64,
                                             ELF::getOffsetOfSymRef(symRefMap, managedMPI)));
    }

    return relocs;
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

size_t vpux::VPUASM::MappedInferenceOp_37XX::getBinarySize() {
    return sizeof(nn_public::VpuMappedInference);
}

size_t vpux::VPUASM::MappedInferenceOp_37XX::getAlignmentRequirements() {
    return alignof(nn_public::VpuMappedInference);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::MappedInferenceOp_37XX::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::MappedInferenceOp_37XX::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::MappedInferenceOp_37XX::getSectionSignature() {
    return ELF::SectionSignature("text.mappedInference", ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::MappedInferenceOp_37XX::hasMemoryFootprint() {
    return true;
}

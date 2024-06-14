//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ConfigureBarrierOp
//

void vpux::VPUASM::ConfigureBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binaryDataSection) {
    nn_public::VpuBarrierCountConfig barrier{};

    barrier.next_same_id_ = getNextSameId();
    barrier.consumer_count_ = getConsumerCount();
    barrier.producer_count_ = getProducerCount();
    barrier.real_id_ = getId();

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&barrier);
    binaryDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUASM::ConfigureBarrierOp::getBinarySize() {
    return sizeof(nn_public::VpuBarrierCountConfig);
}

size_t vpux::VPUASM::ConfigureBarrierOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuBarrierCountConfig);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ConfigureBarrierOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::SHF_EXECINSTR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ConfigureBarrierOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ConfigureBarrierOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("program", "barrier"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::ConfigureBarrierOp::hasMemoryFootprint() {
    return true;
}

mlir::LogicalResult vpux::VPUASM::ConfigureBarrierOp::verify() {
    const auto nextSameId = getNextSameId();
    const auto currentIndex = getTaskIndex().getValue();
    if (nextSameId != -1) {
        const auto uNextSameId = static_cast<uint32_t>(nextSameId);
        if (currentIndex > uNextSameId) {
            return errorAt(getLoc(),
                           "Operation {0}: barrier next_same_id {1} value is smaller than current index value {2}",
                           getOperationName(), uNextSameId, currentIndex);
        }
    }

    const auto id = getId();
    const auto noOfAvailableBarriers = VPUIP::getNumAvailableBarriers(getOperation());
    if (id >= noOfAvailableBarriers) {
        return errorAt(getLoc(), "Operation {0}: barrier id {0} value is higher than available barriers {1}",
                       getOperationName(), id, noOfAvailableBarriers);
    }

    return ::mlir::success();
}

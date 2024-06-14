//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

using namespace vpux;

void vpux::VPUASM::ProfilingMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto denseMetaAttr = getMetadata().dyn_cast<mlir::DenseElementsAttr>();
    auto buf = denseMetaAttr.getRawData();
    binDataSection.appendData(reinterpret_cast<const uint8_t*>(buf.data()), buf.size());
}

size_t vpux::VPUASM::ProfilingMetadataOp::getBinarySize() {
    auto values = getMetadata().getValues<uint8_t>();
    return values.size();
}

size_t vpux::VPUASM::ProfilingMetadataOp::getAlignmentRequirements() {
    return ELF::VPUX_NO_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ProfilingMetadataOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ProfilingMetadataOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

// The operation is placed inside the section (CreateProfilingSectionOp) by a separate pass,
// see: AddProfilingSectionPass
std::optional<ELF::SectionSignature> vpux::VPUASM::ProfilingMetadataOp::getSectionSignature() {
    return std::nullopt;
}

bool vpux::VPUASM::ProfilingMetadataOp::hasMemoryFootprint() {
    return true;
}

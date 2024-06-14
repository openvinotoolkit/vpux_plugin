//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"

uint32_t vpux::NPUReg40XX::getTileSelectMaskForBuffer(VPUASM::DeclareBufferOp buffer) {
    auto bufferLocation = buffer.getBufferType().getLocation();
    if (bufferLocation.getSection() != VPURT::BufferSection::CMX_NN) {
        return 0;
    }

    auto tileIndex = bufferLocation.getSectionIndex();
    return 1 << (tileIndex + NPUReg40XX::CMX_TILE_SELECT_OFFSET);
}

uint32_t vpux::NPUReg40XX::getTileSelectMaskForBuffer(VPUASM::DeclareTaskBufferOp taskBuffer) {
    return 1 << (taskBuffer.getTileIndex() + NPUReg40XX::CMX_TILE_SELECT_OFFSET);
}

template <class OpType>
OpType vpux::NPUReg40XX::getOpFrom(vpux::ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> attr) {
    if (!attr) {
        return nullptr;
    }
    auto attrRef = _symRefMap.lookupSymbol(attr.value());
    return mlir::dyn_cast_or_null<OpType>(attrRef);
}

uint32_t vpux::NPUReg40XX::getKernelEntry(vpux::ELF::SymbolReferenceMap& _symRefMap,
                                          std::optional<mlir::SymbolRefAttr> attr) {
    auto kernelEntryOp = getOpFrom<vpux::VPUASM::DeclareKernelEntryOp>(_symRefMap, attr);
    return kernelEntryOp ? kernelEntryOp.getKernelEntry() : 0;
}

uint64_t vpux::NPUReg40XX::getKernelTextSize(vpux::ELF::SymbolReferenceMap& _symRefMap,
                                             std::optional<mlir::SymbolRefAttr> attr) {
    auto kernelTextOp = getOpFrom<vpux::VPUASM::DeclareKernelTextOp>(_symRefMap, attr);
    return kernelTextOp ? kernelTextOp.getBinarySize() : 0;
}

llvm::StringRef vpux::NPUReg40XX::getKernelPath(vpux::ELF::SymbolReferenceMap& _symRefMap,
                                                std::optional<mlir::SymbolRefAttr> kernelPath,
                                                mlir::SymbolRefAttr taskType) {
    if (vpux::VPUIP::isCacheOpTaskType(taskType)) {
        static const mlir::DenseMap<VPU::ActShaveTaskType, llvm::StringRef> taskTypeKernelPathMap = {
                {VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE, "cache_flush_invalidate"},
                {VPU::ActShaveTaskType::CACHE_FLUSH, "cache_flush"},
                {VPU::ActShaveTaskType::CACHE_INVALIDATE, "cache_invalidate"},
                {VPU::ActShaveTaskType::CACHE_PREFETCH, "cache_prefetch"}};

        auto cacheTaskType = vpux::VPU::symbolizeActShaveTaskType(taskType.getLeafReference().strref());
        return taskTypeKernelPathMap.at(cacheTaskType.value());
    }

    auto kernelEntryOp = getOpFrom<vpux::VPUASM::DeclareKernelEntryOp>(_symRefMap, kernelPath);
    return kernelEntryOp.getKernelPath();
}

npu40xx::nn_public::VpuActWLType vpux::NPUReg40XX::getActWLType(mlir::SymbolRefAttr taskType) {
    static const mlir::DenseMap<VPU::ActShaveTaskType, npu40xx::nn_public::VpuActWLType> taskTypeActWLTypeMap = {
            {VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE, npu40xx::nn_public::VpuActWLType::WL_CACHE_OP_FLUSHINV},
            {VPU::ActShaveTaskType::CACHE_FLUSH, npu40xx::nn_public::VpuActWLType::WL_CACHE_OP_FLUSH},
            {VPU::ActShaveTaskType::CACHE_INVALIDATE, npu40xx::nn_public::VpuActWLType::WL_CACHE_OP_INVALIDATE},
            {VPU::ActShaveTaskType::CACHE_PREFETCH, npu40xx::nn_public::VpuActWLType::WL_CACHE_OP_PREFETCH},
            {VPU::ActShaveTaskType::COMPUTE, npu40xx::nn_public::VpuActWLType::WL_KERNEL}};

    auto cacheTaskType = vpux::VPU::symbolizeActShaveTaskType(taskType.getLeafReference().strref());
    return taskTypeActWLTypeMap.at(cacheTaskType.value());
}

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <limits>

#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

using vpux::checked_cast;

namespace vpux {

void VPURegMapped::ResolveTaskLocationPass::createTaskLocationBuffers(VPURegMapped::TaskBufferLayoutOp taskLayoutOp,
                                                                      MetadataBuffersContainer& metadataBuffers) {
    auto function = getOperation();
    auto builder = mlir::OpBuilder::atBlockBegin(&function.getBody().front());
    auto context = function.getContext();

    auto populateTaskBuffers = [&](size_t tile, VPURegMapped::TaskType type, const auto& sizesPerTaskType) {
        // order of DeclareTaskBuffer is important as it must be aligned with firmware expectations
        // tile0: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
        // tile1: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
        // ...
        const auto sizesPerList = sizesPerTaskType.lookup(type);
        auto& metadataBuffersPerTaskType = metadataBuffers.data[tile][type];
        metadataBuffersPerTaskType.resize(sizesPerList.size());
        for (const auto& entryPerList : llvm::enumerate(sizesPerList)) {
            const auto list = entryPerList.index();
            const auto sizePerList =
                    entryPerList.value().dynamicSize;  // can be modified from "dynamicSize" to "staticSize" if
                                                       // generating all task buffers is ever needed

            for (auto i : irange(sizePerList)) {
                auto offsetAttr = mlir::IntegerAttr::get(vpux::getUInt64Type(context),
                                                         taskLayoutOp.getTaskBufferOffset(type, tile, list, i));
                auto declareTaskBufferOp = builder.create<VPURegMapped::DeclareTaskBufferOp>(
                        function.getLoc(),
                        vpux::VPURegMapped::IndexType::get(context, checked_cast<uint32_t>(tile),
                                                           checked_cast<uint32_t>(list), checked_cast<uint32_t>(i)),
                        type, offsetAttr);
                metadataBuffersPerTaskType[list].push_back(declareTaskBufferOp);
            }
        }
    };

    metadataBuffers.data.resize(metadataBuffers.sizes.size());
    VPUX_THROW_WHEN(_supportedTaskTypes.empty(), "The _supportedTaskTypes was not populated by the arch-specific pass");
    for (const auto& entryPerTile : llvm::enumerate(metadataBuffers.sizes)) {
        const auto tile = entryPerTile.index();
        const auto& sizesPerTaskType = entryPerTile.value();
        for (auto& taskType : _supportedTaskTypes) {
            populateTaskBuffers(tile, taskType, sizesPerTaskType);
        }
    }

    for (auto task : function.getOps<VPURegMapped::TaskOpInterface>()) {
        const auto type = task.getTaskType();
        const auto index = task.getIndexType();
        const auto& taskBuffers = metadataBuffers.data[index.getTileIdx()][type][index.getListIdx()];

        task.setTaskLocation(taskBuffers[index.getValue() % taskBuffers.size()]);
    }
}

}  // namespace vpux

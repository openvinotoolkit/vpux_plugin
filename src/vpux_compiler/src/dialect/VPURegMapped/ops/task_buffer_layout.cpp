//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/NPU40XX/dialect/ELF/ops.hpp>
#include <vpux/compiler/dialect/VPURegMapped/ops.hpp>

using namespace vpux;
using namespace VPURegMapped;

namespace {

VPURegMapped::TaskGroupAttr getLayoutForTileAndList(mlir::ArrayAttr taskMap, size_t tile, size_t list) {
    auto taskLayoutVec = parseCustomAttrArray<mlir::ArrayAttr>(taskMap);

    VPUX_THROW_WHEN(tile >= taskLayoutVec.size(), "Tile index value {0} is higher than max registered tile index - {1}",
                    tile, taskLayoutVec.size() - 1);
    auto taskLayoutForTile = parseCustomAttrArray<VPURegMapped::TaskGroupAttr>(taskLayoutVec[tile]);

    VPUX_THROW_WHEN(list >= taskLayoutForTile.size(),
                    "List index value {0} is higher than max registered list index - {1}", list,
                    taskLayoutForTile.size() - 1);
    return taskLayoutForTile[list];
}

}  // namespace

//
// TaskBufferLayoutOp
//

mlir::LogicalResult TaskBufferLayoutOp::verify() {
    auto operation = getOperation();

    auto parentOp = operation->getParentOp();
    if (!(mlir::isa_and_nonnull<mlir::func::FuncOp, ELF::MainOp>(parentOp))) {
        return mlir::failure();
    }

    return mlir::success();
}

size_t TaskBufferLayoutOp::getDynamicTaskCount(VPURegMapped::TaskType type, size_t tile, size_t list) {
    auto taskLayoutDict = getTaskListSizeMap();
    auto taskTypeStr = VPURegMapped::stringifyTaskType(type);
    auto taskLayoutAttr = mlir::cast_if_present<mlir::ArrayAttr>(taskLayoutDict.get(taskTypeStr));
    VPUX_THROW_UNLESS(taskLayoutAttr, "No task layout found for {0} task type", taskTypeStr.str());

    auto layout = getLayoutForTileAndList(taskLayoutAttr, tile, list);

    return layout.getDynamicSize().getUInt();
}

size_t TaskBufferLayoutOp::getStaticTaskCount(VPURegMapped::TaskType type, size_t tile, size_t list) {
    auto taskLayoutDict = getTaskListSizeMap();
    auto taskTypeStr = VPURegMapped::stringifyTaskType(type);
    auto taskLayoutAttr = mlir::cast_if_present<mlir::ArrayAttr>(taskLayoutDict.get(taskTypeStr));
    VPUX_THROW_UNLESS(taskLayoutAttr, "No task layout found for {0} task type", taskTypeStr.str());

    auto layout = getLayoutForTileAndList(taskLayoutAttr, tile, list);

    return layout.getStaticSize().getUInt();
}

size_t TaskBufferLayoutOp::getTaskBufferOffset(VPURegMapped::TaskType type, size_t tile, size_t list, size_t index) {
    auto taskLayoutDict = getTaskListSizeMap();
    auto taskTypeStr = VPURegMapped::stringifyTaskType(type);
    auto taskLayoutAttr = mlir::cast_if_present<mlir::ArrayAttr>(taskLayoutDict.get(taskTypeStr));
    VPUX_THROW_UNLESS(taskLayoutAttr, "No task layout found for {0} task type", taskTypeStr.str());

    auto layout = getLayoutForTileAndList(taskLayoutAttr, tile, list);
    auto offset = layout.getOffset().getUInt() + layout.getBinaryElementSize().getUInt() * index;

    return offset;
}

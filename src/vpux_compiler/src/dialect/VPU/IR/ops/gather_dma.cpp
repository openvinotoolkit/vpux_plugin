//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherDMAOp::inferReturnTypes(mlir::MLIRContext*, std::optional<mlir::Location>,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPU::GatherDMAOpAdaptor gatherDMAOp(operands, attrs);
    const auto indicesType = gatherDMAOp.getIndices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();

    const auto axis = gatherDMAOp.getAxisValue().value();

    const auto inputType = gatherDMAOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    auto outputShape = inputShape.toValues();
    outputShape[vpux::Dim(axis)] = indicesShape[vpux::Dim(axis)];

    auto outType = inputType.changeShape(Shape(std::move(outputShape)));
    //  Only DDR->CMX DMA gather is implemented
    const auto memSpaceCMX =
            vpux::IndexedSymbolAttr::get(indicesType.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    outType = outType.changeMemSpace(memSpaceCMX);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

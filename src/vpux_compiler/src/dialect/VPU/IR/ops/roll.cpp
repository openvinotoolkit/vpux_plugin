//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RollOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RollOpAdaptor roll(operands, attrs, prop);
    if (mlir::failed(roll.verify(loc))) {
        return mlir::failure();
    }

    const bool shiftContentIsSplat = roll.getShift().getDefiningOp<Const::DeclareOp>().getContentAttr().isSplat();
    const auto inShapeShift = getShape(roll.getShift());

    if (!shiftContentIsSplat && inShapeShift.size() == 1) {
        auto shiftData = IE::constInputToData(loc, roll.getShift());
        if (mlir::failed(shiftData)) {
            return mlir::failure();
        }

        auto axesData = IE::constInputToData(loc, roll.getAxes());
        if (mlir::failed(axesData)) {
            return mlir::failure();
        }

        auto shiftShape = shiftData.value();
        auto axesShape = axesData.value();

        if (shiftShape.size() != axesShape.size()) {
            return errorAt(loc,
                           "If shift is a 1D vector, axes must be a 1D tensor of the same size. Got shift size {0} and "
                           "axes size {1}.",
                           shiftShape.size(), axesShape.size());
        }
    }

    const auto inType = roll.getData().getType().cast<vpux::NDTypeInterface>();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVN1SumOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVN1SumOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto iType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto iShape = iType.getShape().raw();
    const auto outN = iShape[Dims4D::Act::N.ind()];
    auto outC = op.getAcrossChannels() ? 1 : iShape[Dims4D::Act::C.ind()];
    auto outW = op.getNormalizeVariance() ? 2 : 1;  // {sum, sqSum} or {sum}
    SmallVector<int64_t> oShape{outN, outC, 1, outW};

    // output-precision = f32, irrespective of input-precision
    // output-layout = NHWC
    const auto typeComponents = TypeComponents()
                                        .setShape(Shape(oShape))
                                        .setDimsOrder(DimsOrder::NHWC)
                                        .setElementType(mlir::Float32Type::get(ctx));

    auto oType = iType.changeTypeComponents(typeComponents);
    inferredReturnTypes.push_back(oType);

    return mlir::success();
}

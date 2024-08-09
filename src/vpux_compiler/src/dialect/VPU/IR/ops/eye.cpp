//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::EyeOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties prop, mlir::RegionRange,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::EyeOpAdaptor eye(operands, attrs, prop);
    if (mlir::failed(eye.verify(loc))) {
        return mlir::failure();
    }

    const auto numRowsVal = eye.getNumRowsValueAttr().getValue().getSExtValue();
    const auto numColumnsVal = eye.getNumColumnsValueAttr().getValue().getSExtValue();
    const auto batchShapeVal = parseIntArrayAttr<int64_t>(eye.getBatchShapeValueAttr());

    SmallVector<int64_t> outShape = {numRowsVal, numColumnsVal};
    if (batchShapeVal[0] != 0) {
        for (size_t i = 0; i < batchShapeVal.size(); i++) {
            outShape.insert(outShape.begin() + i, batchShapeVal[i]);
        }
    }

    const auto inType = eye.getDiagonalIndex().getType().cast<NDTypeInterface>();
    const auto outType = mlir::RankedTensorType::get(outShape, eye.getOutputType(), createTensorAttrFromType(inType));
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

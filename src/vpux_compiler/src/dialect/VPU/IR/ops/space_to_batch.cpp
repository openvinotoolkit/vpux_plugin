//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SpaceToBatch::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SpaceToBatchAdaptor spb(operands, attrs, prop);
    if (mlir::failed(spb.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = spb.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape().raw();

    const auto blockShape = parseIntArrayAttr<int64_t>(spb.getBlockShapeValueAttr());
    const auto padsBegin = parseIntArrayAttr<int64_t>(spb.getPadsBeginValueAttr());
    const auto padsEnd = parseIntArrayAttr<int64_t>(spb.getPadsEndValueAttr());

    auto outShape = SmallVector<int64_t>(inputShape.size());

    outShape[0] = inputShape[0] *
                  std::accumulate(blockShape.begin(), blockShape.end(), int64_t(1), std::multiplies<int64_t>());

    for (size_t i = 1; i < inputShape.size(); i++) {
        outShape[i] = (inputShape[i] + padsBegin[i] + padsEnd[i]) / blockShape[i];
    }

    const auto outType = inputType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

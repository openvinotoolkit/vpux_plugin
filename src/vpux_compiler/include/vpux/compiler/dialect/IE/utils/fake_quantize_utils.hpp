//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "mlir/Support/LogicalResult.h"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/utils/core/logger.hpp"

#include <functional>

namespace vpux {
namespace IE {

mlir::LogicalResult broadcastContentAttrs(vpux::Const::ContentAttr& inLowContentAttr,
                                          vpux::Const::ContentAttr& inHighContentAttr,
                                          vpux::Const::ContentAttr& transformContentAttr, const Logger& log);

mlir::FailureOr<std::tuple<vpux::Const::ContentAttr, vpux::Const::ContentAttr, mlir::RankedTensorType>>
applyTransformation(vpux::Const::ContentAttr inLowContentAttr, vpux::Const::ContentAttr inHighContentAttr,
                    vpux::Const::ContentAttr transformContentAttr,
                    const std::function<float(float, float)>& transformCb, const Logger& log);

mlir::LogicalResult applyScaleShift(const Const::ContentAttr& scale, const Const::ContentAttr& shift,
                                    Const::ContentAttr& low, Const::ContentAttr& high,
                                    vpux::NDTypeInterface& storageType, const Logger& log);

mlir::LogicalResult revertScaleShift(const Const::ContentAttr& scale, const Const::ContentAttr& shift,
                                     Const::ContentAttr& low, Const::ContentAttr& high,
                                     vpux::NDTypeInterface& storageType, const Logger& log);

mlir::FailureOr<std::tuple<int64_t, bool>> getLevels(Const::ContentAttr weightsContentAttr, float weightsMinimum);

mlir::FailureOr<std::tuple<mlir::Operation*, Const::ContentAttr, Const::ContentAttr>> getWeightsDequantizeStructure(
        Const::DeclareOp origOp, const Logger& _log);

mlir::FailureOr<Const::ContentAttr> castWeightStorageToHighPrecision(const Const::Content& weightsContent,
                                                                     const Logger& _log);

template <typename StorageT>
float getMinWeightsValue(const Const::Content& weightsContent) {
    const auto& values = weightsContent.getStorageBuf<StorageT>();
    const auto min = std::min_element(values.begin(), values.end());

    VPUX_THROW_WHEN(min == values.end(), "Got empty weights content");
    return static_cast<float>(*min);
}

}  // namespace IE
}  // namespace vpux

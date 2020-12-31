//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <numeric>

using namespace vpux;

mlir::LogicalResult vpux::IE::SqueezeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SqueezeOpAdaptor squeeze(operands, attrs);
    if (mlir::failed(squeeze.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = squeeze.input1().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();

    auto inAxes = squeeze.input2().getDefiningOp<mlir::ConstantOp>();
    if (inAxes == nullptr) {
        return mlir::failure();
    }

    const auto denseElementArray = inAxes.value().dyn_cast<mlir::DenseElementsAttr>();
    if (denseElementArray == nullptr) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShapeVec(inDataShape.begin(), inDataShape.end());

    auto axesVec = to_small_vector(denseElementArray.getValues<int64_t>());
    if (axesVec.empty()) {
        for (auto it = outShapeVec.begin(); it != outShapeVec.end();) {
            if (*it == 1) {
                it = outShapeVec.erase(it);
            } else {
                ++it;
            }
        }
    } else {
        std::sort(axesVec.begin(), axesVec.end(), [](int64_t vl, int64_t vr) {
            return vl > vr;
        });

        for (auto a : axesVec) {
            if (a >= static_cast<int64_t>(outShapeVec.size()) || outShapeVec[a] != 1) {
                return mlir::failure();
            }

            outShapeVec.erase(outShapeVec.begin() + a);
        }
    }

    inferredReturnShapes.emplace_back(makeArrayRef(outShapeVec), inDataType.getElementType());
    return mlir::success();
}

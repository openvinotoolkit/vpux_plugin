//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/layer_post_ops_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace VPU {

bool checkForQuantization(mlir::Operation* op, mlir::Operation* postOp) {
    auto isFakeQuantizeOpInput = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op->getOperand(0).getDefiningOp());
    auto isFakeQuantizeOpOutput = true;
    for (auto user : postOp->getUsers()) {
        if (!mlir::isa<IE::FakeQuantizeOp>(user)) {
            isFakeQuantizeOpOutput = false;
            break;
        }
    }

    // since FusePostOps is called also after LowPrecisionPipeline
    const auto operandType = postOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto isQuantizedElemType = operandType.getElementType().isa<mlir::quant::QuantizedType>();

    return (isFakeQuantizeOpOutput && isFakeQuantizeOpInput) || isQuantizedElemType;
};

bool isSupportedHWClampOp(mlir::Operation* mainOp, mlir::Operation* clampOp, const LogCb& logCb) {
    if (auto clamp = mlir::dyn_cast<IE::ClampOp>(clampOp)) {
        const auto minVal = clamp.getMinAttr().getValueAsDouble();
        const auto isQuantized = vpux::VPU::checkForQuantization(mainOp, clampOp);
        if (!isDoubleEqual(minVal, 0.0) && !isQuantized) {
            logCb(llvm::formatv("Float {0} at `{1}` doesn't support non-zero clamp min", clampOp->getName(),
                                clampOp->getLoc()));
            return false;
        }
        // Disable MaxPool fused with Clamp since it is not fully supported by firmware.
        // Tracking Number: E#145636
        if (mlir::isa<IE::MaxPoolOp>(mainOp)) {
            const auto maxVal = clamp.getMaxAttr().getValueAsDouble();
            const auto maxValueFP16 = checked_cast<double>(std::numeric_limits<vpux::type::float16>::max());
            // Given upper bound as fp16 max value, keep fusing Clamp into MaxPool to pass CI
            // Tracking Number: E#146652
            if ((!isDoubleEqual(maxVal, maxValueFP16))) {
                logCb(llvm::formatv("{0} at `{1}` cannot be fused into MaxPool due to lack of firmware support",
                                    clampOp->getName(), clampOp->getLoc()));
                return false;
            }
        }
        return true;
    }
    logCb(llvm::formatv("{0} at `{1}` is not clamp op", clampOp->getName(), clampOp->getLoc()));
    return false;
}

mlir::DictionaryAttr mergeClampAttrs(mlir::DictionaryAttr currentClampAttr, IE::ClampOp clampOp) {
    SmallVector<mlir::NamedAttribute> newClampAttr;

    auto maxId = mlir::StringAttr::get(clampOp.getContext(), "max");
    auto minId = mlir::StringAttr::get(clampOp.getContext(), "min");

    const auto minClampOp = clampOp.getMinAttr().getValueAsDouble();
    const auto maxClampOp = clampOp.getMaxAttr().getValueAsDouble();

    double currentClampMin = 0, currentClampMax = 0;

    if (currentClampAttr.contains(maxId)) {
        currentClampMax = currentClampAttr.get(maxId).dyn_cast<mlir::FloatAttr>().getValueAsDouble();
    }

    if (currentClampAttr.contains(minId)) {
        currentClampMin = currentClampAttr.get(minId).dyn_cast<mlir::FloatAttr>().getValueAsDouble();
    }

    const auto newMin = std::max(currentClampMin, minClampOp);
    const auto newMax = std::min(currentClampMax, maxClampOp);
    const auto newMinAttr = getFPAttr(clampOp.getContext(), newMin);
    const auto newMaxAttr = getFPAttr(clampOp.getContext(), newMax);

    newClampAttr.emplace_back(maxId, newMaxAttr);
    newClampAttr.emplace_back(minId, newMinAttr);

    return mlir::DictionaryAttr::get(clampOp.getContext(), newClampAttr);
}

void setHWClampOp(mlir::Operation* mainOp, mlir::Operation* activationOp) {
    auto maybeClampOp = mlir::dyn_cast<IE::ClampOp>(activationOp);
    VPUX_THROW_WHEN(maybeClampOp == nullptr, "Not ClampOp provided at {0}", activationOp->getLoc());

    auto hasClampAttr = mainOp->hasAttr("clamp");
    mlir::DictionaryAttr clampOpInfo;

    if (hasClampAttr) {
        auto mainClampAttr = mainOp->getAttr("clamp").dyn_cast<mlir::DictionaryAttr>();
        VPUX_THROW_UNLESS(mainClampAttr, "The clamp attribute is expected to be a DictionaryAttr at {0}",
                          mainOp->getLoc());
        clampOpInfo = mergeClampAttrs(mainClampAttr, maybeClampOp);
    } else {
        clampOpInfo = activationOp->getAttrDictionary();
    }
    mainOp->setAttr("clamp", clampOpInfo);
}

}  // namespace VPU
}  // namespace vpux

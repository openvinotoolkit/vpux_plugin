//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_block_rewriters.hpp"

namespace vpux {
namespace VPUIPDPU {

IOType getIOType(mlir::Type type) {
    auto baseType = DPUInvariantBlockRewriter::getBaseType(type);
    if (baseType.isa<mlir::IntegerType>()) {
        return IOType::INT;
    } else if (baseType.isa<mlir::FloatType>()) {
        return IOType::FP;
    }

    return IOType::IOTypeNum;
}

std::pair<mlir::LogicalResult, PPETask> evalPPETasks(const Logger& log, mlir::Region& ppeRegion) {
    PPETask ppeTask{};

    for (auto ppeTaskOp : ppeRegion.getOps<VPUASM::PPETaskOp>()) {
        const auto ppeMode = ppeTaskOp.getPpeLayerType();
        if (ppeMode != VPU::PPEMode::NOOP) {
            if (ppeTask.fixedFunction.ppeMode != VPU::PPEMode::NOOP) {
                log.error("Cannot set more than one PPE task");
                return {mlir::failure(), ppeTask};
            }
            ppeTask.fixedFunction.ppeMode = ppeMode;
        }
        if (ppeTaskOp.getClampLow().has_value()) {
            if (auto intClampLowAttr = ppeTaskOp.getClampLowAttr().dyn_cast<mlir::IntegerAttr>()) {
                ppeTask.fixedFunction.intClampLow = checked_cast<int32_t>(intClampLowAttr.getInt());
            } else if (auto fpClampLowAttr = ppeTaskOp.getClampLowAttr().dyn_cast<mlir::FloatAttr>()) {
                ppeTask.fixedFunction.fpClampLow = static_cast<float>(fpClampLowAttr.getValue().convertToDouble());
            }
        }
        if (ppeTaskOp.getClampHigh().has_value()) {
            if (auto intClampHighAttr = ppeTaskOp.getClampHighAttr().dyn_cast<mlir::IntegerAttr>()) {
                ppeTask.fixedFunction.intClampHigh = checked_cast<int32_t>(intClampHighAttr.getInt());
            } else if (auto fpClampHighAttr = ppeTaskOp.getClampHighAttr().dyn_cast<mlir::FloatAttr>()) {
                ppeTask.fixedFunction.fpClampHigh = static_cast<float>(fpClampHighAttr.getValue().convertToDouble());
            }
        }
        if (ppeTaskOp.getLreluMult().has_value()) {
            ppeTask.fixedFunction.lReluMult = checked_cast<int32_t>(ppeTaskOp.getLreluMult().value());
        }
        if (ppeTaskOp.getLreluShift().has_value()) {
            ppeTask.fixedFunction.lReluShift = checked_cast<uint32_t>(ppeTaskOp.getLreluShift().value());
        }
        if (ppeTaskOp.getQuantScale().has_value()) {
            auto floatScaleAttr = ppeTaskOp.getQuantScaleAttr().getValue()[0];
            // for float values checked cast will fail due to floating point precision representation differences
            // between float and int. Intentionally using static_cast
            ppeTask.fpScaleData =
                    static_cast<float>(floatScaleAttr.cast<mlir::FloatAttr>().getValue().convertToDouble());
        }
        if (ppeTaskOp.getFpPreluAlpha().has_value()) {
            ppeTask.fpPreluAlpha = static_cast<float>(ppeTaskOp.getFpPreluAlpha().value().convertToDouble());
        }
        if (ppeTaskOp.getQuantMult().has_value()) {
            ppeTask.ppeQuantMult =
                    parseIntArrayAttr<int64_t>(ppeTaskOp.getQuantMult().value().dyn_cast<mlir::ArrayAttr>());
        }
        if (ppeTaskOp.getQuantShift().has_value()) {
            ppeTask.ppeQuantShift = parseIntArrayAttr<int64_t>(ppeTaskOp.getQuantShift().value());
        }
        if (ppeTaskOp.getQuantPostShift().has_value()) {
            ppeTask.ppeQuantPostShift = checked_cast<int64_t>(ppeTaskOp.getQuantPostShift().value());
        }
        if (ppeTaskOp.getPpeFpScale().has_value()) {
            const auto fpScale = static_cast<float>(ppeTaskOp.getPpeFpScale().value().convertToDouble());
            ppeTask.ppeFpScale = fpScale;
        }
        if (ppeTaskOp.getPpeFpBias().has_value()) {
            const auto fpBias = static_cast<float>(ppeTaskOp.getPpeFpBias().value().convertToDouble());
            ppeTask.ppeFpBias = fpBias;
        }
    }

    if ((ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LRELU) ||
        (ppeTask.fixedFunction.ppeMode == VPU::PPEMode::LRELUX)) {
        ppeTask.fpPreluAlpha = -0.0f;  // note: -0.0, to ensure zero-gained data uses positive zero in FP32
                                       // (0x00000000), not negative zero (0x80000000)
    }

    return {mlir::success(), ppeTask};
}

mlir::FloatAttr getF32FloatAttrOrNull(mlir::OpBuilder& builder, const std::optional<float>& attr) {
    if (attr.has_value()) {
        return builder.getF32FloatAttr(attr.value());
    }

    return nullptr;
}

mlir::MemRefType getBufferType(mlir::Operation* bufferRef) {
    mlir::MemRefType bufferType;

    if (mlir::isa<VPUASM::DeclareBufferOp>(bufferRef)) {
        auto buffer = mlir::cast<VPUASM::DeclareBufferOp>(bufferRef);
        bufferType = buffer.getBufferType().getMemref();
    } else if (mlir::isa<VPUASM::ConstBufferOp>(bufferRef)) {
        auto buffer = mlir::cast<VPUASM::ConstBufferOp>(bufferRef);
        bufferType = buffer.getBufferType().getMemref();
    } else {
        VPUX_THROW("Not a buffer: {0}", bufferRef);
    }

    return bufferType;
}

uint64_t getSwizzlingKey(mlir::Operation* bufferRef) {
    uint64_t swizzlingKey = 0;

    if (mlir::isa<VPUASM::DeclareBufferOp>(bufferRef)) {
        auto buffer = mlir::cast<VPUASM::DeclareBufferOp>(bufferRef);
        swizzlingKey = buffer.getBufferType().getTraits().getSwizzlingKey();
    } else if (mlir::isa<VPUASM::ConstBufferOp>(bufferRef)) {
        auto buffer = mlir::cast<VPUASM::ConstBufferOp>(bufferRef);
        swizzlingKey = buffer.getBufferType().getTraits().getSwizzlingKey();
    } else {
        VPUX_THROW("Not a buffer: {0}", bufferRef);
    }

    return swizzlingKey;
}

}  // namespace VPUIPDPU
}  // namespace vpux

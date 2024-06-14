//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

namespace vpux {
namespace VPUIPDPU {

enum class IOType { INT, FP, IOTypeNum };

struct PPETask {
    struct PPEFixedFunction {
        VPU::PPEMode ppeMode = VPU::PPEMode::NOOP;
        int32_t intClampLow = std::numeric_limits<int32_t>::min();
        int32_t intClampHigh = std::numeric_limits<int32_t>::max();
        float fpClampLow = std::numeric_limits<float>::lowest();
        float fpClampHigh = std::numeric_limits<float>::max();
        int32_t lReluMult = 1;
        uint32_t lReluShift = 0;
    } fixedFunction;
    VPUIPDPU::PPEIntRoundMode rounding = VPUIPDPU::PPEIntRoundMode::RNE;
    float fpScaleData = 1.0f;
    float fpPreluAlpha = 1.0f;
    std::optional<SmallVector<int64_t>> ppeQuantMult;
    std::optional<SmallVector<int64_t>> ppeQuantShift;
    std::optional<int64_t> ppeQuantPostShift;
    std::optional<float> ppeFpScale = std::nullopt;
    std::optional<float> ppeFpBias = std::nullopt;
};

IOType getIOType(mlir::Type type);

std::pair<mlir::LogicalResult, PPETask> evalPPETasks(const Logger& log, mlir::Region& ppeRegion);

template <typename AttrType, typename ValueType>
AttrType getEnumAttrOrNull(mlir::OpBuilder& builder, const std::optional<ValueType>& attr) {
    if (attr.has_value()) {
        return AttrType::get(builder.getContext(), attr.value());
    }

    return nullptr;
}
mlir::FloatAttr getF32FloatAttrOrNull(mlir::OpBuilder& builder, const std::optional<float>& attr);

mlir::MemRefType getBufferType(mlir::Operation* bufferRef);

uint64_t getSwizzlingKey(mlir::Operation* bufferRef);

}  // namespace VPUIPDPU
}  // namespace vpux

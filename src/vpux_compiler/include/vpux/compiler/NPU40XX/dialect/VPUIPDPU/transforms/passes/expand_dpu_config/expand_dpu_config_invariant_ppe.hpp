//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIPDPU::arch40xx::PPE {

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
    PPEIntRoundMode rounding = PPEIntRoundMode::RNE;
    float fpScaleData = 1.0f;
    float fpPreluAlpha = 1.0f;
    std::optional<SmallVector<int64_t>> ppeQuantMult;
    std::optional<SmallVector<int64_t>> ppeQuantShift;
    std::optional<int64_t> ppeQuantPostShift;
    std::optional<float> ppeFpScale = {};
    std::optional<float> ppeFpBias = {};
};

mlir::FailureOr<PPETask> evalPPETasks(const Logger& log, mlir::Region& ppeRegion);

}  // namespace vpux::VPUIPDPU::arch40xx::PPE

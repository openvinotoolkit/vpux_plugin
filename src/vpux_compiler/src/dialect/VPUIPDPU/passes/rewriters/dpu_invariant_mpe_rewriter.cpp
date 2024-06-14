//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_block_rewriters.hpp"

namespace vpux {
namespace VPUIPDPU {

// Helper function to calculate zero point offset for input activations and weights
uint8_t getZeroPointBias(mlir::Value operand) {
    // Get also ZP
    SmallVector<uint8_t> quantZeroPoints;

    auto type = operand.getType().cast<vpux::NDTypeInterface>();

    auto elementType = type.getElementType();
    if (const auto uniformQuantType = elementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        quantZeroPoints.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType = elementType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto zp = uniformQuantPerAxisType.getZeroPoints();
        quantZeroPoints.resize(zp.size());
        std::transform(zp.begin(), zp.end(), quantZeroPoints.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        quantZeroPoints.push_back(0);
    }

    // Return only the first element as the zero point bias
    return quantZeroPoints[0];
}

DPUInvariantMPERewriter::DPUInvariantMPERewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                                                 std::map<BlockArg, size_t>& invBlockArgsPos,
                                                 mlir::PatternRewriter& rewriter, const Logger& log)
        : DPUInvariantBlockRewriter(origInvOp, invBlock, invBlockArgsPos, rewriter, log) {
}

mlir::LogicalResult DPUInvariantMPERewriter::rewrite() {
    if (insertEntryBlock<VPUIPDPU::MPECfgOp>().failed()) {
        return mlir::failure();
    }

    auto inAct = getInvBlockArg(DPUInvariantBlockRewriter::BlockArg::ACT_IN);
    auto weights = getInvBlockArg(DPUInvariantBlockRewriter::BlockArg::WEIGHTS);

    if (inAct) {
        auto inActType =
                DPUInvariantBlockRewriter::getBaseType(inAct.getType().cast<mlir::MemRefType>().getElementType());
        if (inActType.isInteger(CHAR_BIT)) {
            _rewriter.create<VPUIPDPU::MPEActivationBiasOp>(_origInvOp.getLoc(), getZeroPointBias(inAct));
        }
    }

    if (weights) {
        auto wtType =
                DPUInvariantBlockRewriter::getBaseType(weights.getType().cast<mlir::MemRefType>().getElementType());

        if (wtType.isInteger(CHAR_BIT)) {
            _rewriter.create<VPUIPDPU::MPEWeightsBiasOp>(_origInvOp.getLoc(), getZeroPointBias(weights));
        }
    }

    // mpe_daz not set/used in graph_file nce_lib so then
    // MPEDenormalOperandsFTZOp will not be instantiated here.

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux

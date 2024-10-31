//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux {
namespace IE {

using SupportedMixedPrecisionFunctor = std::function<bool(mlir::Operation*, const bool isPReLUSupported, Logger log)>;

//
// QuantizeWithTwoInputsNCEEltwiseOpGeneric
//

//
// Case 1:
//      [input 1]       [input 2]
//          |               |
//      (quantize)      (quantize)
//          |               |
//         u8 -(EltwiseOp)- u8
//
// Case 2:
//             [input 1]
//                 |
//             (quantize)
//          |               |
//         u8 -(EltwiseOp)- u8
//
// Case 3:
//      [input 1]      [u8_Conv_u8]
//          |               |
//      (quantize)
//          |               |
//         u8 -(EltwiseOp)- u8
//

template <typename ConcreteOp>
class QuantizeWithTwoInputsNCEEltwiseOpGeneric final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    QuantizeWithTwoInputsNCEEltwiseOpGeneric(mlir::MLIRContext* ctx,
                                             const SupportedMixedPrecisionFunctor& isMixPrecisionSupported, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _isMixPrecisionSupported(isMixPrecisionSupported), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
    mlir::LogicalResult findQuantizeOrQuantizedNCE(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                                   mlir::Value addInput,
                                                   SmallVector<mlir::Operation*>& addToQuantizeOps) const;
    mlir::LogicalResult removeQuantOrFusedQuant(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                                ArrayRef<mlir::Operation*> addToQuantizeOps,
                                                mlir::Operation* quantOrQuantizedNCE, mlir::Type elementType) const;
};
}  // namespace IE
}  // namespace vpux

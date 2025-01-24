//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux {
namespace IE {

using SupportedMixedPrecisionFunctor = std::function<bool(mlir::Operation*, const bool isPReLUSupported, Logger log)>;

template <typename ConcreteOp>
mlir::LogicalResult findQuantizeOrQuantizedNCE(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                               mlir::Value eltwiseInput,
                                               SmallVector<mlir::Operation*>& eltwiseToQuantizeOps, Logger log);
template <typename ConcreteOp>
mlir::LogicalResult removeQuantOrFusedQuant(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                            ArrayRef<mlir::Operation*> eltwiseToQuantizeOps,
                                            mlir::Operation* quantOrQuantizedNCE, mlir::Type elementType,
                                            const SupportedMixedPrecisionFunctor& isMixPrecisionSupported, Logger log);

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
};

class QuantizeWithAvgPool final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    QuantizeWithAvgPool(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                        Logger log)
            : mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _isMixPrecisionSupported(isMixPrecisionSupported), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp avgPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
};
}  // namespace IE
}  // namespace vpux

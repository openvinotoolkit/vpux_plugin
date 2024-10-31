//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_outstanding_quant.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;
using namespace IE;

template <typename ConcreteOp>
mlir::LogicalResult QuantizeWithTwoInputsNCEEltwiseOpGeneric<ConcreteOp>::matchAndRewrite(
        ConcreteOp origOp, mlir::PatternRewriter& rewriter) const {
    static_assert(ConcreteOp::template hasTrait<IE::EltwiseOp>(), "Expected operation to be EltwiseOp");
    VPUX_THROW_UNLESS(origOp.getNumOperands() == 2, "Expected operation to take two operands");

    auto isQuantizedInput = [](mlir::TypedValue<mlir::RankedTensorType> value) {
        return value.getType().getElementType().isa<mlir::quant::QuantizedType>();
    };
    const auto noQuantizedInput = !isQuantizedInput(origOp.getInput1()) && !isQuantizedInput(origOp.getInput2());
    if (noQuantizedInput) {
        return matchFailed(rewriter, origOp, "OrigOp doesn't have quantized input");
    }
    if (!_isMixPrecisionSupported(origOp, false, _log)) {
        return matchFailed(rewriter, origOp, "OrigOp doesn't support mixed precision");
    }

    SmallVector<mlir::Operation*> lhsAddToQuantizeOps, rhsAddToQuantizeOps;
    // Walk through FakeQuantize-agnostic ops and find quantize or quantized NCE task
    if (auto result = findQuantizeOrQuantizedNCE(origOp, rewriter, origOp.getInput1(), lhsAddToQuantizeOps);
        result.failed()) {
        return result;
    }
    if (auto result = findQuantizeOrQuantizedNCE(origOp, rewriter, origOp.getInput2(), rhsAddToQuantizeOps);
        result.failed()) {
        return result;
    }
    _log.trace("[{0}] Pop {1} out of lhsAddToQuantizeOps", this->getDebugName(), lhsAddToQuantizeOps.back()->getName());
    mlir::Operation* lhsQuant = lhsAddToQuantizeOps.pop_back_val();
    _log.trace("[{0}] Pop {1} out of rhsAddToQuantizeOps", this->getDebugName(), rhsAddToQuantizeOps.back()->getName());
    mlir::Operation* rhsQuant = rhsAddToQuantizeOps.pop_back_val();

    if (mlir::isa<IE::LayerWithPostOpInterface>(lhsQuant) && mlir::isa<IE::LayerWithPostOpInterface>(rhsQuant)) {
        return matchFailed(rewriter, origOp, "Quantizes for both ancestors have been fused at {0} ({1}) and {2} ({3})",
                           lhsQuant->getName(), lhsQuant->getLoc(), rhsQuant->getName(), rhsQuant->getLoc());
    }
    // At this point at least one of lhsQuant and rhsQunt is QuantizeOp

    auto getInputElementType = [](mlir::Operation* operation) {
        return operation->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    };
    const mlir::Type elementType =
            mlir::isa<IE::QuantizeOp>(lhsQuant) ? getInputElementType(lhsQuant) : getInputElementType(rhsQuant);

    // Detect if lhs and rhs use the same quantize to avoid double delete
    if (lhsQuant == rhsQuant) {
        rhsQuant = nullptr;
    }

    // Remove the quantize or fused-quantize, and update the type alone the way
    if (auto result = removeQuantOrFusedQuant(origOp, rewriter, lhsAddToQuantizeOps, lhsQuant, elementType);
        result.failed()) {
        return result;
    }
    if (auto result = removeQuantOrFusedQuant(origOp, rewriter, rhsAddToQuantizeOps, rhsQuant, elementType);
        result.failed()) {
        return result;
    }

    return mlir::success();
}

template <typename ConcreteOp>
mlir::LogicalResult QuantizeWithTwoInputsNCEEltwiseOpGeneric<ConcreteOp>::findQuantizeOrQuantizedNCE(
        ConcreteOp origOp, mlir::PatternRewriter& rewriter, mlir::Value addInput,
        SmallVector<mlir::Operation*>& addToQuantizeOps) const {
    auto verifyInput = [origOp, &rewriter](mlir::Operation* operation) {
        if (operation == nullptr) {
            return matchFailed(rewriter, origOp, "Producer is a block argument for {0} at {1}", origOp->getName(),
                               origOp->getLoc());
        }
        if (!mlir::isa<IE::ElemTypeInfoOpInterface, IE::LayerWithPostOpInterface, IE::QuantizeOp>(operation)) {
            return matchFailed(rewriter, origOp,
                               "Ancestor {0} at {1} is neither FakeQuantize-agnostic, NCE, nor Quantize operation",
                               operation->getName(), operation->getLoc());
        }
        VPUX_THROW_UNLESS(!operation->use_empty(),
                          "Expact operation always to have uses because the loop just came from one of it's uses; "
                          "the loop is walking up the calling chain");
        if (!operation->hasOneUse()) {
            // TODO: Should use hasOneUser when it's available in MLIR:
            // https://llvm.org/doxygen/classllvm_1_1Value.html#a2e987c6af902aad6baa39bd5b7ef322c
            const auto hasOneUser =
                    std::equal(++operation->user_begin(), operation->user_end(), operation->user_begin());
            if (!hasOneUser) {
                return matchFailed(rewriter, origOp, "Ancestor has more than one consumer for {0} at {1} ",
                                   operation->getName(), operation->getLoc());
            }
        }
        if (mlir::isa<IE::ElemTypeInfoOpInterface>(operation) && operation->getNumOperands() > 1) {
            return matchFailed(rewriter, origOp,
                               "ElemTypeInfoOpInterface Ancestor has more than one input for {0} at {1}",
                               operation->getName(), operation->getLoc());
        }
        return mlir::success();
    };

    do {
        mlir::Operation* input = [addInput, addToQuantizeOps = ArrayRef(addToQuantizeOps)]() {
            if (addToQuantizeOps.empty()) {
                return addInput.getDefiningOp();
            } else {
                return addToQuantizeOps.back()->getOperand(0).getDefiningOp();
            }
        }();

        if (auto result = verifyInput(input); result.failed()) {
            return result;
        }

        _log.trace("[{0}] Push Op {1} at {2}", this->getDebugName(), input->getName(), input->getLoc());
        addToQuantizeOps.push_back(input);
    } while (!mlir::isa<IE::LayerWithPostOpInterface, IE::QuantizeOp>(addToQuantizeOps.back()));

    auto lastOp = addToQuantizeOps.back();

    const bool verifyLastOp = mlir::isa<IE::LayerWithPostOpInterface, IE::QuantizeOp>(lastOp);
    VPUX_THROW_UNLESS(verifyLastOp, "Expected lastOp to be NCE task or QuantizeOp");

    const std::string operationType = mlir::isa<IE::QuantizeOp>(lastOp) ? "Quantize" : "NCE task";
    _log.trace("[{0}] Found {1} {2} at {3}, stop pattern searching", this->getDebugName(), operationType,
               lastOp->getName(), lastOp->getLoc());

    return mlir::success();
}

template <typename ConcreteOp>
mlir::LogicalResult QuantizeWithTwoInputsNCEEltwiseOpGeneric<ConcreteOp>::removeQuantOrFusedQuant(
        ConcreteOp origOp, mlir::PatternRewriter& rewriter, ArrayRef<mlir::Operation*> addToQuantizeOps,
        mlir::Operation* quantOrQuantizedNCE, mlir::Type elementType) const {
    if (mlir::isa_and_nonnull<IE::LayerWithPostOpInterface>(quantOrQuantizedNCE)) {
        const auto isPerChannel = quantOrQuantizedNCE->getResult(0)
                                          .getType()
                                          .cast<vpux::NDTypeInterface>()
                                          .getElementType()
                                          .isa<mlir::quant::UniformQuantizedPerAxisType>();
        if (!_isMixPrecisionSupported(quantOrQuantizedNCE, !isPerChannel, _log)) {
            return matchFailed(rewriter, origOp, "Producer {0} is not supported", quantOrQuantizedNCE->getName());
        }

        auto* newNCETask = rewriter.clone(*quantOrQuantizedNCE);
        vpux::NDTypeInterface newType = newNCETask->getResult(0).getType();
        newType = newType.changeElemType(elementType);
        newNCETask->getResult(0).setType(newType);
        newNCETask->moveBefore(quantOrQuantizedNCE);

        _log.trace("[{0}] Replace {1} {2} at {3} with {4} {5} at {6}", this->getDebugName(),
                   quantOrQuantizedNCE->getName(), quantOrQuantizedNCE->getResult(0).getType(),
                   quantOrQuantizedNCE->getLoc(), newNCETask->getName(), newNCETask->getResult(0).getType(),
                   newNCETask->getLoc());

        rewriter.replaceOp(quantOrQuantizedNCE, newNCETask->getResult(0));
    } else if (mlir::isa_and_nonnull<IE::QuantizeOp>(quantOrQuantizedNCE)) {
        _log.trace("[{0}] Remove {1} at {2}", this->getDebugName(), quantOrQuantizedNCE->getName(),
                   quantOrQuantizedNCE->getLoc());
        rewriter.replaceOp(quantOrQuantizedNCE, quantOrQuantizedNCE->getOperand(0));
    } else {
        _log.trace("[{0}] Quantize is already erased on lhs", this->getDebugName());
    }

    for (auto iterator = addToQuantizeOps.rbegin(); iterator != addToQuantizeOps.rend(); ++iterator) {
        _log.trace("[{0}] Change {1} at {2} to {3}", this->getDebugName(), (*iterator)->getName(),
                   (*iterator)->getLoc(), (*iterator)->getResult(0).getType());
        inferReturnTypes(*iterator, InferShapedTypeMode::ELEM_TYPE);
    }

    return mlir::success();
}

template class IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::AddOp>;
template class IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::MultiplyOp>;
template class IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::SubtractOp>;

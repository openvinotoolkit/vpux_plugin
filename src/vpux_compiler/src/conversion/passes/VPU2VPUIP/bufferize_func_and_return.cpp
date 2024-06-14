//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

//
// One-shot-bufferization based funcOp and ReturnOp bufferization
//

namespace {

//
// getFuncOneShotAnalysisState
//

const mlir::bufferization::func_ext::FuncAnalysisState& getFuncOneShotAnalysisState(
        const mlir::bufferization::AnalysisState& state) {
    VPUX_THROW_WHEN(!mlir::isa<mlir::bufferization::OneShotAnalysisState>(state), "Expected OneShotAnalysisState");

    auto* result = static_cast<const mlir::bufferization::OneShotAnalysisState&>(state)
                           .getExtension<mlir::bufferization::func_ext::FuncAnalysisState>();
    VPUX_THROW_WHEN(result == nullptr, "FuncAnalysisState does not exist");

    return *result;
}

//
// getFuncOpAnalysisState
//

mlir::bufferization::func_ext::FuncOpAnalysisState getFuncOpAnalysisState(
        const mlir::bufferization::AnalysisState& state, mlir::func::FuncOp funcOp) {
    if (!mlir::isa<mlir::bufferization::OneShotAnalysisState>(state)) {
        return mlir::bufferization::func_ext::FuncOpAnalysisState::NotAnalyzed;
    }
    auto* funcState = static_cast<const mlir::bufferization::OneShotAnalysisState&>(state)
                              .getExtension<mlir::bufferization::func_ext::FuncAnalysisState>();
    if (!funcState) {
        return mlir::bufferization::func_ext::FuncOpAnalysisState::NotAnalyzed;
    }
    const auto& analyzedFuncOps = funcState->analyzedFuncOps;
    auto it = analyzedFuncOps.find(funcOp);
    if (it == analyzedFuncOps.end()) {
        return mlir::bufferization::func_ext::FuncOpAnalysisState::NotAnalyzed;
    }
    return it->second;
}

//
//  getEquivalentFuncArgIdx
//

std::optional<int64_t> getEquivalentFuncArgIdx(mlir::func::FuncOp funcOp,
                                               const mlir::bufferization::func_ext::FuncAnalysisState& state,
                                               int64_t returnValIdx) {
    auto funcOpIt = state.equivalentFuncArgs.find(funcOp);
    if (funcOpIt == state.equivalentFuncArgs.end()) {
        // No equivalence info stores for funcOp.
        return std::nullopt;
    }

    auto retValIt = funcOpIt->getSecond().find(returnValIdx);
    if (retValIt == funcOpIt->getSecond().end()) {
        // Return value has no equivalent bbArg.
        return std::nullopt;
    }

    return retValIt->getSecond();
}

//
// CallOpBufferizeModel
//

class CallOpBufferizeModel : public BufferizableOpInterfaceExternalModelBase<CallOpBufferizeModel, mlir::func::CallOp> {
public:
    bool bufferizesToMemoryReadImpl(mlir::func::CallOp op, mlir::OpOperand& opOperand,
                                    const mlir::bufferization::AnalysisState& state) const;
    bool bufferizesToMemoryWriteImpl(mlir::func::CallOp op, mlir::OpOperand& opOperand,
                                     const mlir::bufferization::AnalysisState& state) const;
    mlir::bufferization::AliasingOpResultList getAliasingOpResultsImpl(
            mlir::func::CallOp op, mlir::OpOperand& opOperand, const mlir::bufferization::AnalysisState& state) const;
    mlir::LogicalResult bufferizeImpl(mlir::func::CallOp op, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      mlir::func::CallOp::Adaptor adaptor) const;
};

bool CallOpBufferizeModel::bufferizesToMemoryReadImpl(mlir::func::CallOp op, mlir::OpOperand& opOperand,
                                                      const mlir::bufferization::AnalysisState& state) const {
    auto callOp = mlir::cast<mlir::func::CallOp>(op);
    auto funcOp = getCalledFunction(callOp);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        return true;
    }

    const auto& funcState = getFuncOneShotAnalysisState(state);
    return funcState.readBbArgs.lookup(funcOp).contains(opOperand.getOperandNumber());
}

bool CallOpBufferizeModel::bufferizesToMemoryWriteImpl(mlir::func::CallOp op, mlir::OpOperand& opOperand,
                                                       const mlir::bufferization::AnalysisState& state) const {
    auto callOp = mlir::cast<mlir::func::CallOp>(op);
    auto funcOp = getCalledFunction(callOp);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        // FuncOp not analyzed yet. Assume that OpOperand is written.
        return true;
    }

    const auto& funcState = getFuncOneShotAnalysisState(state);
    return funcState.writtenBbArgs.lookup(funcOp).contains(opOperand.getOperandNumber());
}

mlir::bufferization::AliasingOpResultList CallOpBufferizeModel::getAliasingOpResultsImpl(
        mlir::func::CallOp op, mlir::OpOperand& opOperand, const mlir::bufferization::AnalysisState& state) const {
    auto callOp = mlir::cast<mlir::func::CallOp>(op);
    auto funcOp = getCalledFunction(callOp);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        // FuncOp not analyzed yet. Any OpResult may be aliasing.
        return mlir::bufferization::detail::unknownGetAliasingOpResults(opOperand);  // Note: using 'detail' namespace!
    }

    // Get aliasing results from state.
    const auto& funcState = getFuncOneShotAnalysisState(state);
    auto aliasingReturnVals = funcState.aliasingReturnVals.lookup(funcOp).lookup(opOperand.getOperandNumber());

    // Check if the aliasing OpResult is equivalent to the OpOperand.
    std::optional<int64_t> equivalent = {};
    if (aliasingReturnVals.size() == 1) {
        equivalent = getEquivalentFuncArgIdx(funcOp, funcState, aliasingReturnVals.front());
        VPUX_THROW_WHEN((equivalent.has_value() && *equivalent != opOperand.getOperandNumber()),
                        "inconsistent analysis state");
    }

    mlir::bufferization::AliasingOpResultList result;
    for (auto resultIdx : aliasingReturnVals) {
        result.addAlias({callOp->getOpResult(resultIdx),
                         equivalent.has_value() ? mlir::bufferization::BufferRelation::Equivalent
                                                : mlir::bufferization::BufferRelation::Unknown,
                         /*isDefinite=*/equivalent.has_value()});
    }
    return result;
}

mlir::LogicalResult CallOpBufferizeModel::bufferizeImpl(mlir::func::CallOp op, mlir::RewriterBase& rewriter,
                                                        const mlir::bufferization::BufferizationOptions&,
                                                        mlir::func::CallOp::Adaptor) const {
    auto log = Logger::global().nest("one-shot-bufferize-CallOp", 0);
    log.trace("Got '{0}' at '{1}'", op->getName(), op->getLoc());

    auto callOp = mlir::cast<mlir::func::CallOp>(op);

    // 1. Compute the result types of the new CallOp.
    SmallVector<mlir::Type> resultTypes;
    for (auto result : callOp.getResults()) {
        auto returnType = result.getType();
        if (!mlir::isa<mlir::TensorType>(returnType)) {
            // Non-tensor values are returned.
            resultTypes.push_back(returnType);
            continue;
        }
        auto resultType = vpux::getBufferType(result);
        resultTypes.push_back(resultType);
    }

    // 2. Rewrite tensor operands as memrefs based on type of the already bufferized callee.
    auto newOperands = vpux::bufferizeOperands(rewriter, callOp->getOperands());

    // 3. Create the new CallOp.
    auto funcOp = getCalledFunction(callOp);
    auto newCallOp =
            rewriter.create<mlir::func::CallOp>(callOp.getLoc(), funcOp.getSymName(), resultTypes, newOperands);
    newCallOp->setAttrs(callOp->getAttrs());

    // 4. Replace the old op with the new op.
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, callOp, newCallOp->getResults());

    return mlir::success();
}

}  // namespace

//
// ReturnOpBufferizeModel
//

namespace {

class ReturnOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<ReturnOpBufferizeModel, mlir::func::ReturnOp> {
public:
    mlir::LogicalResult bufferizeImpl(mlir::func::ReturnOp, mlir::RewriterBase&,
                                      const mlir::bufferization::BufferizationOptions&,
                                      mlir::func::ReturnOp::Adaptor) const {
        return mlir::success();
    }
};

}  // namespace

namespace {

//
// getAssumedUniqueReturnOp
//

mlir::func::ReturnOp getAssumedUniqueReturnOp(mlir::func::FuncOp funcOp) {
    mlir::func::ReturnOp returnOp;
    for (auto& b : funcOp.getBody()) {
        if (auto candidateOp = mlir::dyn_cast<mlir::func::ReturnOp>(b.getTerminator())) {
            if (returnOp) {
                return nullptr;
            }
            returnOp = candidateOp;
        }
    }
    return returnOp;
}

//
// FuncOpBufferizeModel
//

class FuncOpBufferizeModel : public BufferizableOpInterfaceExternalModelBase<FuncOpBufferizeModel, mlir::func::FuncOp> {
public:
    bool isWritable(mlir::Operation*, mlir::Value, const mlir::bufferization::AnalysisState&) const {
        return true;
    }

    mlir::LogicalResult bufferizeImpl(mlir::func::FuncOp op, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      mlir::func::FuncOp::Adaptor) const;
};

mlir::LogicalResult FuncOpBufferizeModel::bufferizeImpl(mlir::func::FuncOp funcOp, mlir::RewriterBase& rewriter,
                                                        const mlir::bufferization::BufferizationOptions&,
                                                        mlir::func::FuncOp::Adaptor) const {
    auto log = Logger::global().nest("one-shot-bufferize-FuncOp", 0);
    log.trace("Got '{0}' at '{1}'", funcOp->getName(), funcOp->getLoc());

    // Construct the bufferized function type.
    SmallVector<mlir::Type> argTypes;
    auto& frontBlock = funcOp.getBody().front();
    for (auto& bbArg : frontBlock.getArguments()) {
        argTypes.push_back(vpux::getBufferType(bbArg));
    }

    auto returnOp = getAssumedUniqueReturnOp(funcOp);
    VPUX_THROW_WHEN(returnOp == nullptr, "expected func with single return op");
    auto loc = returnOp.getLoc();

    // 1. Rewrite the bbArgs. Turn every tensor bbArg into a memref bbArg.
    for (auto& bbArg : frontBlock.getArguments()) {
        auto tensorType = mlir::dyn_cast<mlir::TensorType>(bbArg.getType());
        // Non-tensor types stay the same.
        if (!tensorType) {
            continue;
        }

        // Collect all uses of the bbArg.
        SmallVector<mlir::OpOperand*> bbArgUses;
        for (auto& use : bbArg.getUses()) {
            bbArgUses.push_back(&use);
        }

        // Change the bbArg type to memref.
        auto originType = bbArg.getType();
        auto memrefType = argTypes[bbArg.getArgNumber()];
        bbArg.setType(memrefType);

        // Replace all uses of the original tensor bbArg.
        rewriter.setInsertionPointToStart(&frontBlock);
        if (!bbArgUses.empty()) {
            auto toTensorOp = rewriter.create<mlir::bufferization::ToTensorOp>(funcOp.getLoc(), originType, bbArg);
            for (auto* use : bbArgUses) {
                use->set(toTensorOp);
            }
        }
    }

    // 2. For each result, keep track of which inplace argument it reuses.
    SmallVector<mlir::Value> returnValues;
    for (auto& returnOperand : returnOp->getOpOperands()) {
        auto returnVal = returnOperand.get();
        auto tensorType = mlir::dyn_cast<mlir::TensorType>(returnVal.getType());

        // If not a tensor type just forward it.
        if (!tensorType) {
            returnValues.push_back(returnVal);
            continue;
        }

        rewriter.setInsertionPoint(returnOp);
        auto resultType = vpux::getBufferType(returnVal);
        auto toMemrefVal = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, resultType, returnVal).getResult();
        returnValues.push_back(toMemrefVal);
    }

    // 3. Rewrite the terminator without the in-place bufferizable values.
    returnOp.getOperandsMutable().assign(returnValues);

    // 4. Rewrite the FuncOp type to buffer form.
    funcOp.setType(mlir::FunctionType::get(funcOp->getContext(), argTypes, mlir::ValueRange(returnValues).getTypes()));

    return mlir::success();
}

}  // namespace

//
// registerFuncAndReturnBufferizableOpInterfaces
//

void vpux::registerFuncAndReturnBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::func::FuncDialect*) {
        mlir::func::FuncOp::attachInterface<FuncOpBufferizeModel>(*ctx);
        mlir::func::ReturnOp::attachInterface<ReturnOpBufferizeModel>(*ctx);
        mlir::func::CallOp::attachInterface<CallOpBufferizeModel>(*ctx);
    });
}

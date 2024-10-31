//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {

using FunctionCalls = mlir::DenseMap<mlir::func::FuncOp, mlir::SmallVector<mlir::func::CallOp>>;
using FunctionWithSwKernelCalls =
        mlir::DenseMap<mlir::func::FuncOp, std::tuple<mlir::DenseSet<int64_t> /*shave-read-args*/,
                                                      mlir::DenseSet<int64_t> /*shave-written-args*/,
                                                      mlir::SmallVector<mlir::func::CallOp>>>;

mlir::BlockArgument addInputCopy(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::async::ExecuteOp& userExecOp, mlir::BlockArgument blockArg,
                                 mlir::OpResult newBufferResult) {
    auto userExecOpDependencies = userExecOp.getDependencies();
    auto userExecOpDependenciesVector = to_small_vector(userExecOpDependencies);

    builder.setInsertionPoint(userExecOp);

    // create async::ExecuteOp for the new NNDMA
    auto newExecOp = builder.create<mlir::async::ExecuteOp>(loc, newBufferResult.getType(),
                                                            userExecOpDependenciesVector, std::nullopt, nullptr);
    VPUIP::VPUIPDialect::setExecutor(newExecOp,
                                     vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::ExecutorKind::DMA_NN)));

    auto bodyBlock = newExecOp.getBody();
    builder.setInsertionPointToStart(bodyBlock);
    auto nndmaOp = builder.create<VPUIP::NNDMAOp>(loc, blockArg, newBufferResult);
    builder.create<mlir::async::YieldOp>(loc, nndmaOp->getResults());

    auto newExecOpToken = newExecOp.getToken();
    userExecOp.getDependenciesMutable().append(newExecOpToken);

    // use the outBuff of the NNDMA as input for the SwKernel
    auto newExecOpResult = newExecOp.getBodyResults()[0];
    auto newExecOpAsyncType = newExecOpResult.getType().dyn_cast<mlir::async::ValueType>();
    userExecOp.getBodyOperandsMutable().append(newExecOpResult);
    return userExecOp.getBody()->addArgument(newExecOpAsyncType.getValueType(), newExecOpResult.getLoc());
}

void addOutputCopy(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Location loc,
                   mlir::async::ExecuteOp userExecOp, mlir::BlockArgument blockArg, mlir::OpResult newBufferResult,
                   unsigned blockArgIndex) {
    builder.setInsertionPointAfter(userExecOp);

    // if there are multiple outputs get the index of the one that used the blockArg
    auto userExecOpResult = userExecOp.getBodyResults()[blockArgIndex];
    auto userExecOpToken = userExecOp.getToken();
    auto newExecOp = builder.create<mlir::async::ExecuteOp>(loc, newBufferResult.getType(), userExecOpToken,
                                                            userExecOpResult, nullptr);
    VPUIP::VPUIPDialect::setExecutor(newExecOp,
                                     vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::ExecutorKind::DMA_NN)));

    auto bodyBlock = newExecOp.getBody();
    builder.setInsertionPointToStart(bodyBlock);
    auto nndmaInput = bodyBlock->getArgument(0);
    auto nndmaOp = builder.create<VPUIP::NNDMAOp>(loc, nndmaInput, blockArg);
    builder.create<mlir::async::YieldOp>(loc, nndmaOp->getResults());

    // modify async::await to wait after the result of the NNDMA
    auto bodyResultUsers = userExecOpResult.getUsers();
    auto awaitUser = llvm::find_if(bodyResultUsers, [](const auto& user) {
        return mlir::isa<mlir::async::AwaitOp>(user);
    });

    if (awaitUser != bodyResultUsers.end()) {
        auto awaitOp = mlir::dyn_cast<mlir::async::AwaitOp>(*awaitUser);
        awaitOp.setOperand(newExecOp.getBodyResults()[0]);
    }
}

//
// AddCopyBetweenSWKernelsAndNetworkIOPass
//

class AddCopyBetweenSWKernelsAndNetworkIOPass final :
        public VPUIP::AddCopyBetweenSWKernelsAndNetworkIOBase<AddCopyBetweenSWKernelsAndNetworkIOPass> {
public:
    explicit AddCopyBetweenSWKernelsAndNetworkIOPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    FunctionCalls _functionCalls;
    FunctionWithSwKernelCalls _functionWithSwKernelCalls;

    void safeRunOnModule() final;
};

void AddCopyBetweenSWKernelsAndNetworkIOPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    mlir::func::FuncOp mainFuncOp;
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.isPrivate()) {
            return;
        }

        mainFuncOp = funcOp;

        mainFuncOp.walk([&](mlir::func::CallOp callOp) {
            auto calledFuncOp = vpux::getCalledFunction(callOp);
            if (calledFuncOp && calledFuncOp.isPrivate()) {
                _functionCalls[calledFuncOp].push_back(callOp);
            }
        });
    });

    for (const auto& pair : _functionCalls) {
        auto func = pair.first;
        mlir::DenseSet<int64_t> inputIndices;
        mlir::DenseSet<int64_t> outputIndices;
        func.walk([&](VPUIP::SwKernelOp swKernelOp) {
            auto swKernelInputs = swKernelOp.getInputs();
            for (auto input : swKernelInputs) {
                if (auto inputBlockArg = mlir::dyn_cast<mlir::BlockArgument>(input)) {
                    inputIndices.insert(inputBlockArg.getArgNumber());
                }
            }
            auto swKernelOutput = swKernelOp.getOutputBuffs();
            for (auto output : swKernelOutput) {
                if (auto outputBlockArg = mlir::dyn_cast<mlir::BlockArgument>(output)) {
                    outputIndices.insert(outputBlockArg.getArgNumber());
                }
            }
        });

        if (inputIndices.empty() && outputIndices.empty()) {
            continue;
        }
        _functionWithSwKernelCalls[func] = std::make_tuple(inputIndices, outputIndices, pair.second);
    }

    mlir::OpBuilder builder(mainFuncOp);
    auto ctx = builder.getContext();

    auto funcArgs = mainFuncOp.getArguments();
    for (const auto& arg : funcArgs) {
        auto blockArg = mlir::cast<mlir::BlockArgument>(arg);
        auto blockArgUses = blockArg.getUses();
        for (auto& use : blockArgUses) {
            if (!mlir::isa_and_nonnull<VPUIP::SwKernelOp>(use.getOwner())) {
                continue;
            }

            auto swKernelUser = mlir::cast<VPUIP::SwKernelOp>(use.getOwner());

            builder.setInsertionPointToStart(&mainFuncOp.getBody().front());

            auto loc = swKernelUser.getLoc();

            // find blockArg index
            auto blockArgIndex = use.getOperandNumber();

            auto userExecOp = swKernelUser->getParentOfType<mlir::async::ExecuteOp>();

            // declare new DDR buffer
            auto blockArgType = blockArg.getType();
            mlir::Operation* newBufferOp =
                    builder.create<mlir::memref::AllocOp>(loc, blockArgType.cast<mlir::MemRefType>());
            auto newBufferResult = newBufferOp->getResult(0);

            auto isInputUser = llvm::any_of(swKernelUser.getInputs(), [blockArg](auto swKernelInput) {
                return swKernelInput == blockArg;
            });
            if (isInputUser) {
                auto innerAsyncArg = addInputCopy(ctx, builder, loc, userExecOp, blockArg, newBufferResult);
                swKernelUser.setOperand(blockArgIndex, innerAsyncArg);
            } else {
                // use a new DDR buffer instead of the blockArg as output for the SwKernel
                swKernelUser.setOperand(blockArgIndex, newBufferResult);

                // if there are multiple SW.Kernel.runs get the index of the one that used the blockArg
                auto blockArgOutputIndex = blockArgIndex - swKernelUser.getInputs().size();
                addOutputCopy(ctx, builder, loc, userExecOp, blockArg, newBufferResult, blockArgOutputIndex);
            }
        }
    }

    for (auto& [privateFunc, callOpData] : _functionWithSwKernelCalls) {
        auto inputIdxs = std::get<0>(callOpData);
        auto outputIdxs = std::get<1>(callOpData);
        auto callOps = std::get<2>(callOpData);
        for (auto& callOp : callOps) {
            auto calledFuncArgs = callOp.getOperands();
            for (auto calledFuncArg : calledFuncArgs) {
                if (!mlir::isa<mlir::BlockArgument>(calledFuncArg)) {
                    continue;
                }
                auto calledFuncBlockArg = mlir::cast<mlir::BlockArgument>(calledFuncArg);

                auto isMainFuncBlockArg = llvm::any_of(funcArgs, [&calledFuncBlockArg](auto funcArg) {
                    return funcArg == calledFuncBlockArg;
                });

                if (!isMainFuncBlockArg) {
                    continue;
                }

                auto operandIt = llvm::find(callOp.getArgOperands(), calledFuncBlockArg);
                auto callOpArgIndex = std::distance(callOp.getArgOperands().begin(), operandIt);

                auto isSwKernelInputArg = inputIdxs.contains(callOpArgIndex);
                auto isSwKernelOutputArg = outputIdxs.contains(callOpArgIndex);

                if (!isSwKernelInputArg && !isSwKernelOutputArg) {
                    continue;
                }

                auto loc = callOp.getLoc();

                auto callOpExecOp = callOp->getParentOfType<mlir::async::ExecuteOp>();
                auto callOperation = callOp.getOperation();

                builder.setInsertionPointToStart(&mainFuncOp.getBody().front());
                auto blockArgType = calledFuncBlockArg.getType();
                mlir::Operation* newBufferOp =
                        builder.create<mlir::memref::AllocOp>(loc, blockArgType.cast<mlir::MemRefType>());
                auto newBufferResult = newBufferOp->getResult(0);

                if (isSwKernelInputArg && isSwKernelOutputArg) {
                    auto callOpExecOpDependencies = callOpExecOp.getDependencies();
                    auto callOpExecOpDependenciesVector = to_small_vector(callOpExecOpDependencies);

                    builder.setInsertionPoint(callOpExecOp);

                    // create async::ExecuteOp for the new NNDMA
                    auto newExecOp = builder.create<mlir::async::ExecuteOp>(
                            loc, newBufferResult.getType(), callOpExecOpDependenciesVector, std::nullopt, nullptr);
                    VPUIP::VPUIPDialect::setExecutor(
                            newExecOp, vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::ExecutorKind::DMA_NN)));

                    auto bodyBlock = newExecOp.getBody();
                    builder.setInsertionPointToStart(bodyBlock);
                    auto nndmaOp = builder.create<VPUIP::NNDMAOp>(loc, calledFuncBlockArg, newBufferResult);
                    builder.create<mlir::async::YieldOp>(loc, nndmaOp->getResults());

                    auto newExecOpToken = newExecOp.getToken();
                    callOpExecOp.getDependenciesMutable().append(newExecOpToken);

                    // use the outBuff of the NNDMA as input for the SwKernel
                    auto newExecOpResult = newExecOp.getBodyResults()[0];
                    auto newExecOpAsyncType = newExecOpResult.getType().dyn_cast<mlir::async::ValueType>();
                    callOpExecOp.getBodyOperandsMutable().append(newExecOpResult);
                    auto innerAsyncArg = callOpExecOp.getBody()->addArgument(newExecOpAsyncType.getValueType(),
                                                                             newExecOpResult.getLoc());

                    callOperation->setOperand(callOpArgIndex, innerAsyncArg);

                    // create async::ExecuteOp for the new NNDMA
                    builder.setInsertionPointAfter(callOpExecOp);

                    // if there are multiple outputs get the index of the one that used the blockArg
                    const auto calledFuncOpNumInputs = privateFunc.getNumArguments() - privateFunc.getNumResults();
                    auto blockArgOutputIndex = callOpArgIndex - calledFuncOpNumInputs;
                    auto callOpExecOpResult = callOpExecOp.getBodyResults()[blockArgOutputIndex];
                    auto callOpExecOpToken = callOpExecOp.getToken();
                    auto outputCopyExecOp = builder.create<mlir::async::ExecuteOp>(
                            loc, newBufferResult.getType(), callOpExecOpToken, callOpExecOpResult, nullptr);
                    VPUIP::VPUIPDialect::setExecutor(
                            outputCopyExecOp,
                            vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::ExecutorKind::DMA_NN)));

                    auto outputCopyExecOpBodyBlock = outputCopyExecOp.getBody();
                    builder.setInsertionPointToStart(outputCopyExecOpBodyBlock);
                    auto nndmaInput = outputCopyExecOpBodyBlock->getArgument(0);
                    auto outputNndmaOp = builder.create<VPUIP::NNDMAOp>(loc, nndmaInput, calledFuncBlockArg);
                    builder.create<mlir::async::YieldOp>(loc, outputNndmaOp->getResults());

                    // modify async::await to wait after the result of the NNDMA
                    auto bodyResultUsers = callOpExecOpResult.getUsers();
                    auto awaitUser = llvm::find_if(bodyResultUsers, [](const auto& user) {
                        return mlir::isa<mlir::async::AwaitOp>(user);
                    });

                    if (awaitUser != bodyResultUsers.end()) {
                        auto awaitOp = mlir::dyn_cast<mlir::async::AwaitOp>(*awaitUser);
                        awaitOp.setOperand(newExecOp.getBodyResults()[0]);
                    }
                } else if (isSwKernelInputArg) {
                    auto innerAsyncArg =
                            addInputCopy(ctx, builder, loc, callOpExecOp, calledFuncBlockArg, newBufferResult);
                    callOperation->setOperand(callOpArgIndex, innerAsyncArg);
                } else if (isSwKernelOutputArg) {
                    // use a new DDR buffer instead of the blockArg for the call op
                    callOperation->setOperand(callOpArgIndex, newBufferResult);

                    const auto calledFuncOpNumInputs = privateFunc.getNumArguments() - privateFunc.getNumResults();
                    auto blockArgOutputIndex = callOpArgIndex - calledFuncOpNumInputs;

                    addOutputCopy(ctx, builder, loc, callOpExecOp, calledFuncBlockArg, newBufferResult,
                                  blockArgOutputIndex);
                }
            }
        }
    }
}
}  // namespace

//
// createAddCopyBetweenSWKernelsAndNetworkIOPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(Logger log) {
    return std::make_unique<AddCopyBetweenSWKernelsAndNetworkIOPass>(log);
}

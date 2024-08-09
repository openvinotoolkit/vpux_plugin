//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/format.hpp"

using namespace vpux;

namespace outliner {

struct FuncsInfo {
    SmallVector<SmallVector<mlir::Type>> inputTypes;
    SmallVector<SmallVector<mlir::Type>> outputTypes;
    SmallVector<std::string> funcNames;
};

//
// OutlinerBase
//

class OutlinerBase {
public:
    virtual ~OutlinerBase() = default;

    void outline(mlir::ModuleOp moduleOp, StringRef functionSuffix) {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto outlinedTargets = getOutliningTargets(netFunc);
        if (outlinedTargets.empty()) {
            _log.debug("Empty outline targets");
            return;
        }

        const auto extractFuncInfo = [&](auto& outlinedTargets, auto netFunc) {
            SmallVector<SmallVector<mlir::Type>> inputTypes;
            SmallVector<SmallVector<mlir::Type>> outputTypes;
            SmallVector<std::string> funcNames;
            for (const auto& slices : outlinedTargets | indexed) {
                const auto& slice = slices.value().front();

                SmallVector<mlir::Type> sliceInputTypes;
                SmallVector<mlir::Type> sliceOutputTypes;
                for (const auto input : slice.inputs) {
                    sliceInputTypes.push_back(input.getType());
                }
                for (const auto output : slice.outputs) {
                    sliceOutputTypes.push_back(output.getType());
                }
                inputTypes.push_back(sliceInputTypes);
                outputTypes.push_back(sliceOutputTypes);
                funcNames.push_back(printToString("{0}_{1}{2}", netFunc.getName(), functionSuffix, slices.index() + 1));
            }
            return FuncsInfo{std::move(inputTypes), std::move(outputTypes), std::move(funcNames)};
        };

        auto funcsInfo = extractFuncInfo(outlinedTargets, netFunc);
        buildFuncOps(moduleOp, funcsInfo, outlinedTargets);
        buildCallOps(moduleOp, funcsInfo, outlinedTargets);
    }

protected:
    OutlinerBase(const Logger& log): _log(log) {
    }

    Logger getLogger() const {
        return _log;
    }

private:
    virtual SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp funcOp) = 0;
    virtual void buildFuncOps(mlir::ModuleOp moduleOp, const FuncsInfo& funcsInfo,
                              ArrayRef<OutliningInstance> outlinedTargets) = 0;
    virtual void buildCallOps(mlir::ModuleOp moduleOp, const FuncsInfo& funcsInfo,
                              ArrayRef<OutliningInstance> outlinedTargets) = 0;

private:
    Logger _log;
};

//
// Naive
//

class Naive final : public OutlinerBase {
public:
    Naive(size_t numParts, const Logger& log): OutlinerBase(log), _splitter(numParts, log) {
    }

    static constexpr StringRef name() {
        return "naive";
    }

private:
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp netFunc) override {
        return _splitter.getOutliningTargets(netFunc);
    }

    void buildFuncOps(mlir::ModuleOp moduleOp, const FuncsInfo& funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& slices : outlinedTargets | indexed) {
            const auto& slice = slices.value().front();
            auto sliceIdx = slices.index();
            const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo.inputTypes[sliceIdx]),
                                                          ArrayRef(funcsInfo.outputTypes[sliceIdx]));
            const auto funcLoc = appendLoc(netFunc.getLoc(), "_part{0}", sliceIdx + 1);
            auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo.funcNames[sliceIdx], funcType);
            func.setPrivate();

            OpBuilderLogger builderLog(getLogger().nest());
            auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

            mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
            for (size_t i = 0; i < slice.inputs.size(); i++) {
                oldToNewMap[slice.inputs[i]] = func.getArgument(i);
            }
            for (const auto op : slice.operations) {
                mlir::IRMapping mapper;
                for (auto operand : op->getOperands()) {
                    mapper.map(operand, oldToNewMap[operand]);
                }
                auto clonedOp = builder.clone(*op, mapper);
                if (mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::FakeConvertOp>(op)) {
                    clonedOp->setLoc(appendLoc(clonedOp->getLoc(), formatv("_part{0}", sliceIdx + 1).str()));
                }
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }
            SmallVector<mlir::Value> funcOutputFromSlices;
            for (const auto output : slice.outputs) {
                funcOutputFromSlices.push_back(oldToNewMap[output]);
            }
            const auto returnLoc = appendLoc(netFunc.getLoc(), "_part{0}_return", sliceIdx + 1);
            builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, const FuncsInfo& funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        OpBuilderLogger builderLog(getLogger().nest());
        auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front(), &builderLog);
        DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;

        SmallVector<mlir::Value> prevOutput;
        for (const auto& arg : netFunc.getArguments()) {
            oldToNewArgMap[arg] = arg;
        }

        for (const auto& [slicesIdx, slices] : outlinedTargets | indexed) {
            const auto& slice = slices.front();

            SmallVector<mlir::Value> newInputs;
            for (const auto input : slice.inputs) {
                newInputs.push_back(oldToNewArgMap[input]);
            }

            const auto callLoc = appendLoc(netFunc.getLoc(), "_part{0}_call", slicesIdx + 1);
            auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcsInfo.funcNames[slicesIdx],
                                                              funcsInfo.outputTypes[slicesIdx], newInputs);
            for (const auto& res : newCall.getResults()) {
                size_t idx = res.getResultNumber();
                oldToNewArgMap[slice.outputs[idx]] = res;
            }
        }
        netFunc.walk([&](mlir::func::ReturnOp ret) {
            for (auto i : irange(ret.getNumOperands())) {
                ret.setOperand(i, oldToNewArgMap[ret.getOperand(i)]);
            }
        });
    }

private:
    FunctionOutlinerNaive _splitter;
};

class RepeatingBlocks final : public OutlinerBase {
public:
    RepeatingBlocks(size_t minOpsInBlock, size_t maxNumIterations, const Logger& log)
            : OutlinerBase(log), _splitter(minOpsInBlock, maxNumIterations, log) {
    }

    static constexpr StringRef name() {
        return "repeating-blocks";
    }

private:
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp netFunc) override {
        return _splitter.getOutliningTargets(netFunc);
    }

    void buildFuncOps(mlir::ModuleOp moduleOp, const FuncsInfo& funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& slices : outlinedTargets | indexed) {
            // Only creates a single function for all instances
            const auto& slice = slices.value().front();
            auto sliceIdx = slices.index();
            const auto funcLoc = appendLoc(netFunc.getLoc(), "_fn{0}", sliceIdx + 1);
            const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo.inputTypes[sliceIdx]),
                                                          ArrayRef(funcsInfo.outputTypes[sliceIdx]));
            auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo.funcNames[sliceIdx], funcType);
            func.setPrivate();

            OpBuilderLogger builderLog(getLogger().nest());
            auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

            DenseMap<mlir::Value, mlir::Value> oldToNewMap;
            for (size_t i = 0; i < slice.inputs.size(); i++) {
                oldToNewMap[slice.inputs[i]] = func.getArgument(i);
            }
            for (const auto op : slice.operations) {
                mlir::IRMapping mapper;
                for (auto operand : op->getOperands()) {
                    mapper.map(operand, oldToNewMap[operand]);
                }
                auto clonedOp = builder.clone(*op, mapper);

                // Override the connection from the block arguments of the function to the user operation if it was
                // explicitly mapped by the analysis. This helps cover the case where the first instance (the one being
                // cloned) has some inputs reused, while the other instances may use separate arguments
                // E.g.: %0 = ...
                //       %1 = Add(%0, %0)  // instance 1
                //       %2 = Add(%0, %1)  // instance 2
                // The outlined function containing Add should have two operands, each connected to one operand. The
                // first call will pass the same value twice, while the second instance will pass different values
                if (!slice.inputUserMapping.empty()) {
                    for (auto& operand : clonedOp->getOpOperands()) {
                        if (!mlir::isa<mlir::BlockArgument>(operand.get())) {
                            continue;
                        }
                        const auto inputMappingIt = llvm::find_if(
                                slice.inputUserMapping, [&](const std::pair<mlir::Operation*, size_t>& user) {
                                    return user.first == op && user.second == operand.getOperandNumber();
                                });
                        if (inputMappingIt == slice.inputUserMapping.end()) {
                            continue;
                        }
                        const auto argIdx = std::distance(slice.inputUserMapping.begin(), inputMappingIt);
                        clonedOp->setOperand(operand.getOperandNumber(), func.getArgument(argIdx));
                    }
                }

                if (mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::FakeConvertOp>(op)) {
                    clonedOp->setLoc(appendLoc(clonedOp->getLoc(), formatv("_fn{0}", sliceIdx + 1).str()));
                }
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }

            SmallVector<mlir::Value> funcOutputFromSlices;
            for (const auto output : slice.outputs) {
                funcOutputFromSlices.push_back(oldToNewMap[output]);
            }
            const auto returnLoc = appendLoc(netFunc.getLoc(), "_fn{0}_return", sliceIdx + 1);
            builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, const FuncsInfo& funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        OpBuilderLogger builderLog(getLogger().nest());
        auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front(), &builderLog);
        DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;

        SmallVector<mlir::Value> prevOutput;
        for (const auto& arg : netFunc.getArguments()) {
            oldToNewArgMap[arg] = arg;
        }

        for (const auto& [slicesIdx, slices] : outlinedTargets | indexed) {
            for (const auto& [sliceIdx, slice] : slices | indexed) {
                SmallVector<mlir::Value> newInputs;
                for (const auto input : slice.inputs) {
                    if (oldToNewArgMap.contains(input)) {
                        newInputs.push_back(oldToNewArgMap[input]);
                    } else {
                        newInputs.push_back(input);
                    }
                    if (auto producerOp = newInputs.back().getDefiningOp()) {
                        if (!producerOp->isBeforeInBlock(&(*builder.getInsertionPoint()))) {
                            builder.setInsertionPointAfter(producerOp);
                        }
                    }
                }

                const auto callLoc = appendLoc(netFunc.getLoc(), "_fn{0}_call{1}", slicesIdx + 1, sliceIdx);
                auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcsInfo.funcNames[slicesIdx],
                                                                  funcsInfo.outputTypes[slicesIdx], newInputs);
                for (const auto& res : newCall.getResults()) {
                    size_t idx = res.getResultNumber();
                    oldToNewArgMap[slice.outputs[idx]] = res;
                }
            }
        }
        netFunc.walk([&](mlir::Operation* op) {
            bool changedOperands = false;
            for (auto i : irange(op->getNumOperands())) {
                if (oldToNewArgMap.find(op->getOperand(i)) != oldToNewArgMap.end()) {
                    op->setOperand(i, oldToNewArgMap[op->getOperand(i)]);
                    changedOperands = true;
                }
            }
            if (changedOperands) {
                moveOperationIfNeeded(op);
            }
        });
    }

    // It is possible for an instance of a repeating block to only contain a subset of the operations found in a slice
    // of the IR. When that happens, it is necessary to check whether the operations that are interleaved with those
    // from the instance need to be moved after the newly-inserted call operation
    void moveOperationIfNeeded(mlir::Operation* op) {
        bool movedOperation = false;
        for (auto operand : op->getOperands()) {
            auto producerOp = operand.getDefiningOp();
            if (producerOp == nullptr) {
                continue;
            }
            if (op->isBeforeInBlock(producerOp)) {
                op->moveAfter(producerOp);
                movedOperation = true;
            }
        }

        if (movedOperation) {
            for (auto result : op->getResults()) {
                for (auto userOp : result.getUsers()) {
                    moveOperationIfNeeded(userOp);
                }
            }
        }
    }

private:
    FunctionOutlinerRepeatingBlocks _splitter;
};

//
// Batching
//

class Batching final : public OutlinerBase {
public:
    Batching(const Logger& log): OutlinerBase(log), _splitter(log) {
    }

    static constexpr StringRef name() {
        return "batching";
    }

private:
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp netFunc) override {
        return _splitter.getOutliningTargets(netFunc);
    }

    void buildFuncOps(mlir::ModuleOp moduleOp, const FuncsInfo& funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& slices : outlinedTargets | indexed) {
            const auto& slice = slices.value().front();
            auto sliceIdx = slices.index();
            const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo.inputTypes[sliceIdx]),
                                                          ArrayRef(funcsInfo.outputTypes[sliceIdx]));
            auto func = builder.create<mlir::func::FuncOp>(moduleOp.getLoc(), funcsInfo.funcNames[sliceIdx], funcType);
            func.setPrivate();

            OpBuilderLogger builderLog(getLogger().nest());
            auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

            mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
            for (size_t i = 0; i < slice.inputs.size(); i++) {
                oldToNewMap[slice.inputs[i]] = func.getArgument(i);
            }
            for (const auto op : slice.operations) {
                mlir::IRMapping mapper;
                for (auto operand : op->getOperands()) {
                    mapper.map(operand, oldToNewMap[operand]);
                }
                auto clonedOp = builder.clone(*op, mapper);
                if (mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::FakeConvertOp>(op)) {
                    clonedOp->setLoc(appendLoc(clonedOp->getLoc(), formatv("_fn{0}", sliceIdx + 1).str()));
                }
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }
            SmallVector<mlir::Value> funcOutputFromSlices;
            for (const auto output : slice.outputs) {
                funcOutputFromSlices.push_back(oldToNewMap[output]);
            }
            builder.create<mlir::func::ReturnOp>(func.getLoc(), funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, const FuncsInfo& funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        OpBuilderLogger builderLog(getLogger().nest());
        auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front(), &builderLog);
        DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;

        SmallVector<mlir::Value> prevOutput;
        for (const auto& arg : netFunc.getArguments()) {
            oldToNewArgMap[arg] = arg;
        }

        for (const auto& slices : outlinedTargets | indexed) {
            for (const auto& slice : slices.value() | indexed) {
                SmallVector<mlir::Value> newInputs;
                for (const auto input : slice.value().inputs) {
                    if (oldToNewArgMap.contains(input)) {
                        newInputs.push_back(oldToNewArgMap[input]);
                    } else {
                        newInputs.push_back(input);
                    }
                    if (auto producerOp = newInputs.back().getDefiningOp()) {
                        if (!producerOp->isBeforeInBlock(&(*builder.getInsertionPoint()))) {
                            builder.setInsertionPointAfter(producerOp);
                        }
                    }
                }

                auto newCall = builder.create<mlir::func::CallOp>(netFunc.getLoc(), funcsInfo.funcNames[slices.index()],
                                                                  funcsInfo.outputTypes[slices.index()], newInputs);
                for (const auto& res : newCall.getResults()) {
                    size_t idx = res.getResultNumber();
                    oldToNewArgMap[slice.value().outputs[idx]] = res;
                }
            }
        }
        netFunc.walk([&](mlir::Operation* op) {
            for (auto i : irange(op->getNumOperands())) {
                if (oldToNewArgMap.find(op->getOperand(i)) != oldToNewArgMap.end()) {
                    op->setOperand(i, oldToNewArgMap[op->getOperand(i)]);
                }
            }
        });
    }

private:
    FunctionOutlinerBatching _splitter;
};

}  // namespace outliner

namespace {

//
// OutlinerPass
//

class OutlinerPass final : public IE::OutlinerBase<OutlinerPass> {
public:
    explicit OutlinerPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    mlir::LogicalResult delegateInitializeOptions(StringRef outliningMode);

private:
    void safeRunOnModule() final;

private:
    // TODO: #115448 the pass should not know explicitly about type of outliner and its parameters
    std::string _mode = "naive";
    size_t _numParts = 2;
    size_t _minOpsInBlock = 30;
    size_t _maxNumIterations = 50;
};

mlir::LogicalResult OutlinerPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (mode.hasValue()) {
        _mode = mode;
    }
    if (numParts.hasValue()) {
        _numParts = numParts;
    }
    if (minOpsInBlock.hasValue()) {
        _minOpsInBlock = minOpsInBlock;
    }
    if (maxNumIterations.hasValue()) {
        _maxNumIterations = maxNumIterations;
    }
    return mlir::success();
}

mlir::LogicalResult OutlinerPass::delegateInitializeOptions(StringRef outliningMode) {
    return Base::initializeOptions(printToString("{0}={1}", mode.getArgStr(), outliningMode));
}

//
// safeRunOnModule
//

void OutlinerPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    std::transform(_mode.begin(), _mode.end(), _mode.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    if (_mode == outliner::Naive::name()) {
        outliner::Naive outliner(_numParts, _log);
        outliner.outline(moduleOp, "part");
    } else if (_mode == outliner::RepeatingBlocks::name()) {
        outliner::RepeatingBlocks outliner(_minOpsInBlock, _maxNumIterations, _log);
        outliner.outline(moduleOp, "fn");
    } else if (_mode == outliner::Batching::name()) {
        outliner::Batching outliner(_log);
        outliner.outline(moduleOp, "batch");
    } else {
        _log.error("Unknown outliner mode: '{0}'", _mode);
        signalPassFailure();
    }
}

}  // namespace

//
// createOutlinerPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOutlinerPass(const std::string& mode, Logger log) {
    auto pass = std::make_unique<OutlinerPass>(log);
    if (mlir::failed(static_cast<OutlinerPass*>(pass.get())->delegateInitializeOptions(mode))) {
        VPUX_THROW("Incorrect option used for \"{0}\" pass initialization: {1}", pass->getName(), mode);
    }
    return pass;
}

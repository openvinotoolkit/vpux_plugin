//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/format.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/RegionUtils.h>

using namespace vpux;

namespace {

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

}  // namespace

namespace outliner {

struct FuncInfo {
    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Type> outputTypes;
    std::string funcNames;
};

//
// OutlinerBase
//

class OutlinerBase {
public:
    virtual ~OutlinerBase() = default;

    virtual void outline(mlir::ModuleOp moduleOp, StringRef functionSuffix) {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto outlinedTargets = getOutliningTargets(netFunc);
        if (outlinedTargets.empty()) {
            _log.debug("Empty outline targets");
            return;
        }

        _log.info("Creating {0} functions", outlinedTargets.size());

        SmallVector<SmallVector<FuncInfo>> funcsInfo(outlinedTargets.size());
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            const auto slice = slices.front();
            SmallVector<mlir::Type> inputTypes;
            SmallVector<mlir::Type> outputTypes;
            for (const auto input : slice.inputs) {
                inputTypes.push_back(input.getType());
            }
            for (const auto output : slice.outputs) {
                outputTypes.push_back(output.getType());
            }
            const auto funcName = printToString("{0}_{1}{2}", netFunc.getName(), functionSuffix, targetIdx + 1);
            funcsInfo[targetIdx].push_back({std::move(inputTypes), std::move(outputTypes), funcName});
        }

        buildFuncOps(moduleOp, funcsInfo, outlinedTargets);
        buildCallOps(moduleOp, funcsInfo, outlinedTargets);
    }

protected:
    OutlinerBase(std::unique_ptr<IFunctionOutliner> splitter, const Logger& log)
            : _splitter(std::move(splitter)), _log(log) {
    }

    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp funcOp) {
        return _splitter->getOutliningTargets(funcOp);
    }

    Logger getLogger() const {
        return _log;
    }

private:
    virtual void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                              ArrayRef<OutliningInstance> outlinedTargets) = 0;
    virtual void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                              ArrayRef<OutliningInstance> outlinedTargets) = 0;

private:
    std::unique_ptr<IFunctionOutliner> _splitter;
    Logger _log;
};

//
// Naive
//

class Naive final : public OutlinerBase {
public:
    Naive(size_t numParts, const Logger& log)
            : OutlinerBase(std::make_unique<FunctionOutlinerNaive>(numParts, log), log) {
    }

    static constexpr StringRef name() {
        return "naive";
    }

private:
    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            const auto& slice = slices.front();
            const size_t sliceIdx = 0;
            const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                          ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
            const auto funcLoc = appendLoc(netFunc.getLoc(), "_part{0}", targetIdx + 1);
            auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcNames, funcType);
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
                // The input pre-processing operations might be duplicated in multiple functions depending on how
                // their results are used, so their location is customized during cloning
                if (mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::FakeConvertOp>(op)) {
                    clonedOp->setLoc(appendLoc(clonedOp->getLoc(), formatv("_part{0}", targetIdx + 1).str()));
                }
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }
            SmallVector<mlir::Value> funcOutputFromSlices;
            for (const auto output : slice.outputs) {
                funcOutputFromSlices.push_back(oldToNewMap[output]);
            }
            const auto returnLoc = appendLoc(netFunc.getLoc(), "_part{0}_return", targetIdx + 1);
            builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
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

        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            const auto& slice = slices.front();
            const size_t sliceIdx = 0;

            SmallVector<mlir::Value> newInputs;
            for (const auto input : slice.inputs) {
                newInputs.push_back(oldToNewArgMap[input]);
            }

            const auto callLoc = appendLoc(netFunc.getLoc(), "_part{0}_call", targetIdx + 1);
            auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx][sliceIdx].funcNames,
                                                              funcsInfo[targetIdx][sliceIdx].outputTypes, newInputs);
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
};

//
// RepeatingBlocks
//

class RepeatingBlocks final : public OutlinerBase {
public:
    RepeatingBlocks(size_t minOpsInBlock, size_t maxNumIterations, bool weightsAsInputs, const Logger& log)
            : OutlinerBase(std::make_unique<FunctionOutlinerRepeatingBlocks>(minOpsInBlock, maxNumIterations,
                                                                             /*separateFunctions=*/false,
                                                                             weightsAsInputs, log),
                           log) {
    }

    static constexpr StringRef name() {
        return "repeating-blocks";
    }

private:
    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            // Only creates a single function for all instances
            const auto& slice = slices.front();
            const size_t sliceIdx = 0;
            const auto funcLoc = appendLoc(netFunc.getLoc(), "_fn{0}", targetIdx + 1);
            const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                          ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
            auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcNames, funcType);
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

                // The input pre-processing operations might be duplicated in multiple functions depending on how
                // their results are used, so their location is customized during cloning
                if (mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::FakeConvertOp>(op)) {
                    clonedOp->setLoc(appendLoc(clonedOp->getLoc(), formatv("_fn{0}", targetIdx + 1).str()));
                }
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }

            SmallVector<mlir::Value> funcOutputFromSlices;
            for (const auto output : slice.outputs) {
                funcOutputFromSlices.push_back(oldToNewMap[output]);
            }
            const auto returnLoc = appendLoc(netFunc.getLoc(), "_fn{0}_return", targetIdx + 1);
            builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
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

        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
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

                const auto callLoc = appendLoc(netFunc.getLoc(), "fn_{0}_call_{1}", targetIdx + 1, sliceIdx);
                auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx].front().funcNames,
                                                                  funcsInfo[targetIdx].front().outputTypes, newInputs);
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
};

//
// RepeatingBlocksSeparateFunctions
//

class RepeatingBlocksSeparateFunctions final : public OutlinerBase {
public:
    RepeatingBlocksSeparateFunctions(size_t minOpsInBlock, size_t maxNumIterations, const Logger& log)
            : OutlinerBase(std::make_unique<FunctionOutlinerRepeatingBlocks>(minOpsInBlock, maxNumIterations,
                                                                             /*separateFunctions=*/true,
                                                                             /*weightsAsInputs=*/false, log),
                           log) {
    }

    static constexpr StringRef name() {
        return "repeating-blocks-separate-functions";
    }

    void outline(mlir::ModuleOp moduleOp, StringRef functionSuffix) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto outlinedTargets = getOutliningTargets(netFunc);
        if (outlinedTargets.empty()) {
            getLogger().debug("Empty outline targets");
            return;
        }

        size_t numFunctions = 0;
        for (const auto& instance : outlinedTargets) {
            numFunctions += instance.size();
        }
        getLogger().info("Creating {0} functions", numFunctions);

        SmallVector<SmallVector<FuncInfo>> funcsInfo(outlinedTargets.size());
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            for (const auto& [sliceIdx, slice] : slices | indexed) {
                SmallVector<mlir::Type> inputTypes;
                SmallVector<mlir::Type> outputTypes;
                for (const auto input : slice.inputs) {
                    inputTypes.push_back(input.getType());
                }
                for (const auto output : slice.outputs) {
                    outputTypes.push_back(output.getType());
                }
                const auto funcName = printToString("{0}_{1}{2}_block{3}", netFunc.getName(), functionSuffix,
                                                    targetIdx + 1, sliceIdx + 1);
                funcsInfo[targetIdx].push_back({std::move(inputTypes), std::move(outputTypes), funcName});
            }
        }

        buildFuncOps(moduleOp, funcsInfo, outlinedTargets);
        buildCallOps(moduleOp, funcsInfo, outlinedTargets);
    }

private:
    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            for (const auto& [sliceIdx, slice] : slices | indexed) {
                const auto funcLoc = appendLoc(netFunc.getLoc(), "_fn{0}_block{1}", targetIdx + 1, sliceIdx + 1);
                const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                              ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
                auto func =
                        builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcNames, funcType);
                func.setPrivate();

                auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock());

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
                    // Some operation could be duplicated in the IR (e.g. input pre-processing ops, constant
                    // quantization ops), so their location is customized during cloning to distinguish between them
                    clonedOp->setLoc(appendLoc(clonedOp->getLoc(),
                                               formatv("_fn{0}_block{1}", targetIdx + 1, sliceIdx + 1).str()));
                    for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                        oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                    }
                }

                SmallVector<mlir::Value> funcOutputFromSlices;
                for (const auto output : slice.outputs) {
                    funcOutputFromSlices.push_back(oldToNewMap[output]);
                }
                const auto returnLoc =
                        appendLoc(netFunc.getLoc(), "fn_{0}_block_{1}_return", targetIdx + 1, sliceIdx + 1);
                builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
            }
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front());
        DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;

        SmallVector<mlir::Value> prevOutput;
        for (const auto& arg : netFunc.getArguments()) {
            oldToNewArgMap[arg] = arg;
        }

        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
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

                const auto callLoc = appendLoc(netFunc.getLoc(), "fn_{0}_call_{1}", targetIdx + 1, sliceIdx);
                auto newCall =
                        builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx][sliceIdx].funcNames,
                                                           funcsInfo[targetIdx][sliceIdx].outputTypes, newInputs);
                for (const auto& res : newCall.getResults()) {
                    size_t idx = res.getResultNumber();
                    oldToNewArgMap[slice.outputs[idx]] = res;
                }
            }
        }
        SmallVector<mlir::Operation*> opsToMove;
        netFunc.walk([&](mlir::Operation* op) {
            for (auto i : irange(op->getNumOperands())) {
                if (oldToNewArgMap.find(op->getOperand(i)) != oldToNewArgMap.end()) {
                    op->setOperand(i, oldToNewArgMap[op->getOperand(i)]);
                    if (opsToMove.empty() || opsToMove.back() != op) {
                        opsToMove.push_back(op);
                    }
                }
            }
        });
        for (auto op : opsToMove) {
            moveOperationIfNeeded(op);
        }

        mlir::IRRewriter rewriter(builder);
        (void)mlir::simplifyRegions(rewriter, netFunc->getRegions().front());
    }
};

//
// Batching
//

class Batching final : public OutlinerBase {
public:
    Batching(const Logger& log): OutlinerBase(std::make_unique<FunctionOutlinerBatching>(log), log) {
    }

    static constexpr StringRef name() {
        return "batching";
    }

private:
    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override {
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
        builder.setInsertionPoint(netFunc);

        auto* ctx = moduleOp.getContext();
        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
            const auto& slice = slices.front();
            const size_t sliceIdx = 0;
            const auto funcLoc = appendLoc(netFunc.getLoc(), "_fn{0}_block{1}", targetIdx + 1, sliceIdx + 1);
            const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                          ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
            auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcNames, funcType);
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
                // The input pre-processing operations might be duplicated in multiple functions depending on how
                // their results are used, so their location is customized during cloning
                if (mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp, IE::FakeQuantizeOp, IE::FakeConvertOp>(op)) {
                    clonedOp->setLoc(appendLoc(clonedOp->getLoc(), formatv("_fn{0}", targetIdx + 1).str()));
                }
                for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                    oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
                }
            }
            SmallVector<mlir::Value> funcOutputFromSlices;
            for (const auto output : slice.outputs) {
                funcOutputFromSlices.push_back(oldToNewMap[output]);
            }
            const auto returnLoc = appendLoc(netFunc.getLoc(), "fn_{0}_block_{1}_return", targetIdx + 1, sliceIdx + 1);
            builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
        }
    }

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
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

        for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
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

                const auto callLoc = appendLoc(netFunc.getLoc(), "fn_{0}_call_{1}", targetIdx + 1, sliceIdx);
                auto newCall =
                        builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx][sliceIdx].funcNames,
                                                           funcsInfo[targetIdx][sliceIdx].outputTypes, newInputs);
                for (const auto& res : newCall.getResults()) {
                    size_t idx = res.getResultNumber();
                    oldToNewArgMap[slice.outputs[idx]] = res;
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
    mlir::LogicalResult delegateInitializeOptions(StringRef functionOutlining);

private:
    void safeRunOnModule() final;

private:
    std::string _mode;
    vpux::OutlinerPassOptions _options;
};

mlir::LogicalResult OutlinerPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // Throws an exception if functionOutlining is not a valid string.
    _options = vpux::OutlinerPassOptions::createFromString(functionOutlining);

    return mlir::success();
}

mlir::LogicalResult OutlinerPass::delegateInitializeOptions(StringRef functionOutlining) {
    return Base::initializeOptions(printToString("{0}={1}", this->functionOutlining.getArgStr(), functionOutlining));
}

//
// safeRunOnModule
//

void OutlinerPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    for (size_t i = 0; i < _options.count(); ++i) {
        if (i >= 1) {
            _log.warning("Execution of fallback outliner solutions is not yet implemented!");
            break;
        }

        if (const auto* opt = _options.getIf<vpux::NaiveOptions>(i)) {
            outliner::Naive outliner(opt->numParts, _log);
            outliner.outline(moduleOp, "part");
        } else if (const auto* opt = _options.getIf<vpux::RepeatingBlocksOptions>(i)) {
            outliner::RepeatingBlocks outliner(opt->minOpsInBlock, opt->maxNumIterations, opt->weightsAsInputs, _log);
            outliner.outline(moduleOp, "fn");
        } else if (const auto* opt = _options.getIf<vpux::RepeatingBlocksSeparateFunctionsOptions>(i)) {
            outliner::RepeatingBlocksSeparateFunctions outliner(opt->minOpsInBlock, opt->maxNumIterations, _log);
            outliner.outline(moduleOp, "fn");
        } else if (const auto* opt = _options.getIf<vpux::BatchingOptions>(i)) {
            std::ignore = opt;
            outliner::Batching outliner(_log);
            outliner.outline(moduleOp, "batching");
        } else {
            llvm_unreachable("Validity of mode should have been checked in pass initialization!");
        }
    }
}

}  // namespace

//
// createOutlinerPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOutlinerPass(const std::string& functionOutlining, Logger log) {
    auto pass = std::make_unique<OutlinerPass>(log);
    if (mlir::failed(static_cast<OutlinerPass*>(pass.get())->delegateInitializeOptions(functionOutlining))) {
        VPUX_THROW("Incorrect option used for \"{0}\" pass initialization: {1}", pass->getName(), functionOutlining);
    }
    return pass;
}

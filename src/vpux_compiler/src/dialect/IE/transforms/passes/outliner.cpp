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

namespace {

struct FuncsInfo {
    SmallVector<SmallVector<mlir::Type>> inputTypes;
    SmallVector<SmallVector<mlir::Type>> outputTypes;
    SmallVector<std::string> funcNames;
};

//
// OutlinerPass
//

class OutlinerPass final : public IE::OutlinerBase<OutlinerPass> {
public:
    explicit OutlinerPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnModule() final;
    void buildFuncOps(const FuncsInfo& funcsInfo, ArrayRef<OutliningInstance> outliningTargets);
    void buildCallOps(const FuncsInfo& funcsInfo, ArrayRef<OutliningInstance> outliningTargets);

private:
    // TODO: #115448 the pass should not know explicitly about type of outliner and its parameters
    size_t _numParts = 2;
    Logger _log;
};

mlir::LogicalResult OutlinerPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!numParts.hasValue()) {
        return mlir::success();
    }

    _numParts = numParts;
    return mlir::success();
}

//
// safeRunOnModule
//

void OutlinerPass::safeRunOnModule() {
    auto module = getOperation();
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    // Create the split functions
    FunctionOutlinerNaive funcOutline(_numParts, _log);
    auto outlinedTargets = funcOutline.getOutliningTargets(netFunc);

    // Avoid Splitting function when splitting function is impossible given the amount of slices given.
    if (outlinedTargets.empty()) {
        _log.debug("Empty outline targets");
        return;
    }

    const auto extractFuncInfo = [](auto& outlinedTargets, auto netFunc) {
        SmallVector<SmallVector<mlir::Type>> inputTypes;
        SmallVector<SmallVector<mlir::Type>> outputTypes;
        SmallVector<std::string> funcNames;
        for (const auto& slices : outlinedTargets | indexed) {
            VPUX_THROW_WHEN(slices.value().size() != 1, "Slices size must be equal to 1 but currently is equal to {0}",
                            slices.value().size());
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
            funcNames.push_back(printToString("{0}_part{1}", netFunc.getName(), slices.index() + 1));
        }
        return FuncsInfo{std::move(inputTypes), std::move(outputTypes), std::move(funcNames)};
    };

    auto funcsInfo = extractFuncInfo(outlinedTargets, netFunc);
    buildFuncOps(funcsInfo, outlinedTargets);
    buildCallOps(funcsInfo, outlinedTargets);
}

void OutlinerPass::buildFuncOps(const FuncsInfo& funcsInfo, ArrayRef<OutliningInstance> outlinedTargets) {
    auto module = getOperation();
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    auto builder = mlir::OpBuilder(module.getBodyRegion());
    builder.setInsertionPoint(netFunc);

    auto* ctx = module.getContext();
    for (const auto& slices : outlinedTargets | indexed) {
        const auto& slice = slices.value().front();
        auto sliceIdx = slices.index();
        const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo.inputTypes[sliceIdx]),
                                                      ArrayRef(funcsInfo.outputTypes[sliceIdx]));
        auto func = builder.create<mlir::func::FuncOp>(module.getLoc(), funcsInfo.funcNames[sliceIdx], funcType);
        func.setPrivate();

        OpBuilderLogger builderLog(_log.nest());
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
        builder.create<mlir::func::ReturnOp>(func.getLoc(), funcOutputFromSlices);
    }
}

void OutlinerPass::buildCallOps(const FuncsInfo& funcsInfo, ArrayRef<OutliningInstance> outlinedTargets) {
    auto module = getOperation();
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front(), &builderLog);
    DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;

    SmallVector<mlir::Value> prevOutput;
    for (const auto& arg : netFunc.getArguments()) {
        oldToNewArgMap[arg] = arg;
    }

    for (const auto& slices : outlinedTargets | indexed) {
        const auto& slice = slices.value().front();

        SmallVector<mlir::Value> newInputs;
        for (const auto input : slice.inputs) {
            newInputs.push_back(oldToNewArgMap[input]);
        }

        auto newCall = builder.create<mlir::func::CallOp>(netFunc.getLoc(), funcsInfo.funcNames[slices.index()],
                                                          funcsInfo.outputTypes[slices.index()], newInputs);
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

}  // namespace

//
// createOutlinerPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOutlinerPass(Logger log) {
    return std::make_unique<OutlinerPass>(log);
}

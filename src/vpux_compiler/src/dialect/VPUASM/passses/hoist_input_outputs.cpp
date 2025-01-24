//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/VPUASM/passes.hpp"

using namespace vpux;

namespace {

class HoistInputOutputsPass final : public VPUASM::HoistInputOutputsBase<HoistInputOutputsPass> {
public:
    explicit HoistInputOutputsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    std::string symbolizeIO(IE::DataInfoOp di);
    void safeRunOnModule() final;
};

std::string HoistInputOutputsPass::symbolizeIO(IE::DataInfoOp di) {
    auto name = di.getName() + "_buffDecl";
    return name.str();
}

void HoistInputOutputsPass::safeRunOnModule() {
    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp netFunc;
    auto moduleOp = getOperation();
    auto ctx = moduleOp.getContext();

    IE::CNNNetworkOp::getFromModule(moduleOp, netOp, netFunc);
    // Build the initial IOBinder container OP
    auto moduleBuilder = mlir::OpBuilder(netFunc.getOperation());
    auto ioBindingsOp = moduleBuilder.create<VPUASM::IOBindingsOp>(netFunc.getLoc());

    // Prepare the builders
    auto netInputBuilder = mlir::OpBuilder::atBlockBegin(&ioBindingsOp.getInputDeclarations().front());
    auto netOutputBuilder = mlir::OpBuilder::atBlockBegin(&ioBindingsOp.getOutputDeclarations().front());
    auto netProfOutputBuilder = mlir::OpBuilder::atBlockBegin(&ioBindingsOp.getProfilingBuffDeclarations().front());
    auto funcBuilder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front());

    auto inputCount = netOp.getNetInputsCount();
    auto outputsCount = netOp.getNetOutputsCount();

    for (auto arg : llvm::enumerate(netFunc.getArguments())) {
        auto argIdx = arg.index();
        auto argVal = arg.value();

        bool isInput = argIdx < inputCount;
        bool isOutput = argIdx >= inputCount && argIdx < inputCount + outputsCount;
        bool isProfOutput = argIdx >= inputCount + outputsCount;
        VPUX_THROW_UNLESS(isInput || isOutput || isProfOutput, "Can't figure out argument type from its number");

        VPURT::BufferSection ioSec;
        uint64_t ioIdx = 0;
        if (isInput) {
            ioSec = VPURT::BufferSection::NetworkInput;
            ioIdx = argIdx;
        } else if (isOutput) {
            ioSec = VPURT::BufferSection::NetworkOutput;
            ioIdx = argIdx - inputCount;
        } else {
            ioSec = VPURT::BufferSection::ProfilingOutput;
            ioIdx = argIdx - inputCount - outputsCount;
        }

        // FuncArgs always declare at "offset 0" and no swizzling
        auto memLocation = VPUASM::MemLocationType::get(ctx, ioSec, ioIdx, 0);
        auto memref = argVal.getType().cast<mlir::MemRefType>();
        auto traits = VPUASM::BufferTraitsType::get(ctx, 0);

        auto buffType = VPUASM::BufferType::get(ctx, memLocation, memref, traits);

        llvm::SmallVector<vpux::IE::DataInfoOp> diVec;
        if (isInput) {
            diVec = netOp.getInputsDataInfo();
        } else if (isOutput) {
            diVec = netOp.getOutputsDataInfo();
        } else if (isProfOutput) {
            diVec = netOp.getProfilingOutputsDataInfo();
        }
        auto io_name = symbolizeIO(diVec[ioIdx]);

        if (isInput) {
            netInputBuilder.create<VPUASM::DeclareBufferOp>(netFunc.getLoc(), io_name, buffType);
        } else if (isOutput) {
            netOutputBuilder.create<VPUASM::DeclareBufferOp>(netFunc.getLoc(), io_name, buffType);
        } else if (isProfOutput) {
            netProfOutputBuilder.create<VPUASM::DeclareBufferOp>(netFunc.getLoc(), io_name, buffType);
        }

        // for IO assume baseOffset and swizzlingKeys to be 0
        auto io = funcBuilder.create<VPURT::DeclareBufferOp>(netFunc.getLoc(), memref, ioSec, ioIdx, 0, 0);

        argVal.replaceAllUsesWith(io);
    }

    // erase function arguments (set emtpy functionType)
    auto emptyFuncType = mlir::FunctionType::get(ctx, mlir::TypeRange(nullptr, 0), mlir::TypeRange(nullptr, 0));
    netFunc.setType(emptyFuncType);

    // erase blockArgs
    auto& netBlock = netFunc.getBody().front();
    mlir::BitVector eraseIndices(netBlock.getNumArguments());

    for (unsigned i = 0; i < netBlock.getNumArguments(); i++) {
        eraseIndices.set(i);
    }
    netBlock.eraseArguments(eraseIndices);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUASM::createHoistInputOutputsPass(Logger log) {
    return std::make_unique<HoistInputOutputsPass>(log);
}

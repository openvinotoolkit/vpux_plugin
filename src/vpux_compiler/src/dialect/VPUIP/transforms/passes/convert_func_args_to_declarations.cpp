//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {

//
// ConvertFuncArgsToDeclarationsPass
//

class ConvertFuncArgsToDeclarationsPass final :
        public VPUIP::ConvertFuncArgsToDeclarationsBase<ConvertFuncArgsToDeclarationsPass> {
public:
    explicit ConvertFuncArgsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertFuncArgsToDeclarationsPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    const auto getNewDecl = [](mlir::OpBuilder& argBuilder, mlir::Value val, VPURT::BufferSection section,
                               int64_t sectionIndex) {
        return argBuilder.create<VPURT::DeclareBufferOp>(val.getLoc(), val.getType(), section, sectionIndex, 0)
                .getResult();
    };

    const auto replaceArgs = [&](mlir::func::FuncOp funcOp, auto getDecl) {
        VPUX_THROW_WHEN(funcOp.isExternal(), "It is assumed that the method is run only for functions with body");
        VPUX_THROW_UNLESS(funcOp.getNumArguments() >= funcOp.getNumResults(), "Function '{0}' is not bufferized",
                          funcOp);

        const auto numInputs = funcOp.getNumArguments() - funcOp.getNumResults();

        auto returnOp = *funcOp.getOps<mlir::func::ReturnOp>().begin();
        auto& firstOp = *funcOp.getOps().begin();

        auto argBuilder = mlir::OpBuilder::atBlockBegin(&funcOp.getBody().front());

        const auto replaceUse = [&](mlir::ValueRange args, VPURT::BufferSection section) {
            for (auto p : args | indexed) {
                auto val = p.value();

                if (val.getUses().empty()) {
                    continue;
                }

                argBuilder.setInsertionPoint(&firstOp);
                auto newArg = getDecl(argBuilder, val, section, p.index());

                _log.trace("Replace all uses of '{0}' with '{1}'",
                           newArg.template getDefiningOp<VPURT::DeclareBufferOp>());
                val.replaceAllUsesExcept(newArg, llvm::SmallPtrSet<mlir::Operation*, 1>{returnOp});
            }
        };

        replaceUse(funcOp.getArguments().take_front(numInputs), VPURT::BufferSection::NetworkInput);
        replaceUse(funcOp.getArguments().drop_front(numInputs), VPURT::BufferSection::NetworkOutput);
    };

    mlir::func::FuncOp netFunc;
    vpux::IE::CNNNetworkOp cnnOp;
    vpux::IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, netFunc);

    replaceArgs(netFunc, getNewDecl);
    netFunc.walk([&](mlir::func::CallOp callOp) {
        auto func = getCalledFunction(callOp);

        // TODO:#108930 -- sanity check
        auto maybeUses = func.getSymbolUses(func->getParentOfType<mlir::ModuleOp>());
        VPUX_THROW_WHEN(!maybeUses.has_value() || maybeUses.value().empty(),
                        "An unused function {0}. It must either be an entry point or be called by the main",
                        func.getSymName());
        auto uses = maybeUses.value();
        VPUX_THROW_WHEN(std::distance(uses.begin(), uses.end()) != 1, "At the moment, several calls are not supported");

        const auto isPureViewLike = [](mlir::Operation* op) {
            return mlir::isa<mlir::ViewLikeOpInterface>(op) && !mlir::isa<VPUIP::LayerOpInterface>(op);
        };

        const auto getDeclaration = [&](mlir::OpBuilder& argBuilder, mlir::Value val, VPURT::BufferSection, int64_t) {
            auto funcBlockArg = mlir::cast<mlir::BlockArgument>(val);

            auto callOperand = callOp.getOperand(funcBlockArg.getArgNumber());
            auto producerOp = callOperand.getDefiningOp();
            SmallVector<mlir::Operation*> viewOps;
            while (producerOp != nullptr && isPureViewLike(producerOp)) {
                viewOps.push_back(producerOp);
                producerOp = mlir::cast<mlir::ViewLikeOpInterface>(producerOp).getViewSource().getDefiningOp();
            }

            auto originDeclOp = mlir::dyn_cast_or_null<VPURT::DeclareBufferOp>(producerOp);
            VPUX_THROW_WHEN(originDeclOp == nullptr, "Could not find declare buffer op for CallOp '{0}' operand '{1}'",
                            callOp, callOperand);

            // clang-format off
            // Clone full chain of pure view like ops to get correct offset in the "child" function after ConvertViewOpsToDeclarations
            // For example:
            // main: %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x10x10x12xf16, @DDR>
            // %1 = VPUIP.SubView %0 [0, 1] [1, 8] : memref<1x10f16, @DDR> to memref<1x8xf16, {order = #NC, strides = [10, 1]}, @DDR>
            //
            // foo1:
            // original argument is %arg1: memref<1x8xf16, {order = #NC, strides = [10, 1]}, @DDR>
            // when clone only DeclareBufferOp:
            // %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x10x10x12xf16, @DDR>
            // %1 = VPUIP.SubView %0 [0, 2] [1, 6] : memref<1x8xf16, {order = #NC, strides = [10, 1]}, @DDR> to memref<1x6xf16, {order = #NC, strides = [10, 1]}, @DDR>
            //
            // After ConvertViewOpsToDeclarations offset is 2, but shoud be 3 = 1("main" offset) + 2("foo1" offset)
            // clang-format on

            auto resValue = argBuilder.clone(*originDeclOp.getOperation())->getResult(0);
            for (auto& currViewOp : viewOps | reversed) {
                mlir::IRMapping mapper;
                mapper.map(mlir::cast<mlir::ViewLikeOpInterface>(currViewOp).getViewSource(), resValue);
                resValue = argBuilder.clone(*currViewOp, mapper)->getResult(0);
            }

            return resValue;
        };

        replaceArgs(func, getDeclaration);
    });
}

}  // namespace

//
// createConvertFuncArgsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertFuncArgsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertFuncArgsToDeclarationsPass>(log);
}

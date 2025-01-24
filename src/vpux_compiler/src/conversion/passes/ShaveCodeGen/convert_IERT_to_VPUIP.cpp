//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IERT/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>

using namespace vpux;

namespace {

//
// ConvertIERT2VPUIPPass
//

class ConvertIERT2VPUIPPass final : public ConvertIERT2VPUIPBase<ConvertIERT2VPUIPPass> {
public:
    explicit ConvertIERT2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    void processPackMemrefsAndExtendedCalls(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp);
};

void ConvertIERT2VPUIPPass::processPackMemrefsAndExtendedCalls(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp) {
    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp entryPointFuncOp;
    IE::CNNNetworkOp::getFromModule(moduleOp, netOp, entryPointFuncOp);

    vpux::VPUIP::createRuntimeKernelDefinition(moduleOp, _log, VPU::getArch(moduleOp));

    auto innerVpuSwModuleOp = moduleOp.lookupSymbol<mlir::ModuleOp>("VPU.SW");

    for (auto extCallOp : llvm::make_early_inc_range(entryPointFuncOp.getOps<vpux::IERT::ExtendedCallOp>())) {
        auto extCallOpPrevOp = mlir::dyn_cast<IERT::PackMemrefsOp>(extCallOp.getOperation()->getPrevNode());
        VPUX_THROW_UNLESS(extCallOpPrevOp != nullptr,
                          "The IERT.ExtendedCallOp has as predecessor an operation that is not an "
                          "IERT.PackMemrefs, or has no predecessor.");

        auto extCallOpPrevPrevOp =
                mlir::dyn_cast<mlir::memref::AllocOp>(extCallOp.getOperation()->getPrevNode()->getPrevNode());
        VPUX_THROW_UNLESS(extCallOpPrevPrevOp != nullptr,
                          "The IERT.ExtendedCallOp has as pre-predecessor an operation that is not a "
                          "memref.alloc.");

        auto llvmFuncOp =
                innerVpuSwModuleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(extCallOp.getCallee().getLeafReference().str());

        VPUX_THROW_WHEN(llvmFuncOp == nullptr,
                        "The IERT.ExtendedCall doesn't have the associated callee LLVMFuncOp in the VPU.SW module");

        mlir::OpBuilder builder(&ctx);

        // Set insertion point right before getNextNode(), so right after extCallOp
        builder.setInsertionPoint(extCallOp.getOperation()->getNextNode());

        // We build a SymbolRefAttr for the full name of llvmFuncOp
        mlir::FlatSymbolRefAttr builtinFlatFunc = mlir::SymbolRefAttr::get(&ctx, llvmFuncOp.getName().str());
        mlir::SymbolRefAttr builtinFuncSymbolRefAttr =
                mlir::SymbolRefAttr::get(&ctx, innerVpuSwModuleOp.getName().value(), {builtinFlatFunc});

        const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
        const auto typeInput =
                extCallOpPrevOp.getOperand(0).getType().cast<vpux::NDTypeInterface>().changeMemSpace(memSpaceCMX);
        auto cmxAllocOpInput = builder.create<mlir::memref::AllocOp>(extCallOpPrevOp.getOperand(0).getLoc(),
                                                                     typeInput.cast<mlir::MemRefType>());
        auto copyOpInput = builder.create<VPUIP::CopyOp>(extCallOpPrevOp->getLoc(), extCallOpPrevOp.getOperand(0),
                                                         cmxAllocOpInput.getMemref());

        /*
        We initially have this code:
            %0 = memref.alloc() // input (normally %arg0)
            %1 = memref.alloc() // output
            %2 = OP0 inputs(%1) outputs(%1) -> memref
        The generated memref.alloc() and VPUIP.Copy ops should be like this:
            %0 = memref.alloc() // input (normally %arg0)
            %1 = memref.alloc() // output
            %4 = memref.alloc() CMX // input
            %5 = memref.alloc() CMX // output
            %6 = COPY inputs(%0) outputs(%4)
            %2 = OP0 inputs(%6) outputs(%5) -> memref
            %7 = COPY inputs(%2) outputs(%1)
            Note that %2 is the result to be returned. But, we return %7.
              See below: .replaceAllUsesWith(copyOpOutput.getOperation());
        */

        // We now handle the output of the sw layer/kernel
        // The extCallOpPrevOp, an IERT.PackMemrefs, has 2 arguments.
        const auto typeOutput =
                extCallOpPrevOp.getOperand(1).getType().cast<vpux::NDTypeInterface>().changeMemSpace(memSpaceCMX);
        auto cmxAllocOpOutput = builder.create<mlir::memref::AllocOp>(extCallOpPrevOp.getOperand(1).getLoc(),
                                                                      typeOutput.cast<mlir::MemRefType>());

        llvm::SmallVector<mlir::Value> inputsSwKernelOp;
        // Normally extCallOpPrevOp has only 1 "output_buff" argument inherited from the sw layer (hence the -1) (see in
        // IERT/ops.td CosOp, HSwishOp; SoftMax excluding the last parameter, axisInd - but maybe axisInd is a constant
        // and can be embedded in the kernel) - TODO: figure out how we treat e.g. 3rd parameter for SoftMax
        inputsSwKernelOp.push_back(copyOpInput);

        auto swKernelOp = builder.create<VPUIP::SwKernelOp>(
                extCallOp.getLoc(),
                inputsSwKernelOp,              // Has to be: mlir::ValueRange inputs
                cmxAllocOpOutput.getResult(),  // Has to be: mlir::ValueRange output_buffs
                builtinFuncSymbolRefAttr,      // Has to be: mlir::SymbolRefAttr kernelFunction
                builder.getI64IntegerAttr(0)   // Has to be: mlir::IntegerAttr tileIndex
        );

        auto copyOpOutput = builder.create<VPUIP::CopyOp>(extCallOp.getLoc(), swKernelOp.getResult(0),
                                                          extCallOpPrevPrevOp.getMemref());

        // We create the body of swKernelOp.
        mlir::Block& blk3 = swKernelOp.getRegion().emplaceBlock();

        // We assume that extCallOpPrevOp has only 1 output argument (hence the -1).
        //   TODO E#67988 treat in the general case.
        for (std::size_t i = 0; i < extCallOpPrevOp.getNumOperands() - 1; i++) {
            blk3.addArgument(extCallOpPrevOp.getOperand(i).getType(), swKernelOp.getLoc());
        }

        blk3.addArgument(extCallOpPrevOp.getOperand(1).getType(),
                         swKernelOp.getLoc());  // TODO: Find a more generic way to do things right.

        builder.setInsertionPointToStart(&blk3);

        llvm::SmallVector<mlir::Value> operandsPackMemrefsNewOp;
        for (std::size_t i = 0; i < extCallOpPrevOp.getNumOperands(); i++) {
            operandsPackMemrefsNewOp.push_back(blk3.getArgument(i));
        }

        auto packMemrefNewOp = builder.create<IERT::PackMemrefsOp>(
                mlir::UnknownLoc::get(&ctx), vpux::IERT::PackedParamsType::get(&ctx), operandsPackMemrefsNewOp);

        builder.create<VPUIP::SwKernelRun>(mlir::UnknownLoc::get(&ctx),
                                           packMemrefNewOp.getResult(),  // Has to be: mlir::ValueRange args
                                           nullptr                       // Has to be: mlir::ArrayAttr attrs
        );

        extCallOp.replaceAllUsesWith(copyOpOutput.getOperation());
        extCallOp.erase();

        extCallOpPrevOp.erase();
    }
}

void ConvertIERT2VPUIPPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    processPackMemrefsAndExtendedCalls(ctx, module);
}

}  // namespace

//
// createConvertLayers2AffinePass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIERT2VPUIPPass(Logger log) {
    return std::make_unique<ConvertIERT2VPUIPPass>(log);
}

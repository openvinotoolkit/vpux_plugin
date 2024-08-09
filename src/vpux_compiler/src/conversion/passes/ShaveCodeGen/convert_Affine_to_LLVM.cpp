//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

#include <llvm/Support/TargetSelect.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

// TODO: E66812, it should be sufficient to have warnings disabled for 3-rd parties
// in CMake but it does not work for early versions of MSVC 2019
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

using namespace vpux;

namespace {

class ConvertAffine2LLVMPass final : public ConvertAffine2LLVMBase<ConvertAffine2LLVMPass> {
public:
    explicit ConvertAffine2LLVMPass(Logger log): _log(log) {
    }

private:
    void safeRunOnModule() final;
    void convertPackedParamsAndExtractParam(mlir::func::FuncOp funcOp);
    void handleSpecialCast(mlir::LLVM::LLVMFuncOp funcOp);

private:
    Logger _log;
};

// The RewritePattern below is taken from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Arith/Transforms/ExpandOps.cpp#L149
//   (referred from
//   https://discourse.llvm.org/t/support-for-lowering-to-llvm-for-the-standard-dialects-maxfop-and-minfop-operations/63588/3)
template <typename OpTy, mlir::arith::CmpFPredicate pred>
struct MaxMinFOpConverter : public mlir::OpRewritePattern<OpTy> {
public:
    using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(OpTy op, mlir::PatternRewriter& rewriter) const final {
        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();

        mlir::Location loc = op.getLoc();
        // If any operand is NaN, 'cmp' will be true (and 'select' returns 'lhs').
        static_assert(pred == mlir::arith::CmpFPredicate::UGT || pred == mlir::arith::CmpFPredicate::ULT,
                      "pred must be either UGT or ULT");
        mlir::Value cmp = rewriter.create<mlir::arith::CmpFOp>(loc, pred, lhs, rhs);
        mlir::Value select = rewriter.create<mlir::arith::SelectOp>(loc, cmp, lhs, rhs);

        // Handle the case where rhs is NaN: 'isNaN(rhs) ? rhs : select'.
        mlir::Value isNaN = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNO, rhs, rhs);
        rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(op, isNaN, rhs, select);
        return mlir::success();
    }
};

void ConvertAffine2LLVMPass::convertPackedParamsAndExtractParam(mlir::func::FuncOp funcOp) {
    const auto funcOpType = funcOp.getFunctionType();
    mlir::MLIRContext* ctx = funcOp.getContext();

    llvm::SmallVector<mlir::Type> newFuncArgTypes;

    // Create a struct type, corresponding to the struct below, which is
    //   the LLVM IR equivalent of an mlir::MemRef. More exactly,
    //   struct<(ptr<f16>, ptr<f16>, i64, array<1 x i64>, array<1 x i64>
    auto llvmI64Type = mlir::IntegerType::get(ctx, 64);
    llvm::SmallVector<mlir::Type, 5> fields;

    // generic opaque pointer type
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);

    fields.push_back(ptrType);
    fields.push_back(ptrType);
    fields.push_back(llvmI64Type);
    // We put the number of dimensions of the memref as the dimension of the array.
    auto arrayType = mlir::LLVM::LLVMArrayType::get(
            llvmI64Type, funcOp.getResultTypes()[0].cast<mlir::MemRefType>().getShape().size());
    fields.push_back(arrayType);
    fields.push_back(arrayType);
    mlir::Type memrefStructType = mlir::LLVM::LLVMStructType::getLiteral(ctx, fields);

    newFuncArgTypes.push_back(ptrType);
    mlir::FunctionType newFuncType =
            mlir::FunctionType::get(ctx, newFuncArgTypes, mlir::TypeRange(funcOpType.getResults()));

    funcOp.setType(newFuncType);

    mlir::OpBuilder builder(ctx);

    // We add a new argument, besides the IERT.PackedParams type argument to funcOp
    funcOp.getBlocks().front().addArgument(ptrType, funcOp.getLoc());

    int indexCounter = 0;

    for (auto epOp : llvm::make_early_inc_range(funcOp.getOps<IERT::ExtractParamOp>())) {
        builder.setInsertionPoint(epOp);

        auto ctIndex = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), builder.getI64Type(),
                                                              builder.getI64IntegerAttr(indexCounter));

        ++indexCounter;

        // The naming of the arguments of the constructor might be a bit confusing because resultType is the
        // now opaque pointer type and elementType is the type of the returned value.
        auto gepOp = builder.create<mlir::LLVM::GEPOp>(funcOp.getLoc(), ptrType, memrefStructType,
                                                       funcOp.getBlocks().front().getArgument(1),
                                                       SmallVector<mlir::LLVM::GEPArg>{ctIndex.getResult()});

        auto loadOp = builder.create<mlir::LLVM::LoadOp>(funcOp.getLoc(), memrefStructType, gepOp);

        auto specialCastOp = builder.create<IERT::SpecialCastOp>(funcOp.getLoc(), funcOp.getResultTypes().front(),
                                                                 loadOp.getOperation()->getResult(0));

        epOp.replaceAllUsesWith(&(*specialCastOp));

        epOp.erase();
    }

    // Only now we erase the IERT::PackedParams typed argument since we erased the operations using it just above
    funcOp.getBlocks().front().eraseArgument(0);
}

void ConvertAffine2LLVMPass::handleSpecialCast(mlir::LLVM::LLVMFuncOp funcOp) {
    for (auto scOp : llvm::make_early_inc_range(funcOp.getOps<IERT::SpecialCastOp>())) {
        // We assume that the LLVM converter will always generate a builtin.unrealized_conversion_cast
        // immediately after IERT.SpecialCast when converting to the LLVM dialect
        // We replace all uses of the builtin.unrealized_conversion_cast with
        //  the operands argument of the scOp.
        mlir::Operation* scOpNext = scOp.getOperation()->getNextNode();

        VPUX_THROW_UNLESS(mlir::isa<mlir::UnrealizedConversionCastOp>(scOpNext),
                          "The IERT.SpecialCastOp has as successor an operation that is not a "
                          "builtin.unrealized_conversion_cast.");

        scOpNext->replaceAllUsesWith(scOp.getInputs().getDefiningOp());

        scOpNext->erase();
        scOp.erase();
    }
}

void ConvertAffine2LLVMPass::safeRunOnModule() {
    auto& ctx = getContext();
    mlir::LLVMConversionTarget target(ctx);
    target.addLegalOp<mlir::ModuleOp>();

    target.addLegalOp<IE::CNNNetworkOp>();
    target.addLegalOp<IE::DataInfoOp>();
    target.addLegalOp<IE::ExecutorResourceOp>();
    target.addLegalOp<IE::MemoryResourceOp>();
    target.addLegalOp<VPUIP::CopyOp>();
    target.addLegalOp<IERT::ExtractParamOp>();
    target.addLegalOp<IERT::SpecialCastOp>();

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();

    mlir::LowerToLLVMOptions options(&ctx);
    mlir::LLVMTypeConverter typeConverter(&ctx, options);

    // static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};
    auto vpuSwmoduleOp = module.lookupSymbol<mlir::ModuleOp>("VPU.SW");

    // We call convertPackedParamsAndExtractParam() for each function requiring it
    for (auto funcOp : vpuSwmoduleOp.getOperation()->getRegion(0).getOps<mlir::func::FuncOp>()) {
        convertPackedParamsAndExtractParam(funcOp);
    }

    for (auto funcOp :
         llvm::make_early_inc_range(vpuSwmoduleOp.getOperation()->getRegion(0).getOps<mlir::func::FuncOp>())) {
        mlir::RewritePatternSet patterns(&ctx);

        mlir::populateAffineToStdConversionPatterns(patterns);
        mlir::populateSCFToControlFlowConversionPatterns(patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

        patterns.add<MaxMinFOpConverter<mlir::arith::MaximumFOp, mlir::arith::CmpFPredicate::UGT>,
                     MaxMinFOpConverter<mlir::arith::MinimumFOp, mlir::arith::CmpFPredicate::ULT>>(&ctx);

        if (failed(applyFullConversion(funcOp, target, std::move(patterns))))
            signalPassFailure();
    }

    for (auto funcOp : vpuSwmoduleOp.getOperation()->getRegion(0).getOps<mlir::LLVM::LLVMFuncOp>()) {
        handleSpecialCast(funcOp);
    }
}

}  // namespace

//
// createConvertAffine2LLVMPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertAffine2LLVMPass(Logger log) {
    return std::make_unique<ConvertAffine2LLVMPass>(log);
}

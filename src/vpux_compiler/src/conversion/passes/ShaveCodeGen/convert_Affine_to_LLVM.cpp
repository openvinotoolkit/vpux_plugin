//
// Copyright (C) 2022-2024 Intel Corporation.
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
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
    void convertPackedParamsAndExtractParamOp(mlir::func::FuncOp funcOp);
    void handleSpecialCast(mlir::LLVM::LLVMFuncOp funcOp);

private:
    Logger _log;
};

// The RewritePattern below is taken from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L149
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

// We convert each ExtractParam ops from funcOp to llvm.getelementptr, (llvm.bitcast), llvm.load ops.
//   And at the end we erase the argument of type PackedParams from the signature of the function.
void ConvertAffine2LLVMPass::convertPackedParamsAndExtractParamOp(mlir::func::FuncOp funcOp) {
    const auto funcOpType = funcOp.getFunctionType();
    mlir::MLIRContext* ctx = funcOp.getContext();
    mlir::OpBuilder builder(ctx);

    std::vector<mlir::LLVM::LLVMPointerType> ptrMemrefStructTypeVec;

    int indexCounter = 0;
    auto llvmI32Type = mlir::IntegerType::get(ctx, 32);
    auto ptrLLVMType = mlir::LLVM::LLVMPointerType::get(ctx);

    for (auto epOp : llvm::make_early_inc_range(funcOp.getOps<IERT::ExtractParamOp>())) {
        llvm::SmallVector<mlir::Type> newFuncArgTypes;

        // Create a struct type, corresponding to the struct below, which is
        //   the LLVM dialect/IR equivalent of an mlir::MemRef. For example,
        //   struct<(ptr<f16>, ptr<f16>, i32, array<1 x i32>, array<1 x i32>)>
        llvm::SmallVector<mlir::Type, 5> structFields;

        // typed pointers got deprecated --> pointerType is now always an opaque pointer
        structFields.push_back(ptrLLVMType);
        structFields.push_back(ptrLLVMType);
        structFields.push_back(llvmI32Type);
        // We put the number of dimensions of the memref as the dimension of the array.
        auto arrayType = mlir::LLVM::LLVMArrayType::get(
                llvmI32Type, epOp.getResult().getType().cast<mlir::MemRefType>().getShape().size());
        structFields.push_back(arrayType);
        structFields.push_back(arrayType);
        auto memrefStructType = mlir::LLVM::LLVMStructType::getLiteral(ctx, structFields);

        auto ptrMemrefStructType = mlir::LLVM::LLVMPointerType::get(ctx);
        ptrMemrefStructTypeVec.push_back(ptrMemrefStructType);

        if (indexCounter == 0) {
            // We assume all the params of the sw layer kernel have the same type

            newFuncArgTypes.push_back(ptrMemrefStructType);
            mlir::FunctionType newFuncType =
                    mlir::FunctionType::get(ctx, newFuncArgTypes, mlir::TypeRange(funcOpType.getResults()));

            funcOp.setType(newFuncType);

            // We add a new argument, besides the IERT.PackedParams type argument to funcOp
            // This argument is of type of the 1st ExtractParam of the kernel.
            //   (We assume all ExtractParams of the kernel have the same type.)
            // LE: This logic needs to be extended - now the above mentioned assumtion constrains us to having all the
            // kernel params being of the same rank E#139446
            funcOp.getBlocks().front().addArgument(ptrMemrefStructType, funcOp.getLoc());

            // We can't erase yet the 0-index argument (IERT::PackedParams typed) from funcOp.getBlocks().front()
            // because it's used by the ExtractParam ops.
        }

        builder.setInsertionPoint(epOp);

        auto ctIndex = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), builder.getI64Type(),
                                                              builder.getI64IntegerAttr(indexCounter));

        auto gepOp = builder.create<mlir::LLVM::GEPOp>(
                funcOp.getLoc(),
                ptrMemrefStructType,  // This has to be: Type resultType
                memrefStructType,     // This has to be: Type elementType
                // We use getArgument(1) because we added earlier this argument with addArgument().
                //   And we can't erase yet the 0-index argument because it's used by the ExtractParam ops.
                funcOp.getBlocks().front().getArgument(1),  // This has to be: Value basePtr
                ctIndex.getResult()                         // This has to be: ValueRange indices
        );
        auto loadOp = builder.create<mlir::LLVM::LoadOp>(funcOp.getLoc(), gepOp.getElemType(),
                                                         gepOp.getResult()  // This has to be: Value addr
        );
        auto specialCastOp =
                builder.create<IERT::SpecialCastOp>(funcOp.getLoc(),
                                                    epOp.getResult().getType(),  // This has to be: Type resultType
                                                    loadOp.getResult()  // This has to be: Value operandToConvert
                );

        ++indexCounter;

        epOp.replaceAllUsesWith(specialCastOp.getOperation());
        epOp.erase();
    }
    // Only now we erase the IERT::PackedParams typed argument since we erased the operations using it above
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

        scOpNext->replaceAllUsesWith(scOp.getOperandToConvert().getDefiningOp());

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
    options.overrideIndexBitwidth(32);
    mlir::LLVMTypeConverter typeConverter(&ctx, options);

    auto vpuSwmoduleOp = module.lookupSymbol<mlir::ModuleOp>("VPU.SW");

    // We call convertPackedParamsAndExtractParam() for each function requiring it
    for (auto funcOp : vpuSwmoduleOp.getOperation()->getRegion(0).getOps<mlir::func::FuncOp>()) {
        convertPackedParamsAndExtractParamOp(funcOp);
    }

    for (auto funcOp :
         llvm::make_early_inc_range(vpuSwmoduleOp.getOperation()->getRegion(0).getOps<mlir::func::FuncOp>())) {
        // Executing the population of patterns for each loop iteration since
        //   the patterns variable is altered by applyFullConversion(...std::move(patterns)).
        // Note: The conversion is inspired from Toy example chapter 6 (https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/).
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

        if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
            signalPassFailure();
        }
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

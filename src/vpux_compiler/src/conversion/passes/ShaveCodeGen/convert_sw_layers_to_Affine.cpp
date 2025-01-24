//
// Copyright (C) 2022-2023 Intel Corporation.
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
#include <mlir/IR/IRMapping.h>
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

/// Insert an allocation and deallocation for the given MemRefType.
static mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc, mlir::PatternRewriter& rewriter,
                                         bool isUnit = false, bool createsDealloc = true) {
    mlir::memref::AllocOp alloc;
    if (isUnit) {
        alloc = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({1}, type.getElementType()));
    } else {
        alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);
    }

    // Make sure to allocate at the beginning of the block.
    auto* parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    if (createsDealloc == true) {
        // Make sure to deallocate this alloc at the end of the block.
        auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
        dealloc->moveBefore(&parentBlock->back());
    }
    return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, a range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = llvm::function_ref<mlir::Value(mlir::OpBuilder& rewriter, mlir::ValueRange memRefOperands,
                                                       mlir::ValueRange loopIvs)>;

static void lowerOpToLoops(mlir::Operation* op, mlir::ValueRange operands, mlir::PatternRewriter& rewriter,
                           LoopIterationFn processIteration) {
    auto memRefType = (*op->result_type_begin()).cast<mlir::MemRefType>();
    auto loc = op->getLoc();

    // Create a nest of affine loops, with one loop per dimension of the shape.
    // The buildAffineLoopNest function takes a callback that is used to construct
    // the body of the innermost loop given a builder, a location and a range of
    // loop induction variables.
    SmallVector<int64_t, 4> lowerBounds(memRefType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(memRefType.getRank(), /*Value=*/1);
    mlir::affine::buildAffineLoopNest(rewriter, loc, lowerBounds, memRefType.getShape(), steps,
                                      // The ivs parameter (and the other formal arguments) are being assigned by
                                      // buildAffineLoopNest() based on the parameters it receives itself
                                      [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
                                          // Call the processing function with the rewriter, the memref operands,
                                          // and the loop induction variables. This function will return the value
                                          // to store at the current index.
                                          mlir::Value valueToStore = processIteration(nestedBuilder, operands, ivs);

                                          VPUX_THROW_UNLESS(operands.size() >= 2, "Need to have 2 operands");

                                          nestedBuilder.create<mlir::affine::AffineStoreOp>(loc, valueToStore,
                                                                                            operands[1], ivs);
                                      });

    rewriter.replaceOp(op, operands[1]);  // We return the container in which we stored the computed values.
                                          // (operands[1] is the output buffer)
}

struct UnaryOpLoweringCos : public mlir::ConversionPattern {
    UnaryOpLoweringCos(mlir::MLIRContext* ctx): mlir::ConversionPattern(IERT::CosOp::getOperationName(), 1, ctx) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](mlir::OpBuilder& builder, mlir::ValueRange memRefOperands, mlir::ValueRange loopIvs) {
                           // Generate an adaptor for the remapped operands of the UnaryOp. This
                           // allows for using the nice named accessors that are generated by the
                           // ODS.
                           IERT::CosOp::Adaptor unaryAdaptor(memRefOperands);

                           // Generate load for the element of 'input' at the inner loop.
                           auto loadedOpnd =
                                   builder.create<mlir::affine::AffineLoadOp>(loc, unaryAdaptor.getInput(), loopIvs);

                           auto cosOp = builder.create<mlir::math::CosOp>(loc, loadedOpnd);

                           return cosOp;
                       });

        return mlir::success();
    }
};

struct UnaryOpLoweringHSwish : public mlir::ConversionPattern {
    UnaryOpLoweringHSwish(mlir::MLIRContext* ctx): mlir::ConversionPattern(IERT::HSwishOp::getOperationName(), 1, ctx) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(
                op, operands, rewriter,
                [loc, op](mlir::OpBuilder& builder, mlir::ValueRange memRefOperands, mlir::ValueRange loopIvs) {
                    // Generate an adaptor for the remapped operands of the UnaryOp. This
                    // allows for using the nice named accessors that are generated by the
                    // ODS.
                    IERT::HSwishOp::Adaptor unaryAdaptor(memRefOperands);

                    // Generate load for the element.
                    auto loadedOpnd = builder.create<mlir::affine::AffineLoadOp>(loc, unaryAdaptor.getInput(), loopIvs);

                    // IMPORTANT: HSwish(x) = x * min(max(x+3, 0), 6) / 6
                    float f;
                    mlir::MLIRContext* ctx = op->getContext();

                    auto memRefOperands0Type = unaryAdaptor.getInput().getType();

                    auto memRefOperands0TypeMemref = memRefOperands0Type.dyn_cast_or_null<mlir::MemRefType>();
                    VPUX_THROW_UNLESS(memRefOperands0TypeMemref != nullptr, "Abnormal situation encountered");

                    mlir::arith::ConstantFloatOp valOp1;
                    bool typeIsF32 = memRefOperands0TypeMemref.getElementType().isF32();
                    bool typeIsF16 = memRefOperands0TypeMemref.getElementType().isF16();

                    if (typeIsF32) {
                        f = 3.0;
                        valOp1 = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(f),
                                                                              mlir::FloatType::getF32(ctx));
                    } else if (typeIsF16) {
                        llvm::APInt data(16, 0x4200);  // 0x4200 is value for 3.0 for f16

                        llvm::APFloat value = llvm::APFloat(llvm::APFloat::IEEEhalf(), data);

                        valOp1 = builder.create<mlir::arith::ConstantFloatOp>(loc, value, mlir::FloatType::getF16(ctx));
                    }

                    auto addFOp = builder.create<mlir::arith::AddFOp>(loc, loadedOpnd, valOp1->getResult(0));

                    mlir::arith::ConstantFloatOp valOp2;
                    if (typeIsF32) {
                        f = 0.0;
                        valOp2 = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(f),
                                                                              mlir::FloatType::getF32(ctx));
                    } else if (typeIsF16) {
                        llvm::APInt data(16, 0x0);
                        llvm::APFloat value = llvm::APFloat(llvm::APFloat::IEEEhalf(), data);
                        valOp2 = builder.create<mlir::arith::ConstantFloatOp>(loc, value, mlir::FloatType::getF16(ctx));
                    }

                    auto maxFOp = builder.create<mlir::arith::MaximumFOp>(loc, addFOp, valOp2->getResult(0));

                    mlir::arith::ConstantFloatOp valOp3;
                    if (typeIsF32) {
                        f = 6.0;
                        valOp3 = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(f),
                                                                              mlir::FloatType::getF32(ctx));
                    } else if (typeIsF16) {
                        llvm::APInt data(16, 0x4600);  // 0x4600 is value for 6.0 for f16
                        llvm::APFloat value = llvm::APFloat(llvm::APFloat::IEEEhalf(), data);
                        valOp3 = builder.create<mlir::arith::ConstantFloatOp>(loc, value, mlir::FloatType::getF16(ctx));
                    }

                    auto minFOp = builder.create<mlir::arith::MinimumFOp>(loc, maxFOp, valOp3->getResult(0));

                    auto divFOp = builder.create<mlir::arith::DivFOp>(loc, minFOp, valOp3->getResult(0));

                    auto mulFOp = builder.create<mlir::arith::MulFOp>(loc, divFOp, loadedOpnd);

                    return mulFOp;
                });

        return mlir::success();
    }
};

struct UnaryOpLoweringSoftMax : public mlir::ConversionPattern {
    UnaryOpLoweringSoftMax(mlir::MLIRContext* ctx)
            : mlir::ConversionPattern(IERT::SoftMaxOp::getOperationName(), 1, ctx) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // SoftMax is defined for example at https://slaystudy.com/implementation-of-softmax-activation-function-in-c-c/

        auto loc = op->getLoc();

        auto memRefType = (*op->result_type_begin()).cast<mlir::MemRefType>();

        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter, true);

        mlir::Value zeroIndex2 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto zeroConst =
                rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(memRefType.getElementType(), 0));
        rewriter.create<mlir::memref::StoreOp>(loc, zeroConst, alloc, zeroIndex2);

        SmallVector<int64_t, 4> lowerBounds(memRefType.getRank(), /*Value=*/0);
        SmallVector<int64_t, 4> steps(memRefType.getRank(), /*Value=*/1);
        mlir::affine::buildAffineLoopNest(
                rewriter, loc, lowerBounds, memRefType.getShape(), steps,
                [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
                    // Call the processing function with the rewriter, the memref operands,
                    // and the loop induction variables. This function will return the value
                    // to store at the current index.

                    VPUX_THROW_UNLESS(operands.size() >= 2, "Need to have 2 operands");

                    auto loadedVal = nestedBuilder.create<mlir::affine::AffineLoadOp>(loc, operands[0], ivs);
                    auto expTerm = nestedBuilder.create<mlir::math::ExpOp>(loc, loadedVal);

                    mlir::Value zeroIndex = nestedBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);

                    auto loadedVal2 = nestedBuilder.create<mlir::affine::AffineLoadOp>(loc, alloc, zeroIndex);

                    auto addOp = nestedBuilder.create<mlir::arith::AddFOp>(loc, loadedVal2, expTerm);

                    nestedBuilder.create<mlir::affine::AffineStoreOp>(loc, addOp, alloc, zeroIndex);
                });

        lowerOpToLoops(
                op, operands, rewriter,
                [alloc, loc](mlir::OpBuilder& builder, mlir::ValueRange memRefOperands, mlir::ValueRange loopIvs) {
                    // Generate an adaptor for the remapped operands of the UnaryOp. This
                    // allows for using the nice named accessors that are generated by the
                    // ODS.
                    IERT::SoftMaxOp::Adaptor unaryAdaptor(memRefOperands);

                    auto loadedOpnd = builder.create<mlir::affine::AffineLoadOp>(loc, unaryAdaptor.getInput(), loopIvs);

                    // Following
                    // https://slaystudy.com/implementation-of-softmax-activation-function-in-c-c/,
                    //   softmax[i] = exp(input[i]) / sum, where sum = \sum_k exp(input[k])

                    auto expOp = builder.create<mlir::math::ExpOp>(loc, loadedOpnd);

                    mlir::Value zeroIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
                    auto acc = builder.create<mlir::affine::AffineLoadOp>(loc, alloc, zeroIndex);

                    auto binOp1 = builder.create<mlir::arith::DivFOp>(loc, expOp, acc->getResult(0));

                    return binOp1;
                });

        return mlir::success();
    }
};

struct UnaryOpLoweringGenericReshape : public mlir::ConversionPattern {
    UnaryOpLoweringGenericReshape(mlir::MLIRContext* ctx)
            : mlir::ConversionPattern(IERT::GenericReshapeOp::getOperationName(), 1, ctx) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();

        auto memRefType = (*op->result_type_begin()).cast<mlir::MemRefType>();

        IERT::GenericReshapeOp grOp;

        // Create a nest of affine loops, with one loop per dimension of the shape.
        // The buildAffineLoopNest function takes a callback that is used to construct
        // the body of the innermost loop given a builder, a location and a range of
        // loop induction variables.
        SmallVector<int64_t, 4> lowerBounds(memRefType.getRank(), /*Value=*/0);
        SmallVector<int64_t, 4> steps(memRefType.getRank(), /*Value=*/1);
        mlir::affine::buildAffineLoopNest(
                rewriter, loc, lowerBounds, memRefType.getShape(), steps,
                // The ivs parameter (and the other formal arguments) are being assigned by
                // buildAffineLoopNest() based on the parameters it receives itself
                [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
                    // Generate an adaptor for the remapped operands of the UnaryOp. This allows
                    // for using the nice named accessors that are generated by the ODS.
                    IERT::GenericReshapeOp::Adaptor unaryAdaptor(operands);

                    grOp = mlir::dyn_cast<IERT::GenericReshapeOp>(op);
                    VPUX_THROW_UNLESS(grOp != nullptr, "op is not of type IERT::GenericReshapeOp");
                    std::size_t sizeIvsNew = grOp.getInput().getType().cast<mlir::MemRefType>().getShape().size();

                    llvm::SmallVector<mlir::Value> ivsNew;
                    for (size_t i = 0; i < sizeIvsNew - 1; i++) {
                        ivsNew.push_back(ivs[0]);
                    }
                    ivsNew.push_back(ivs[ivs.size() - 1]);

                    // Generate load for the element of 'input' at the inner loop.
                    mlir::affine::AffineLoadOp loadOp =
                            nestedBuilder.create<mlir::affine::AffineLoadOp>(loc, unaryAdaptor.getInput(), ivsNew);

                    mlir::Value valueToStore = loadOp;

                    VPUX_THROW_UNLESS(operands.size() >= 1, "Need to have at least 1 operand");

                    nestedBuilder.create<mlir::affine::AffineStoreOp>(loc, valueToStore, grOp.getOutputBuff(), ivs);
                });

        rewriter.replaceOp(op, grOp.getOutputBuff());

        return mlir::success();
    }
};

//
// ConvertSWLayers2AffinePass
//

class ConvertSWLayers2AffinePass final : public ConvertSWLayers2AffineBase<ConvertSWLayers2AffinePass> {
public:
    explicit ConvertSWLayers2AffinePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    template <class TemplateTypeOp>
    void processSoftwareLayer(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp, mlir::func::FuncOp funcOp);
    void processAllSoftwareLayers(mlir::MLIRContext& ctx, mlir::ModuleOp module);

    std::map<std::string, int> counterSwLayers;
};

template <class TemplateTypeOp>
void ConvertSWLayers2AffinePass::processSoftwareLayer(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp,
                                                      mlir::func::FuncOp entryPointFuncOp) {
    OpBuilderLogger builderLog(_log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};
    auto vpuSwmoduleOp = moduleOp.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // Creating the @VPU.SW submodule if it is not yet created
    if (!vpuSwmoduleOp) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody(), &builderLog);
        vpuSwmoduleOp = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(&ctx), vpuSwModuleName);
    }

    for (auto crtOp : llvm::make_early_inc_range(entryPointFuncOp.getBody().getOps<TemplateTypeOp>())) {
        std::string opName = crtOp->getName().stripDialect().str();

        if (counterSwLayers.find(opName) == counterSwLayers.end()) {
            counterSwLayers[opName] = 0;
        }

        std::string newFuncName = "generated_" + opName + std::to_string(counterSwLayers[opName]);

        mlir::Type packedParamsType = vpux::IERT::PackedParamsType::get(&ctx);

        llvm::SmallVector<mlir::Type> newFuncArgTypes;
        newFuncArgTypes.push_back(packedParamsType);

        mlir::OpBuilder bld(&ctx);
        bld.setInsertionPointToStart(vpuSwmoduleOp.getBody(0));

        mlir::FunctionType newFuncType = mlir::FunctionType::get(&ctx, newFuncArgTypes, mlir::TypeRange());

        auto newFuncOp =
                bld.create<mlir::func::FuncOp>(entryPointFuncOp.getLoc(), llvm::StringRef(newFuncName), newFuncType);

        // Adding a body to the function
        mlir::Block* newFuncOpBody = newFuncOp.addEntryBlock();
        bld.setInsertionPointToEnd(newFuncOpBody);

        mlir::Type type32Attr = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);

        std::vector<vpux::IERT::ExtractParamOp> newExtractParamOp;
        for (unsigned int i = 0; i < crtOp.getOperation()->getNumOperands(); i++) {
            newExtractParamOp.push_back(
                    bld.create<IERT::ExtractParamOp>(newFuncOp.getLoc(), crtOp.getOperation()->getOperand(i).getType(),
                                                     newFuncOp.getArgument(0), mlir::IntegerAttr::get(type32Attr, i)));
        }

        mlir::IRMapping mapper;

        // We map the old values of crtOp to the results of the newExtractParamOp elements
        for (unsigned int i = 0; i < crtOp.getOperation()->getNumOperands(); i++) {
            mapper.map(crtOp.getOperation()->getOperand(i), newExtractParamOp[i].getResult());
        }
        bld.clone(*(crtOp.getOperation()), mapper);
        bld.create<mlir::func::ReturnOp>(newFuncOp.getLoc(), mlir::ValueRange());

        // We now insert in the entry point function, from crtOp
        bld.setInsertionPoint(crtOp.getOperation());

        auto newPackMemrefsOp = bld.create<IERT::PackMemrefsOp>(entryPointFuncOp.getLoc(), packedParamsType,
                                                                mlir::ValueRange(crtOp.getOperation()->getOperands()));

        mlir::SymbolRefAttr symRefAttr = mlir::SymbolRefAttr::get(
                mlir::StringAttr::get(&ctx, "VPU.SW"), llvm::ArrayRef<mlir::FlatSymbolRefAttr>(mlir::SymbolRefAttr::get(
                                                               mlir::StringAttr::get(&ctx, newFuncName))));

        auto newFuncCallOp = bld.create<IERT::ExtendedCallOp>(entryPointFuncOp.getLoc(), symRefAttr,
                                                              mlir::TypeRange(crtOp.getResult().getType()),
                                                              mlir::ValueRange(newPackMemrefsOp.getResult()));
        crtOp.replaceAllUsesWith(newFuncCallOp);
        crtOp.erase();

        counterSwLayers[opName]++;
    }
}

void ConvertSWLayers2AffinePass::processAllSoftwareLayers(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp) {
    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp entryPointFuncOp;
    IE::CNNNetworkOp::getFromModule(moduleOp, netOp, entryPointFuncOp);

    processSoftwareLayer<IERT::CosOp>(ctx, moduleOp, entryPointFuncOp);
    processSoftwareLayer<IERT::HSwishOp>(ctx, moduleOp, entryPointFuncOp);
    processSoftwareLayer<IERT::SoftMaxOp>(ctx, moduleOp, entryPointFuncOp);
    processSoftwareLayer<IERT::GenericReshapeOp>(ctx, moduleOp, entryPointFuncOp);
}

void ConvertSWLayers2AffinePass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    processAllSoftwareLayers(ctx, module);

    mlir::ConversionTarget target(ctx);

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `Arith`, `MemRef`, and `Func` dialects.
    target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                           mlir::math::MathDialect, mlir::func::FuncDialect>();

    // To avoid getting strange legalization errors with operations such as IERT::CosOp, IERT::AsinOp, etc
    target.addIllegalDialect<vpux::IERT::IERTDialect>();

    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<VPUIP::CopyOp>();
    target.addLegalOp<IERT::ExtendedCallOp>();
    target.addLegalOp<IERT::PackMemrefsOp>();
    target.addLegalOp<IERT::ExtractParamOp>();

    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<UnaryOpLoweringCos>(&ctx);
    patterns.add<UnaryOpLoweringHSwish>(&ctx);
    patterns.add<UnaryOpLoweringSoftMax>(&ctx);
    patterns.add<UnaryOpLoweringGenericReshape>(&ctx);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2AffinePass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2AffinePass(Logger log) {
    return std::make_unique<ConvertSWLayers2AffinePass>(log);
}

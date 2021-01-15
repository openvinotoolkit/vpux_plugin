//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <precision_utils.h>

using namespace vpux;

namespace {

//
// ConvertPrecisionToFP16Pass
//

class ConvertPrecisionToFP16Pass final : public IE::ConvertPrecisionToFP16Base<ConvertPrecisionToFP16Pass> {
public:
    explicit ConvertPrecisionToFP16Pass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnOperation() final;

public:
    class FuncOpConverter;
    class GenericOpConverter;

public:
    static const mlir::PatternBenefit genericBenefit;
    static const mlir::PatternBenefit specificBenefit;

private:
    void passBody();

private:
    Logger _log;
};

const mlir::PatternBenefit ConvertPrecisionToFP16Pass::genericBenefit(1);
const mlir::PatternBenefit ConvertPrecisionToFP16Pass::specificBenefit(2);

void ConvertPrecisionToFP16Pass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// FuncOpConverter
//

class ConvertPrecisionToFP16Pass::FuncOpConverter final : public mlir::OpConversionPattern<mlir::FuncOp> {
public:
    FuncOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::FuncOp>(typeConverter, ctx, specificBenefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::FuncOp funcOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertPrecisionToFP16Pass::FuncOpConverter::matchAndRewrite(
        mlir::FuncOp funcOp, ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter& rewriter) const {
    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    return rewriteFuncPrototype(funcOp, *converter, rewriter, _log);
}

//
// GenericOpConverter
//

class ConvertPrecisionToFP16Pass::GenericOpConverter final : public mlir::ConversionPattern {
public:
    GenericOpConverter(mlir::TypeConverter& typeConverter, Logger log)
            : mlir::ConversionPattern(genericBenefit, typeConverter, MatchAnyOpTypeTag{}), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertPrecisionToFP16Pass::GenericOpConverter::matchAndRewrite(
        mlir::Operation* origOp, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", *origOp);

    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(converter->convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// passBody
//

void ConvertPrecisionToFP16Pass::passBody() {
    auto& ctx = getContext();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::RankedTensorType tensor) {
        if (tensor.getElementType().isF32()) {
            return mlir::RankedTensorType::get(tensor.getShape(), mlir::Float16Type::get(tensor.getContext()));
        } else {
            return tensor;
        }
    });
    typeConverter.addSourceMaterialization([](mlir::OpBuilder& builder, mlir::RankedTensorType type,
                                              mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.createOrFold<IE::ConvertOp>(loc, inputs[0], mlir::TypeAttr::get(type.getElementType()));
    });
    typeConverter.addTargetMaterialization([](mlir::OpBuilder& builder, mlir::RankedTensorType type,
                                              mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.createOrFold<IE::ConvertOp>(loc, inputs[0], mlir::TypeAttr::get(type.getElementType()));
    });

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addLegalOp<IE::ConvertOp>();
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getType()) && typeConverter.isLegal(&funcOp.getBody());
    });

    mlir::OwningRewritePatternList patterns;
    patterns.insert<FuncOpConverter>(typeConverter, &ctx, _log.nest());
    patterns.insert<GenericOpConverter>(typeConverter, _log.nest());
    IE::ConvertOp::getCanonicalizationPatterns(patterns, &ctx);

    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module.getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertPrecisionToFP16Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPrecisionToFP16Pass(Logger log) {
    return std::make_unique<ConvertPrecisionToFP16Pass>(log);
}

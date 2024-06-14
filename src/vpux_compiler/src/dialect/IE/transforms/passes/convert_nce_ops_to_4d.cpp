//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

mlir::Value extendTensor(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input) {
    if (!input) {
        return nullptr;
    }

    // Extend shape with Height = 1. [N C W] -> [N C 1 W]
    const auto shape = input.getType().cast<vpux::NDTypeInterface>().getShape();

    auto newShape = to_small_vector(shape);
    newShape.insert(newShape.end() - 1, 1);

    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    return rewriter.createOrFold<IE::ReshapeOp>(loc, input, nullptr, false, newShapeAttr);
}

mlir::ArrayAttr append(mlir::MLIRContext* context, mlir::ArrayAttr attr, int64_t value) {
    auto vector = parseIntArrayAttr<int64_t>(attr);
    vector.insert(vector.begin(), value);
    return getIntArrayAttr(context, vector);
}

mlir::Value composeTensorOnCD(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input) {
    if (!input) {
        return nullptr;
    }
    const auto shape = input.getType().cast<vpux::NDTypeInterface>().getShape();
    SmallVector<int64_t> newShape{shape[Dims5D::Act::N], shape[Dims5D::Act::C] * shape[Dims5D::Act::D],
                                  shape[Dims5D::Act::H], shape[Dims5D::Act::W]};
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    return rewriter.createOrFold<IE::ReshapeOp>(loc, input, nullptr, false, newShapeAttr);
}

mlir::ArrayAttr eraseAttrBegin(mlir::MLIRContext* context, mlir::ArrayAttr attr) {
    auto vector = parseIntArrayAttr<int64_t>(attr);
    vector.erase(vector.begin());
    return getIntArrayAttr(context, vector);
}

//
// ConvGeneralExpansion
//
template <class ConcreteOp>
class ConvGeneralExpansion final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvGeneralExpansion(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ConvGeneralExpansion<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Convert NCE to 4D for '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto* ctx = origOp->getContext();

    const auto newInput = extendTensor(rewriter, origOp->getLoc(), origOp->getOperand(0));
    const auto newFilter = extendTensor(rewriter, origOp->getLoc(), origOp.getFilter());

    const auto newStrides = append(ctx, origOp.getStrides(), 1);
    const auto newPadsBegin = append(ctx, origOp.getPadsBegin(), 0);
    const auto newPadsEnd = append(ctx, origOp.getPadsEnd(), 0);
    const auto newDilations = append(ctx, origOp.getDilations(), 1);

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector<mlir::Value>{newInput, newFilter});
    auto* newConvOp = rewriter.clone(*origOp.getOperation(), mapper);

    VPUX_THROW_UNLESS(newConvOp->hasAttr("pads_begin") && newConvOp->hasAttr("pads_end") &&
                              newConvOp->hasAttr("strides") && newConvOp->hasAttr("dilations"),
                      "Cannot get all attributions");
    newConvOp->setAttr("pads_begin", newPadsBegin);
    newConvOp->setAttr("pads_end", newPadsEnd);
    newConvOp->setAttr("strides", newStrides);
    newConvOp->setAttr("dilations", newDilations);

    if (auto transposedConv = mlir::dyn_cast<IE::TransposedConvolutionOp>(origOp.getOperation())) {
        const auto newOutputPadding = append(ctx, transposedConv.getOutputPadding(), 0);
        newConvOp->setAttr("output_padding", newOutputPadding);
    }

    vpux::inferReturnTypes(newConvOp, vpux::InferShapedTypeMode::ALL);

    const auto outputType = origOp.getOutput().getType().template dyn_cast<vpux::NDTypeInterface>();
    const auto outputShapeAttr = getIntArrayAttr(ctx, outputType.getShape());
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConvOp->getResult(0), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 4D '{0}'", origOp->getName());
    return mlir::success();
}

//
// PoolingGeneralExpansion
//
template <class ConcreteOp>
class PoolingGeneralExpansion final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    PoolingGeneralExpansion(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult PoolingGeneralExpansion<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Convert NCE to 4D for '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto* ctx = origOp->getContext();

    auto isInputShape3D = getShape(origOp.getInput()).size() == 3;
    const auto newInput = isInputShape3D ? extendTensor(rewriter, origOp->getLoc(), origOp.getInput())
                                         : composeTensorOnCD(rewriter, origOp->getLoc(), origOp->getOperand(0));
    if (newInput == nullptr) {
        _log.trace("Generate new Input failed at '{0}' '", origOp->getLoc());
        return mlir::failure();
    }
    const auto newKernelSize =
            isInputShape3D ? append(ctx, origOp.getKernelSize(), 1) : eraseAttrBegin(ctx, origOp.getKernelSize());
    const auto newStrides =
            isInputShape3D ? append(ctx, origOp.getStrides(), 1) : eraseAttrBegin(ctx, origOp.getStrides());
    const auto newPadsBegin =
            isInputShape3D ? append(ctx, origOp.getPadsBegin(), 0) : eraseAttrBegin(ctx, origOp.getPadsBegin());
    const auto newPadsEnd =
            isInputShape3D ? append(ctx, origOp.getPadsEnd(), 0) : eraseAttrBegin(ctx, origOp.getPadsEnd());

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector<mlir::Value>{newInput});
    auto* newPoolingOp = rewriter.clone(*origOp.getOperation(), mapper);

    VPUX_THROW_UNLESS(newPoolingOp->hasAttr("pads_begin") && newPoolingOp->hasAttr("pads_end") &&
                              newPoolingOp->hasAttr("strides") && newPoolingOp->hasAttr("kernel_size"),
                      "Cannot get all attributions");
    newPoolingOp->setAttr("pads_begin", newPadsBegin);
    newPoolingOp->setAttr("pads_end", newPadsEnd);
    newPoolingOp->setAttr("strides", newStrides);
    newPoolingOp->setAttr("kernel_size", newKernelSize);
    vpux::inferReturnTypes(newPoolingOp, vpux::InferShapedTypeMode::ALL);

    const auto outputType = origOp.getOutput().getType().template dyn_cast<vpux::NDTypeInterface>();
    const auto outputShapeAttr = getIntArrayAttr(ctx, outputType.getShape());
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newPoolingOp->getResult(0), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 4D '{0}'", origOp->getName());
    return mlir::success();
}

template <class ConcreteOp>
bool isLegalPoolNceOp(ConcreteOp pool) {
    const auto inputShape = getShape(pool.getInput());
    if (inputShape.size() == 3) {
        return false;
    }
    if (inputShape.size() == 5) {
        auto attrDValueEqualTo = [](mlir::ArrayRef<int64_t> attr, int64_t value) {
            if (attr.size() != 3) {
                return false;
            }
            return attr[0] == value;
        };

        const auto kernelShape = Shape(parseIntArrayAttr<int64_t>(pool.getKernelSize()));
        const auto strides = parseIntArrayAttr<int64_t>(pool.getStrides());
        const auto padsBegin = parseIntArrayAttr<int64_t>(pool.getPadsBegin());
        const auto padsEnd = parseIntArrayAttr<int64_t>(pool.getPadsEnd());
        if (kernelShape[Dims5D::Kernel::Z] != 1) {
            return true;
        }

        return !attrDValueEqualTo(strides, 1) || !attrDValueEqualTo(padsBegin, 0) || !attrDValueEqualTo(padsEnd, 0);
    }
    return true;
}

//
// ConvertNceOpsTo4DPass
//
class ConvertNceOpsTo4DPass final : public IE::ConvertNceOpsTo4DBase<ConvertNceOpsTo4DPass> {
public:
    explicit ConvertNceOpsTo4DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertNceOpsTo4DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalNceOp = [&](mlir::Operation* op) {
        const auto inputShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape();
        return inputShape.size() != 3;
    };

    const auto isLegalGroupConvOp = [&](IE::GroupConvolutionOp groupConv) {
        const auto inputShape = groupConv.getFilter().getType().cast<vpux::NDTypeInterface>().getShape();
        const auto hasGroups = groupConv.getGroups().value_or(0) != 0;
        return (inputShape.size() + hasGroups) != 4;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalNceOp);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalGroupConvOp);
    target.addDynamicallyLegalOp<IE::TransposedConvolutionOp>(isLegalNceOp);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>(&isLegalPoolNceOp<IE::MaxPoolOp>);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(&isLegalPoolNceOp<IE::AvgPoolOp>);
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvGeneralExpansion<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralExpansion<IE::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralExpansion<IE::TransposedConvolutionOp>>(&ctx, _log);
    patterns.add<PoolingGeneralExpansion<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<PoolingGeneralExpansion<IE::AvgPoolOp>>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertNceOpsTo4DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNceOpsTo4DPass(Logger log) {
    return std::make_unique<ConvertNceOpsTo4DPass>(log);
}

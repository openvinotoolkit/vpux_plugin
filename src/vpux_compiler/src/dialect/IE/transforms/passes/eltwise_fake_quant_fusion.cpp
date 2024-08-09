//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <functional>

using namespace vpux;

namespace {

//
// EltwiseFakeQuantizeFusion
//

template <typename ConcreteOp>
class EltwiseFakeQuantizeFusion final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    EltwiseFakeQuantizeFusion(mlir::MLIRContext* ctx, const FuncRef<float(float, float)> compute, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _compute(compute), _log(log) {
        this->setDebugName("EltwiseFakeQuantizeFusion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    FuncRef<float(float, float)> _compute;
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult EltwiseFakeQuantizeFusion<ConcreteOp>::matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    //
    //  Pattern matched:
    //
    //                                           +------------+
    //                                           |Scalar Const|
    //                                           +------------+
    //                                                 |
    //     +-----------------------------+   +---------------------+
    //     | non LayerWithPostOpInterface|   | optional per tensor |
    //     |     input producer          |   |   FakeQuantizeOp    |
    //     +-----------------------------+   +---------------------+
    //           |                                     |
    //     +------------+                              |
    //     | ConcreteOp |------------------------------+
    //     +------------+
    //       |  in_L in_H out_L out_H
    //       |    |    |     |     |
    //     +---------------------------+
    //     | per tensor FakeQuantizeOp |
    //     +---------------------------+
    //
    //  Replace with a single FakeQuantize with input low and high adjusted
    //
    //   +-----------------------------+
    //   | non LayerWithPostOpInterface|
    //   |     input producer          |            out_H
    //   +-----------------------------+       out_L  |
    //          |           in_H -,+,*,/ scalar  |    |
    //          | in_L -,+,*,/ scalar  |         |    |
    //          |            |         |         |    |
    //     +-------------------------------------------+
    //     |        per tensor FakeQuantizeOp          |
    //     +-------------------------------------------+
    //

    auto inLowConst = fakeQuantizeOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fakeQuantizeOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fakeQuantizeOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fakeQuantizeOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        _log.nest().trace("Got non constant parameters for FakeQuantize '{0}'", fakeQuantizeOp->getLoc());
        return mlir::failure();
    }

    auto inLowContentAttr = inLowConst.getContentAttr();
    auto inHighContentAttr = inHighConst.getContentAttr();
    auto outLowContentAttr = outLowConst.getContentAttr();
    auto outHighContentAttr = outHighConst.getContentAttr();
    if (!inLowContentAttr.isSplat() || !inHighContentAttr.isSplat() || !outLowContentAttr.isSplat() ||
        !outHighContentAttr.isSplat()) {
        _log.nest().trace("Got non scalar FakeQuantize parameters at '{0}'", fakeQuantizeOp->getLoc());
        return mlir::failure();
    }

    auto concreteParentOp = fakeQuantizeOp.getInput().getDefiningOp<ConcreteOp>();
    if (concreteParentOp == nullptr) {
        _log.nest().trace("The FakeQuantize input must be ConcreteOp at '{0}'", fakeQuantizeOp.getLoc());
        return mlir::failure();
    }

    bool lhsIsActivation = vpux::VPU::isEltwiseLhsActivation<ConcreteOp>(concreteParentOp);
    auto concreteProducerOp = lhsIsActivation ? concreteParentOp.getInput1().getDefiningOp()
                                              : concreteParentOp.getInput2().getDefiningOp();
    // The ConcreteOp is later fused as bias if producer op is executed on DPU
    if (concreteProducerOp != nullptr && mlir::isa<IE::LayerWithPostOpInterface>(concreteProducerOp)) {
        _log.nest().trace("The ConcreteOp input must not inherit the LayerWithPostOpInterface at '{0}'",
                          fakeQuantizeOp.getLoc());
        return mlir::failure();
    }

    auto concreteScalarInputFqOp = lhsIsActivation
                                           ? concreteParentOp.getInput2().template getDefiningOp<IE::FakeQuantizeOp>()
                                           : concreteParentOp.getInput1().template getDefiningOp<IE::FakeQuantizeOp>();
    auto concreteScalarInputDeclareOp =
            lhsIsActivation ? concreteParentOp.getInput2().template getDefiningOp<Const::DeclareOp>()
                            : concreteParentOp.getInput1().template getDefiningOp<Const::DeclareOp>();
    if (concreteScalarInputFqOp != nullptr) {
        concreteScalarInputDeclareOp = concreteScalarInputFqOp.getInput().template getDefiningOp<Const::DeclareOp>();
    }

    if (concreteScalarInputDeclareOp == nullptr) {
        _log.nest().trace("Second input of Concrete is not a DeclareOp at '{0}'", fakeQuantizeOp->getLoc());
        return mlir::failure();
    }

    auto concreteScalarInputContentAttr = concreteScalarInputDeclareOp.getContentAttr();
    if (!concreteScalarInputContentAttr.isSplat()) {
        _log.nest().trace("Constant Concrete input must be scalar at '{0}'", fakeQuantizeOp.getLoc());
        return mlir::failure();
    }

    auto concreteScalarInputValue = concreteScalarInputContentAttr.fold().template getSplatValue<float>();
    if (concreteScalarInputFqOp != nullptr) {
        auto concreteFQInLowConst = concreteScalarInputFqOp.getInputLow().template getDefiningOp<Const::DeclareOp>();
        auto concreteFQInHighConst = concreteScalarInputFqOp.getInputHigh().template getDefiningOp<Const::DeclareOp>();
        auto concreteFQOutLowConst = concreteScalarInputFqOp.getOutputLow().template getDefiningOp<Const::DeclareOp>();
        auto concreteFQOutHighConst =
                concreteScalarInputFqOp.getOutputHigh().template getDefiningOp<Const::DeclareOp>();
        if (concreteFQInLowConst == nullptr || concreteFQInHighConst == nullptr || concreteFQOutLowConst == nullptr ||
            concreteFQOutHighConst == nullptr) {
            _log.nest().trace("Got non constant parameters of FakeQuantize '{0}'", concreteScalarInputFqOp->getLoc());
            return mlir::failure();
        }
        auto concreteFQInLowContentAttr = concreteFQInLowConst.getContentAttr();
        auto concreteFQInHighContentAttr = concreteFQInHighConst.getContentAttr();
        auto concreteFQOutLowContentAttr = concreteFQOutLowConst.getContentAttr();
        auto concreteFQOutHighContentAttr = concreteFQOutHighConst.getContentAttr();
        if (!concreteFQInLowContentAttr.isSplat() || !concreteFQInHighContentAttr.isSplat() ||
            !concreteFQOutLowContentAttr.isSplat() || !concreteFQOutHighContentAttr.isSplat()) {
            _log.nest().trace("Got non scalar fake quantize range '{0}'", concreteScalarInputFqOp->getLoc());
            return mlir::failure();
        }
        auto concreteInLowValue = concreteFQInLowContentAttr.fold().template getSplatValue<float>();
        auto concreteInHighValue = concreteFQInHighContentAttr.fold().template getSplatValue<float>();
        auto concreteOutLowValue = concreteFQOutLowContentAttr.fold().template getSplatValue<float>();
        auto concreteOutHighValue = concreteFQOutHighContentAttr.fold().template getSplatValue<float>();
        auto levels = concreteScalarInputFqOp.getLevels();
        float fLevels = checked_cast<float>(levels.value());
        concreteScalarInputValue = fakeQuantize(concreteScalarInputValue, concreteInLowValue, concreteInHighValue,
                                                concreteOutLowValue, concreteOutHighValue, fLevels);
    }

    auto oldInLowValue = inLowContentAttr.fold().getSplatValue<float>();
    auto oldInHighValue = inHighContentAttr.fold().getSplatValue<float>();
    auto newInLowValue = _compute(oldInLowValue, concreteScalarInputValue);
    auto newInHighValue = _compute(oldInHighValue, concreteScalarInputValue);
    auto newInLowConst = Const::createFloatConst(rewriter, fakeQuantizeOp.getLoc(),
                                                 inLowConst.getType().cast<mlir::RankedTensorType>(), newInLowValue);
    auto newInHighConst = Const::createFloatConst(rewriter, fakeQuantizeOp.getLoc(),
                                                  inHighConst.getType().cast<mlir::RankedTensorType>(), newInHighValue);

    auto concreteActivationInputOp = lhsIsActivation ? concreteParentOp.getInput1() : concreteParentOp.getInput2();
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
            fakeQuantizeOp, concreteActivationInputOp, newInLowConst, newInHighConst, fakeQuantizeOp.getOutputLow(),
            fakeQuantizeOp.getOutputHigh(), fakeQuantizeOp.getLevelsAttr(), fakeQuantizeOp.getLowFpTypeAttr(),
            fakeQuantizeOp.getAutoBroadcastAttr());
    return mlir::success();
}

//
// EltwiseFakeQuantizeFusionPass
//

class EltwiseFakeQuantizeFusionPass final : public IE::EltwiseFakeQuantizeFusionBase<EltwiseFakeQuantizeFusionPass> {
public:
    explicit EltwiseFakeQuantizeFusionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void EltwiseFakeQuantizeFusionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    // Because the Eltwise operation with one scalar input is the producer op for the FakeQuantize below the
    // mathematical operation that will fuse the Eltwise operation in FakeQuantize input range will be exactly the
    // opposite one. Example: Subtract(scalar input = 3) - > FakeQuantize(inLow = -5, inHigh = 7, ...) will convert to
    // FakeQuantize(inLow = -2, inHigh = 10, ...), the effect if Subtract scalar being incorporated in the FakeQuantize
    // input range
    patterns.add<EltwiseFakeQuantizeFusion<IE::AddOp>>(&ctx, std::minus<float>(), _log);
    patterns.add<EltwiseFakeQuantizeFusion<IE::SubtractOp>>(&ctx, std::plus<float>(), _log);
    // TODO: E#129083
    // patterns.add<EltwiseFakeQuantizeFusion<IE::MultiplyOp>>(&ctx, std::divides<float>(), _log);
    patterns.add<EltwiseFakeQuantizeFusion<IE::DivideOp>>(&ctx, std::multiplies<float>(), _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createEltwiseFakeQuantizeFusionPass(Logger log) {
    return std::make_unique<EltwiseFakeQuantizeFusionPass>(log);
}

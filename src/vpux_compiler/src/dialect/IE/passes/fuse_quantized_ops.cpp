//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// FuseWithConv
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (conv) --- (dequantize) -- [filter]
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithConv final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithConv(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithConv::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto convOp = quantizeOp.input().getDefiningOp<IE::ConvolutionOp>();
    if (convOp == nullptr) {
        return mlir::failure();
    }

    if (VPUIP::NCEInvariant::verifyKernel(convOp, _log).failed()) {
        return mlir::failure();
    }

    auto inputDequantizeOp = convOp.input().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    auto filterDequantizeOp = convOp.filter().getDefiningOp<IE::DequantizeOp>();
    if (filterDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(), filterDequantizeOp.input(), convOp.bias(),
            convOp.strides(), convOp.pads_begin(), convOp.pads_end(), convOp.dilations(), convOp.post_opAttr());

    return mlir::success();
}

//
// FuseWithMaxPool
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (pool)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithMaxPool final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithMaxPool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithMaxPool");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithMaxPool::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto maxPoolOp = quantizeOp.input().getDefiningOp<IE::MaxPoolOp>();
    if (maxPoolOp == nullptr) {
        return mlir::failure();
    }

    if (VPUIP::NCEInvariant::verifyKernel(maxPoolOp, _log).failed()) {
        return mlir::failure();
    }

    auto inputDequantizeOp = maxPoolOp.input().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(), maxPoolOp.kernel_size(), maxPoolOp.strides(),
            maxPoolOp.pads_begin(), maxPoolOp.pads_end(), maxPoolOp.rounding_type(), maxPoolOp.post_opAttr());

    return mlir::success();
}

//
// FuseWithEltwiseAdd
//

//
//      [input 1]    [input 2]
//          |            |
//     (dequantize) (dequantize)
//          |            |
//           (EltwiseAdd)
//                |
//            [output]
//                |
//           (quantize)
//

class FuseWithEltwiseAdd final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithEltwiseAdd(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithEltwiseAdd");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithEltwiseAdd::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                        mlir::PatternRewriter& rewriter) const {
    const auto quantOutType = quantizeOp.output().getType();
    auto quantElemOutType = quantOutType.cast<mlir::ShapedType>().getElementType();
    if (quantElemOutType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return mlir::failure();
    }

    auto addOp = quantizeOp.input().getDefiningOp<IE::AddOp>();
    if (addOp == nullptr) {
        return mlir::failure();
    }

    const auto checkDequantizeOp = [](IE::DequantizeOp dequantOp) {
        if (dequantOp == nullptr) {
            return mlir::failure();
        }

        const auto dequantInType = dequantOp.input().getType();
        auto dequantElemInType = dequantInType.cast<mlir::ShapedType>().getElementType();
        if (dequantElemInType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
            return mlir::failure();
        }

        return mlir::success();
    };

    auto input1DequantizeOp = addOp.input1().getDefiningOp<IE::DequantizeOp>();
    if (mlir::failed(checkDequantizeOp(input1DequantizeOp))) {
        return mlir::failure();
    }

    auto input2DequantizeOp = addOp.input2().getDefiningOp<IE::DequantizeOp>();
    if (mlir::failed(checkDequantizeOp(input2DequantizeOp))) {
        return mlir::failure();
    }

    // Perform check for input types. In case they are quantized such check
    // will also cover if quant parameters are aligned
    if (input1DequantizeOp.input().getType() != input2DequantizeOp.input().getType()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(quantizeOp, quantizeOp.getType(), input1DequantizeOp.input(),
                                           input2DequantizeOp.input(), addOp.auto_broadcastAttr(), addOp.post_opAttr());

    return mlir::success();
}

//
// FuseWithSlice
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (slice)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithSlice final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithSlice::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto sliceOp = quantizeOp.input().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return mlir::failure();
    }

    auto inputDequantizeOp = sliceOp.source().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::SliceOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.input(),
                                             sliceOp.static_offsetsAttr(), sliceOp.static_sizesAttr());

    return mlir::success();
}

//
// FuseWithConcat
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (concat)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithConcat final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithConcat::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto concatOp = quantizeOp.input().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(concatOp.inputs().size());

    const auto inType = concatOp.inputs().front().getType().cast<mlir::RankedTensorType>();
    const auto perAxisQType = inType.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();

    auto dequantizeOp = concatOp.inputs().front().getDefiningOp<IE::DequantizeOp>();
    const auto dequantizeInputType = dequantizeOp.input().getType().cast<mlir::RankedTensorType>().getElementType();

    for (auto in : concatOp.inputs()) {
        auto inputDequantizeOp = in.getDefiningOp<IE::DequantizeOp>();
        if (inputDequantizeOp == nullptr) {
            return mlir::failure();
        }

        auto inputDequantizeOpType = inputDequantizeOp.input().getType().cast<mlir::RankedTensorType>();
        auto inputDequantizeElemType = inputDequantizeOpType.getElementType();
        if (dequantizeInputType != inputDequantizeElemType) {
            return mlir::failure();
        }

        const auto curPerAxisQType = inputDequantizeElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();

        if ((perAxisQType == nullptr && curPerAxisQType != nullptr) ||
            (perAxisQType != nullptr && curPerAxisQType == nullptr)) {
            return mlir::failure();
        }
        if (perAxisQType != nullptr && curPerAxisQType != nullptr) {
            if (!canBeMerged(curPerAxisQType, perAxisQType)) {
                return mlir::failure();
            }
        }

        newConcatInputs.push_back(inputDequantizeOp.input());
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(quantizeOp, quantizeOp.getType(), newConcatInputs, concatOp.axis(),
                                              concatOp.offset(), concatOp.stride());

    return mlir::success();
}

//
// FuseQuantizedOpsPass
//

class FuseQuantizedOpsPass final : public IE::FuseQuantizedOpsBase<FuseQuantizedOpsPass> {
public:
    explicit FuseQuantizedOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseQuantizedOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<FuseWithConv>(&ctx, _log);
    patterns.add<FuseWithEltwiseAdd>(&ctx, _log);
    patterns.add<FuseWithSlice>(&ctx, _log);
    patterns.add<FuseWithMaxPool>(&ctx, _log);
    patterns.add<FuseWithConcat>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseQuantizedOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseQuantizedOpsPass(Logger log) {
    return std::make_unique<FuseQuantizedOpsPass>(log);
}

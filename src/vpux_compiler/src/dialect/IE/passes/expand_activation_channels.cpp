//
// Copyright Intel Corporation.
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
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// calcPadsEnd
//

Shape calcPadsEnd(mlir::ShapedType origType, int64_t channelAlignment) {
    const auto act_channel_dim = IE::ConvolutionOp::act_channel_dim();

    const auto origShape = getShape(origType);

    const auto extendedChannels = alignVal(origShape[act_channel_dim], channelAlignment);

    Shape padsEndArrayShape(origShape.size(), 0);
    padsEndArrayShape[act_channel_dim] = extendedChannels - origShape[act_channel_dim];

    return padsEndArrayShape;
}

//
// generalRewrite
//

//
// Max/Avg Pooling and Convolution Ops should be handled there
//
// opCreator - function, which should place back operation, which being proceed, with new expanded input
//

mlir::LogicalResult generalRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
                                   FuncRef<mlir::Operation*(mlir::Value, int64_t)> opCreator, Logger log) {
    const auto act_channel_dim = IE::ConvolutionOp::act_channel_dim();

    auto* ctx = origOp->getContext();

    const auto inputType = origOp->getOperand(0).getType().cast<mlir::ShapedType>();
    const auto outputType = origOp->getResult(0).getType().cast<mlir::ShapedType>();

    const auto channelAlignement = VPUIP::NCEInvariant::getChannelAlignment(inputType.getElementType());
    const auto inPadsEnd = calcPadsEnd(inputType, channelAlignement);
    const auto outPadsEnd = calcPadsEnd(outputType, channelAlignement);

    log.trace("Input padding : {0}", inPadsEnd);
    log.trace("Output padding : {0}", outPadsEnd);

    if (inPadsEnd[act_channel_dim] == 0 && outPadsEnd[act_channel_dim] == 0) {
        return matchFailed(log, rewriter, origOp, "Both input and output channels are already aligned");
    }

    mlir::Value paddedInput;
    if (inPadsEnd[act_channel_dim] == 0) {
        log.trace("Input channels are already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        log.trace("Expand input tensor");

        const SmallVector<int64_t> inPadsBegin(inPadsEnd.size(), 0);

        paddedInput =
                rewriter.create<IE::ExpandOp>(origOp->getLoc(), origOp->getOperand(0),
                                              getInt32ArrayAttr(ctx, inPadsBegin), getInt32ArrayAttr(ctx, inPadsEnd));
    }

    log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(paddedInput, outPadsEnd[act_channel_dim]);

    if (outPadsEnd[act_channel_dim] == 0) {
        log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp->getResult(0));
    } else {
        log.trace("Extract meaningful part from extened output");

        const auto outShape = outputType.getShape();
        const SmallVector<int64_t> offsets(outShape.size(), 0);
        const SmallVector<int64_t> strides(outShape.size(), 1);

        auto subTensorOp = rewriter.create<mlir::tensor::ExtractSliceOp>(
                origOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), mlir::ValueRange{},
                mlir::ValueRange{}, mlir::ValueRange{}, getInt64ArrayAttr(ctx, offsets),
                getInt64ArrayAttr(ctx, outShape), getInt64ArrayAttr(ctx, strides));

        rewriter.replaceOp(origOp, subTensorOp.result());
    }

    return mlir::success();
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("MaxPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t) -> mlir::Operation* {
        return rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), expandedInput, origOp.kernel_size(), origOp.strides(),
                                              origOp.pads_begin(), origOp.pads_end(), origOp.rounding_type());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// ConvolutionRewriter
//

class ConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        // We have to expand channels count for filter as well
        const auto filter_out_channel_dim = IE::ConvolutionOp::filter_out_channel_dim();
        const auto filter_in_channel_dim = IE::ConvolutionOp::filter_in_channel_dim();
        const auto act_channel_dim = IE::ConvolutionOp::act_channel_dim();

        const auto filterShape = getShape(origOp.filter());

        const auto newInputShape = getShape(expandedInput);
        const auto inChanPadEnd = newInputShape[act_channel_dim] - filterShape[filter_in_channel_dim];

        mlir::Value paddedFilter;

        if (inChanPadEnd == 0 && outChanPadEnd == 0) {
            paddedFilter = origOp.filter();
        } else {
            const SmallVector<int64_t> filterPadsBegin(filterShape.size(), 0);

            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[filter_out_channel_dim] = outChanPadEnd;
            filterPadsEnd[filter_in_channel_dim] = inChanPadEnd;

            const auto padValue = getFP32Attr(getContext(), 0.0f);

            paddedFilter = rewriter.createOrFold<IE::PadOp>(origOp->getLoc(), origOp.filter(), nullptr, nullptr,
                                                            nullptr, getInt32ArrayAttr(getContext(), filterPadsBegin),
                                                            getInt32ArrayAttr(getContext(), filterPadsEnd.raw()),
                                                            padValue, IE::PadMode::CONSTANT);
        }

        mlir::Value paddedBiases;

        if (origOp.bias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.bias();
            } else {
                const auto biasShape = getShape(origOp.bias());

                const SmallVector<uint32_t> biasPadsBegin(biasShape.size(), 0);

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[act_channel_dim] = checked_cast<uint32_t>(outChanPadEnd);

                const auto padValue = getFP32Attr(getContext(), 0.0f);

                paddedBiases = rewriter.createOrFold<IE::PadOp>(origOp->getLoc(), origOp.bias(), nullptr, nullptr,
                                                                nullptr, getInt32ArrayAttr(getContext(), biasPadsBegin),
                                                                getInt32ArrayAttr(getContext(), biasPadsEnd.raw()),
                                                                padValue, IE::PadMode::CONSTANT);
            }
        }

        return rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), expandedInput, paddedFilter, paddedBiases,
                                                  origOp.strides(), origOp.pads_begin(), origOp.pads_end(),
                                                  origOp.dilations());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// ExpandActivationChannelsPass
//

class ExpandActivationChannelsPass final : public IE::ExpandActivationChannelsBase<ExpandActivationChannelsPass> {
public:
    explicit ExpandActivationChannelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ExpandActivationChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        _log.trace("Got MaxPool layer at '{0}'", op->getLoc());

        if (!VPUIP::NCEInvariant::verifyKernel(op, _log.nest()).succeeded()) {
            _log.nest().trace("The operation is not supported by HW, skipping");
            return true;
        }

        return VPUIP::NCEInvariant::verifyChannels(op, _log.nest()).succeeded();
    });
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        _log.trace("Got Convolution layer at '{0}'", op->getLoc());

        if (!VPUIP::NCEInvariant::verifyKernel(op, _log.nest()).succeeded()) {
            _log.nest().trace("The operation is not supported by HW, skipping");
            return true;
        }

        return VPUIP::NCEInvariant::verifyChannels(op, _log.nest()).succeeded();
    });
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ExpandOp, IE::PadOp>();
    target.addLegalOp<mlir::tensor::ExtractSliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MaxPoolRewriter>(&ctx, _log);
    patterns.insert<ConvolutionRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createExpandActivationChannelsPass(Logger log) {
    return std::make_unique<ExpandActivationChannelsPass>(log);
}

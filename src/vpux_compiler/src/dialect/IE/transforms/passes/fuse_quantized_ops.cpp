//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/attributes_properties_conversion.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

bool areAllUsersQuantized(mlir::Operation* op) {
    for (auto user : op->getUsers()) {
        if (mlir::dyn_cast<IE::QuantizeOp>(user) == nullptr) {
            return false;
        }
    }
    return true;
}

//
// FuseWithConvBase
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

template <class ConcreteOp>
class FuseWithConvBase : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithConvBase(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        this->setDebugName("FuseWithConvBase");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;
    virtual bool isSupportedConvBasedOp(ConcreteOp origOp, Logger log) const = 0;
    virtual ConcreteOp createNewConvBasedOp(IE::QuantizeOp quantizeOp, ConcreteOp origOp, mlir::Value newInput,
                                            mlir::Value newWeights, mlir::PatternRewriter& rewriter) const = 0;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult FuseWithConvBase<ConcreteOp>::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto convBaseOp = quantizeOp.getInput().getDefiningOp<ConcreteOp>();
    if (convBaseOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(convBaseOp)) {
        return mlir::failure();
    }

    if (!isSupportedConvBasedOp(convBaseOp, _log)) {
        return mlir::failure();
    }

    auto inputDequantizeOp = convBaseOp.getInput().template getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    auto filterDequantizeOp = convBaseOp.getFilter().template getDefiningOp<IE::DequantizeOp>();
    if (filterDequantizeOp == nullptr) {
        return mlir::failure();
    }

    // On VPUX37XX, VPUX40XX the prelu alpha multiplier used for integer input is unsigned, on floating
    // input it is signed. If input is floating, output is integer, quantize output need to be per tensor, this will
    // check in mix-precision pass
    const auto arch = VPU::getArch(quantizeOp->getParentOfType<mlir::ModuleOp>());
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU37XX,
            VPU::ArchKind::NPU40XX,
    };
    if (compatibleTargets.count(arch) > 0) {
        if (convBaseOp.getPostOpAttr() != nullptr &&
            convBaseOp.getPostOpAttr().getName().getValue() == IE::LeakyReluOp::getOperationName()) {
            IE::LeakyReluOp::Adaptor leakyRelu(std::nullopt, nullptr,
                                               toProperties<IE::LeakyReluOp>(convBaseOp.getPostOpAttr().getAttrs()));
            if (leakyRelu.verify(convBaseOp->getLoc()).succeeded()) {
                const auto alpha = leakyRelu.getNegativeSlope().convertToDouble();
                if (alpha < 0.0) {
                    return mlir::failure();
                }
            }
        }
    }

    // Could not fuse if bias rescale check fail
    if (mlir::failed(checkRescaledBiasRange(convBaseOp))) {
        return mlir::failure();
    }

    auto newConvBaseOp = createNewConvBasedOp(quantizeOp, convBaseOp, inputDequantizeOp.getInput(),
                                              filterDequantizeOp.getInput(), rewriter);
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newConvBaseOp)) {
        rewriter.eraseOp(newConvBaseOp);
        return mlir::failure();
    }

    rewriter.replaceOp(quantizeOp, newConvBaseOp.getOutput());
    return mlir::success();
}

//
// FuseWithConv
//

class FuseWithConv final : public FuseWithConvBase<IE::ConvolutionOp> {
public:
    FuseWithConv(mlir::MLIRContext* ctx, Logger log): FuseWithConvBase<IE::ConvolutionOp>(ctx, log) {
        setDebugName("FuseWithConv");
    }

    bool isSupportedConvBasedOp(IE::ConvolutionOp conv, Logger log) const override;
    IE::ConvolutionOp createNewConvBasedOp(IE::QuantizeOp quantizeOp, IE::ConvolutionOp conv, mlir::Value newInput,
                                           mlir::Value newWeights, mlir::PatternRewriter& rewriter) const override;
};

bool FuseWithConv::isSupportedConvBasedOp(IE::ConvolutionOp conv, Logger log) const {
    return VPU::NCEConvolutionOp::verifyKernel(conv, log).succeeded();
}

IE::ConvolutionOp FuseWithConv::createNewConvBasedOp(IE::QuantizeOp quantizeOp, IE::ConvolutionOp conv,
                                                     mlir::Value newInput, mlir::Value newWeights,
                                                     mlir::PatternRewriter& rewriter) const {
    auto newConv = rewriter.create<IE::ConvolutionOp>(conv->getLoc(), quantizeOp.getType(), newInput, newWeights,
                                                      conv.getBias(), conv.getStrides(), conv.getPadsBegin(),
                                                      conv.getPadsEnd(), conv.getDilations(), conv.getPostOpAttr(),
                                                      conv.getClampAttr(), conv.getStaticScaleAttr());

    return newConv;
}

//
// FuseWithGroupConv
//

class FuseWithGroupConv final : public FuseWithConvBase<IE::GroupConvolutionOp> {
public:
    FuseWithGroupConv(mlir::MLIRContext* ctx, Logger log): FuseWithConvBase<IE::GroupConvolutionOp>(ctx, log) {
        setDebugName("FuseWithGroupConv");
    }

    bool isSupportedConvBasedOp(IE::GroupConvolutionOp grConvOp, Logger log) const override;
    IE::GroupConvolutionOp createNewConvBasedOp(IE::QuantizeOp quantizeOp, IE::GroupConvolutionOp grConvOp,
                                                mlir::Value newInput, mlir::Value newWeights,
                                                mlir::PatternRewriter& rewriter) const override;
};

bool FuseWithGroupConv::isSupportedConvBasedOp(IE::GroupConvolutionOp grConvOp, Logger log) const {
    return VPU::NCEDepthConvolutionOp::verifyKernel(grConvOp, log).succeeded();
}

IE::GroupConvolutionOp FuseWithGroupConv::createNewConvBasedOp(IE::QuantizeOp quantizeOp,
                                                               IE::GroupConvolutionOp grConvOp, mlir::Value newInput,
                                                               mlir::Value newWeights,
                                                               mlir::PatternRewriter& rewriter) const {
    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            grConvOp->getLoc(), quantizeOp.getType(), newInput, newWeights, grConvOp.getBias(), grConvOp.getStrides(),
            grConvOp.getPadsBegin(), grConvOp.getPadsEnd(), grConvOp.getDilations(), grConvOp.getGroupsAttr(),
            grConvOp.getPostOpAttr(), grConvOp.getClampAttr());

    return newGroupConv;
}

//
// FuseWithTransposedConv
//

class FuseWithTransposedConv final : public FuseWithConvBase<IE::TransposedConvolutionOp> {
public:
    FuseWithTransposedConv(mlir::MLIRContext* ctx, Logger log)
            : FuseWithConvBase<IE::TransposedConvolutionOp>(ctx, log) {
        setDebugName("FuseWithTransposedConv");
    }

    bool isSupportedConvBasedOp(IE::TransposedConvolutionOp transposedConvOp, Logger log) const override;
    IE::TransposedConvolutionOp createNewConvBasedOp(IE::QuantizeOp quantizeOp,
                                                     IE::TransposedConvolutionOp transposedConvOp, mlir::Value newInput,
                                                     mlir::Value newWeights,
                                                     mlir::PatternRewriter& rewriter) const override;
};

bool FuseWithTransposedConv::isSupportedConvBasedOp(IE::TransposedConvolutionOp transposedConvOp, Logger log) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    return VPU::isSupportedSEPTransposedConv(transposedConvOp, logCb, /*checkLayout=*/false,
                                             /*checkChannelAlignment=*/false);
}

IE::TransposedConvolutionOp FuseWithTransposedConv::createNewConvBasedOp(IE::QuantizeOp quantizeOp,
                                                                         IE::TransposedConvolutionOp transposedConvOp,
                                                                         mlir::Value newInput, mlir::Value newWeights,
                                                                         mlir::PatternRewriter& rewriter) const {
    auto newTransposedConv = rewriter.create<IE::TransposedConvolutionOp>(
            transposedConvOp->getLoc(), quantizeOp.getType(), newInput, newWeights, transposedConvOp.getOutputShape(),
            transposedConvOp.getBias(), transposedConvOp.getStrides(), transposedConvOp.getPadsBegin(),
            transposedConvOp.getPadsEnd(), transposedConvOp.getDilations(), transposedConvOp.getOutputPaddingAttr(),
            transposedConvOp.getPostOpAttr(), transposedConvOp.getClampAttr());

    return newTransposedConv;
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
    auto maxPoolOp = quantizeOp.getInput().getDefiningOp<IE::MaxPoolOp>();
    if (maxPoolOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(maxPoolOp)) {
        return mlir::failure();
    }

    if (VPU::NCEMaxPoolOp::verifyKernel(maxPoolOp, _log).failed()) {
        return mlir::failure();
    }

    // MaxPool IDU does not support zero-point subtraction, so it compensates by ignoring output zero-point as well.
    // Since we are not subtracting the input zero-point, the non-linear post-op will operate on improper data.
    // Only zero-centered values would be supported. Currently, quantized MaxPool is disabled for all post-ops.
    auto mainOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(maxPoolOp.getOperation());
    if (mainOp != nullptr) {
        if (mainOp.getPostOp().has_value()) {
            return mlir::failure();
        }
    }

    auto inputDequantizeOp = maxPoolOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
                    quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(), maxPoolOp.getKernelSize(),
                    maxPoolOp.getStrides(), maxPoolOp.getPadsBegin(), maxPoolOp.getPadsEnd(),
                    maxPoolOp.getRoundingType(), maxPoolOp.getPostOpAttr(), maxPoolOp.getClampAttr())
            ->setLoc(maxPoolOp->getLoc());

    return mlir::success();
}

//
// FuseWithAveragePool
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

class FuseWithAveragePool final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithAveragePool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithAveragePool");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithAveragePool::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto avgPoolOp = quantizeOp.getInput().getDefiningOp<IE::AvgPoolOp>();
    if (avgPoolOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(avgPoolOp)) {
        return mlir::failure();
    }

    if (VPU::NCEAveragePoolOp::verifyKernel(avgPoolOp, _log).failed()) {
        return mlir::failure();
    }

    // AveragePool IDU does not support zero-point subtraction, so it compensates by ignoring output zero-point as well.
    // Since we are not subtracting the input zero-point, the non-linear post-op will operate on improper data.
    // Only zero-centered values would be supported. Currently, quantized AveragePool is disabled for all post-ops.
    auto mainOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(avgPoolOp.getOperation());
    if (mainOp != nullptr) {
        if (mainOp.getPostOp().has_value()) {
            return mlir::failure();
        }
    }

    auto inputDequantizeOp = avgPoolOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(),
                                               avgPoolOp.getKernelSize(), avgPoolOp.getStrides(),
                                               avgPoolOp.getPadsBegin(), avgPoolOp.getPadsEnd(),
                                               avgPoolOp.getRoundingTypeAttr(), avgPoolOp.getExcludePadsAttr(),
                                               avgPoolOp.getPostOpAttr(), avgPoolOp.getClampAttr())
            ->setLoc(avgPoolOp->getLoc());

    return mlir::success();
}

bool isLegalFuseOp(mlir::Operation* concreteOp, IE::QuantizeOp quantizeOp) {
    if (!areAllUsersQuantized(concreteOp)) {
        return false;
    }

    auto inputDequantizeOp = concreteOp->getOperand(0).getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return false;
    }

    auto origOutput = quantizeOp.getOutput();
    auto origInput = inputDequantizeOp.getInput();
    auto tileOpInputElementType = origInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    auto tileOpOutputElementType = origOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    if (tileOpInputElementType != tileOpOutputElementType) {
        return false;
    }

    return true;
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
    auto sliceOp = quantizeOp.getInput().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return mlir::failure();
    }

    if (!isLegalFuseOp(sliceOp, quantizeOp)) {
        return matchFailed(rewriter, sliceOp, "Quantize op cannot fuse into op {0} at {1}", sliceOp->getName(),
                           sliceOp->getLoc());
    }

    rewriter.replaceOpWithNewOp<IE::SliceOp>(quantizeOp, quantizeOp.getType(),
                                             sliceOp.getSource().getDefiningOp<IE::DequantizeOp>().getInput(),
                                             sliceOp.getStaticOffsetsAttr(), sliceOp.getStaticSizesAttr())
            ->setLoc(sliceOp->getLoc());

    return mlir::success();
}

//
// FuseWithTile
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (tile)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithTile final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithTile(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithTile");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithTile::matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const {
    auto tileOp = quantizeOp.getInput().getDefiningOp<IE::TileOp>();
    if (tileOp == nullptr) {
        return mlir::failure();
    }

    if (!isLegalFuseOp(tileOp, quantizeOp)) {
        return matchFailed(rewriter, tileOp, "Quantize op cannot fuse into op {0} at {1}", tileOp->getName(),
                           tileOp->getLoc());
    }

    rewriter.replaceOpWithNewOp<IE::TileOp>(quantizeOp, quantizeOp.getType(),
                                            tileOp.getInput().getDefiningOp<IE::DequantizeOp>().getInput(), nullptr,
                                            tileOp.getRepeatsValuesAttr());

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
    auto concatOp = quantizeOp.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(concatOp)) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(concatOp.getInputs().size());

    auto dequantizeOp = concatOp.getInputs().front().getDefiningOp<IE::DequantizeOp>();
    if (dequantizeOp == nullptr) {
        return mlir::failure();
    }

    for (auto in : concatOp.getInputs()) {
        auto inputDequantizeOp = in.getDefiningOp<IE::DequantizeOp>();
        if (inputDequantizeOp == nullptr) {
            return mlir::failure();
        }

        if (!newConcatInputs.empty()) {
            const auto prevElemType = newConcatInputs.back().getType().cast<vpux::NDTypeInterface>().getElementType();
            const auto curElemType =
                    inputDequantizeOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();

            if (const auto prevPerAxisType = prevElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
                if (const auto curPerAxisType = curElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
                    if (!canBeMerged(prevPerAxisType, curPerAxisType)) {
                        return mlir::failure();
                    }
                } else {
                    return mlir::failure();
                }
            } else if (prevElemType != curElemType) {
                return mlir::failure();
            }
        }

        newConcatInputs.push_back(inputDequantizeOp.getInput());
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(quantizeOp, newConcatInputs, concatOp.getPerAxisAttr(),
                                              concatOp.getStaticOffsetsAttr())
            ->setLoc(concatOp->getLoc());

    return mlir::success();
}

//
// FuseWithReduce
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (reduce)
//          |
//       [output]
//          |
//      (quantize)
//

template <class ConcreteOp>
class FuseWithReduce final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithReduce(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        this->setDebugName("FuseWithReduce");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult FuseWithReduce<ConcreteOp>::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto reduceOp = quantizeOp.getInput().getDefiningOp<ConcreteOp>();
    if (reduceOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(reduceOp)) {
        return mlir::failure();
    }

    auto isNCESupported = VPU::NCEInvariant::isSupported(reduceOp.getOperation(), _log);
    if (isNCESupported.failed()) {
        return mlir::failure();
    }

    auto inputDequantizeOp = reduceOp.getInput().template getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }
    auto axes = getIntArrayAttr(this->getContext(), IE::extractAxes(reduceOp->getLoc(), reduceOp));
    rewriter.replaceOpWithNewOp<ConcreteOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(),
                                            /*axes*/ nullptr,
                                            /*axes_value*/ axes, /*keep_dims*/ true)
            ->setLoc(reduceOp->getLoc());
    return mlir::success();
}

//
// FuseWithEltwiseConverter
//

//
//      [input 1]     [input 2]
//          |             |
//     (dequantize)  (dequantize)
//          |             |
//           -(EltwiseOp)-
//                 |
//             [output]
//                 |
//            (quantize)
//

template <class ConcreteOp>
class FuseWithEltwiseConverter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithEltwiseConverter(mlir::MLIRContext* ctx,
                             FuncRef<mlir::LogicalResult(mlir::Type, mlir::Type)> checkInputTypes, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _checkInputTypes(checkInputTypes), _log(log) {
        this->setDebugName("FuseWithEltwiseConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    FuncRef<mlir::LogicalResult(mlir::Type, mlir::Type)> _checkInputTypes;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult FuseWithEltwiseConverter<ConcreteOp>::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    const auto quantOutType = quantizeOp.getOutput().getType();
    auto quantElemOutType = quantOutType.cast<vpux::NDTypeInterface>().getElementType();
    if (quantElemOutType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return mlir::failure();
    }

    auto eltwiseOp = quantizeOp.getInput().getDefiningOp<ConcreteOp>();
    if (eltwiseOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(eltwiseOp)) {
        return mlir::failure();
    }

    if (eltwiseOp.getInput1().getType().template cast<vpux::NDTypeInterface>().getShape() !=
        eltwiseOp.getInput2().getType().template cast<vpux::NDTypeInterface>().getShape()) {
        return mlir::failure();
    }

    const auto checkDequantizeOp = [](IE::DequantizeOp dequantOp) {
        if (dequantOp == nullptr) {
            return mlir::failure();
        }

        const auto dequantInType = dequantOp.getInput().getType();
        auto dequantElemInType = dequantInType.template cast<vpux::NDTypeInterface>().getElementType();
        if (!dequantElemInType.template isa<mlir::quant::UniformQuantizedType>()) {
            return mlir::failure();
        }

        return mlir::success();
    };

    auto input1DequantizeOp = eltwiseOp.getInput1().template getDefiningOp<IE::DequantizeOp>();
    if (mlir::failed(checkDequantizeOp(input1DequantizeOp))) {
        return mlir::failure();
    }

    auto input2DequantizeOp = eltwiseOp.getInput2().template getDefiningOp<IE::DequantizeOp>();
    if (mlir::failed(checkDequantizeOp(input2DequantizeOp))) {
        return mlir::failure();
    }

    const auto input1Type =
            input1DequantizeOp.getInput().getType().template cast<vpux::NDTypeInterface>().getElementType();
    const auto input2Type =
            input2DequantizeOp.getInput().getType().template cast<vpux::NDTypeInterface>().getElementType();
    if (mlir::failed(_checkInputTypes(input1Type, input2Type))) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ConcreteOp>(quantizeOp, quantizeOp.getType(), input1DequantizeOp.getInput(),
                                            input2DequantizeOp.getInput(), eltwiseOp.getAutoBroadcastAttr(),
                                            eltwiseOp.getPostOpAttr(), eltwiseOp.getClampAttr())
            ->setLoc(eltwiseOp->getLoc());

    return mlir::success();
}

//
// FuseWithDepth2Space
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (DepthToSpace)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithDepth2Space final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithDepth2Space(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithDepth2Space");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithDepth2Space::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto depth2SpaceOp = quantizeOp.getInput().getDefiningOp<IE::DepthToSpaceOp>();
    if (depth2SpaceOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(depth2SpaceOp)) {
        return mlir::failure();
    }

    auto inputDequantizeOp = depth2SpaceOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(),
                                                    depth2SpaceOp.getBlockSizeAttr(), depth2SpaceOp.getModeAttr())
            ->setLoc(depth2SpaceOp->getLoc());

    return mlir::success();
}

//
// FuseWithInterpolate
//

//
//       [input]
//          |
//     (dequantize)
//          |
//     (interpolate)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithInterpolate final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithInterpolate(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithInterpolate");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithInterpolate::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto interpOp = quantizeOp.getInput().getDefiningOp<IE::InterpolateOp>();
    if (interpOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(interpOp)) {
        return mlir::failure();
    }

    auto isNCESupported = VPU::NCEInvariant::isSupported(interpOp.getOperation(), _log);
    if (isNCESupported.failed()) {
        return mlir::failure();
    }

    auto inputDequantizeOp = interpOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::InterpolateOp>(
                    quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(), nullptr, nullptr, nullptr,
                    interpOp.getSizesAttr().value_or(nullptr), interpOp.getScalesAttr().value_or(nullptr),
                    interpOp.getAxesAttr().value_or(nullptr), interpOp.getTileOffsetAttrAttr(),
                    interpOp.getInitialInputDimsAttrAttr(), interpOp.getInitialOutputDimsAttrAttr(), interpOp.getAttr())
            ->setLoc(interpOp->getLoc());

    return mlir::success();
}

//
// FuseQuantizedOpsPass
//

class FuseQuantizedOpsPass final : public IE::FuseQuantizedOpsBase<FuseQuantizedOpsPass> {
public:
    explicit FuseQuantizedOpsPass(const bool seOpsEnabled, const bool seExperimentalOpsEnabled, Logger log)
            : _seOpsEnabled(seOpsEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;
};

mlir::LogicalResult FuseQuantizedOpsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }

    if (seExperimentalOpsEnabled.hasValue()) {
        _seExperimentalOpsEnabled = seExperimentalOpsEnabled.getValue();
    }

    return mlir::success();
}

void FuseQuantizedOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    const auto arch = VPU::getArch(func->getParentOfType<mlir::ModuleOp>());

    const auto checkAddInputTypes = [&](mlir::Type input1Type, mlir::Type input2Type) -> mlir::LogicalResult {
        auto dequantElemIn1Type = input1Type.cast<mlir::quant::UniformQuantizedType>();
        auto dequantElemIn2Type = input2Type.cast<mlir::quant::UniformQuantizedType>();

        // Perform check for input types. AddOp supports quantization with different zp, but not different scales.
        if (dequantElemIn1Type.getExpressedType() != dequantElemIn2Type.getExpressedType() ||
            dequantElemIn1Type.getStorageType() != dequantElemIn2Type.getStorageType() ||
            dequantElemIn1Type.isSigned() != dequantElemIn2Type.isSigned()) {
            return mlir::failure();
        }

        if (!supportsPerInputEltwiseScale(arch) && !isFloatEqual(static_cast<float>(dequantElemIn1Type.getScale()),
                                                                 static_cast<float>(dequantElemIn2Type.getScale()))) {
            return mlir::failure();
        }

        return mlir::success();
    };

    // Can be used for MultiplyOp and SubtractOp as well
    const auto checkMulInputTypes = [](mlir::Type input1Type, mlir::Type input2Type) -> mlir::LogicalResult {
        auto dequantElemIn1Type = input1Type.cast<mlir::quant::UniformQuantizedType>();
        auto dequantElemIn2Type = input2Type.cast<mlir::quant::UniformQuantizedType>();

        // Perform check for input types. MultiplyOp and SubtractOp supports quantization with different scales and zp.
        if (dequantElemIn1Type.getExpressedType() != dequantElemIn2Type.getExpressedType() ||
            dequantElemIn1Type.getStorageType() != dequantElemIn2Type.getStorageType() ||
            dequantElemIn1Type.isSigned() != dequantElemIn2Type.isSigned()) {
            return mlir::failure();
        }

        return mlir::success();
    };

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseWithConv>(&ctx, _log);
    patterns.add<FuseWithGroupConv>(&ctx, _log);
    patterns.add<FuseWithEltwiseConverter<IE::AddOp>>(&ctx, checkAddInputTypes, _log);
    patterns.add<FuseWithSlice>(&ctx, _log);
    patterns.add<FuseWithMaxPool>(&ctx, _log);
    patterns.add<FuseWithTile>(&ctx, _log);
    patterns.add<FuseWithReduce<IE::ReduceMeanOp>>(&ctx, _log);
    patterns.add<FuseWithAveragePool>(&ctx, _log);
    patterns.add<FuseWithConcat>(&ctx, _log);
    // VPUX37XX and VPUX40XX NCE does not support element-wise multiplication, skip the fusion
    const std::set<VPU::ArchKind> incompatibleTargets = {
            VPU::ArchKind::NPU37XX,
            VPU::ArchKind::NPU40XX,
    };
    if (incompatibleTargets.count(arch) <= 0) {
        patterns.add<FuseWithEltwiseConverter<IE::MultiplyOp>>(&ctx, checkMulInputTypes, _log);
    } else {
        patterns.add<FuseWithDepth2Space>(&ctx, _log);
    }
    if (_seOpsEnabled) {
        patterns.add<FuseWithInterpolate>(&ctx, _log);
        patterns.add<FuseWithTransposedConv>(&ctx, _log);
    }

    // TODO: optimize for SEP Pad & Roll
    VPUX_UNUSED(_seExperimentalOpsEnabled);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseQuantizedOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseQuantizedOpsPass(const bool seOpsEnabled,
                                                                 const bool seExperimentalOpsEnabled, Logger log) {
    return std::make_unique<FuseQuantizedOpsPass>(seOpsEnabled, seExperimentalOpsEnabled, log);
}

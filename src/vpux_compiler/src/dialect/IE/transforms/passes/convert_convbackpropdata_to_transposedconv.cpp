//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvolutionBackpropDataConversion
//

class ConvolutionBackpropDataConversion final : public mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp> {
public:
    ConvolutionBackpropDataConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp>(ctx), _log(log) {
        setDebugName("ConvolutionBackpropDataConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvolutionBackpropDataConversion::matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::ConvolutionBackpropDataOp Operation '{0}'", origOp->getLoc());

    // Support filter pattern:
    //     Const::DeclareOp
    //             |
    //    (IE::FakeQuantizeOp)
    //             |
    //     (IE::ConvertOp)
    //             |
    // IE::ConvolutionBackpropDataOp

    auto filterTensor = origOp.getFilter();
    auto filterConvertOp = filterTensor.getDefiningOp<IE::ConvertOp>();
    if (filterConvertOp != nullptr) {
        filterTensor = filterConvertOp.getInput();
    }

    auto filterFqOp = filterTensor.getDefiningOp<IE::FakeQuantizeOp>();
    if (filterFqOp != nullptr) {
        filterTensor = filterFqOp.getInput();
    }

    auto filterOp = filterTensor.getDefiningOp<Const::DeclareOp>();
    if (filterOp == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to find filter constant operation");
    }

    // Reverse IC and OC dimensions in filter constant:
    //   [IC, OC, X] -> [OC, IC, X]
    //   [IC, OC, Y, X] -> [OC, IC, Y, X]
    //   [IC, OC, Z, Y, X] -> [OC, IC, Z, Y, X]
    const auto getNewFilterDimsOrder = [](const int64_t rank) {
        SmallVector<vpux::Dim> permutation{};
        if (rank == 3) {
            permutation = {Dim(1), Dim(0), Dim(2)};
        } else if (rank == 4) {
            permutation = {Dim(1), Dim(0), Dim(2), Dim(3)};
        } else if (rank == 5) {
            permutation = {Dim(1), Dim(0), Dim(2), Dim(3), Dim(4)};
        }
        return DimsOrder::fromPermutation(permutation);
    };

    auto filterType = filterOp.getType().cast<NDTypeInterface>();
    auto newDimsOrder = getNewFilterDimsOrder(filterType.getRank());
    auto filterDimOC = Dim(1);
    auto contentAttr = filterOp.getContentAttr().reverse(filterDimOC).transpose(newDimsOrder);
    auto content = contentAttr.fold();
    auto contentType = content.getType();

    // It is necessary to copy the contents of the folded constant into a new buffer since the data conversion
    // does not take place until the values are extracted (e.g. inside `Const::Content::copyTo`)
    const auto contentSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
    std::vector<char> newContent(contentSize);
    content.copyTo(MutableArrayRef(newContent.data(), contentSize));
    const auto foldedBaseContent =
            mlir::DenseElementsAttr::getFromRawBuffer(mlir::cast<mlir::ShapedType>(contentType), ArrayRef(newContent));
    auto newContentAttr = Const::ContentAttr::get(foldedBaseContent);

    auto filterShape = to_small_vector(filterType.getShape());
    std::swap(filterShape[Dims4D::Filter::OC.ind()], filterShape[Dims4D::Filter::IC.ind()]);
    auto newFilterType = filterType.changeShape(Shape(filterShape));

    auto newFilterConstant = rewriter.create<Const::DeclareOp>(filterOp.getLoc(), newFilterType, newContentAttr);
    auto newFilter = newFilterConstant.getOutput();

    const auto transposeFqInput = [&](mlir::Value fqInput) -> mlir::Value {
        auto fqInputType = fqInput.getType().cast<NDTypeInterface>();
        if (fqInputType.getNumElements() == 1) {
            return fqInput;
        }

        auto permutation = to_small_vector(fqInputType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[Dims4D::Filter::OC.ind()], permutation[Dims4D::Filter::IC.ind()]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(filterOp.getLoc(), fqInput, /*order=*/nullptr, orderAttr);
        return transposeOp.getOutput();
    };

    if (filterFqOp != nullptr) {
        // In case the filter is quantized per-axis, make sure the axes are also correct for the new filter
        auto inputLow = transposeFqInput(filterFqOp.getInputLow());
        auto inputHigh = transposeFqInput(filterFqOp.getInputHigh());
        auto outputLow = transposeFqInput(filterFqOp.getOutputLow());
        auto outputHigh = transposeFqInput(filterFqOp.getOutputHigh());
        newFilter = rewriter.createOrFold<IE::FakeQuantizeOp>(
                filterFqOp.getLoc(), newFilter, inputLow, inputHigh, outputLow, outputHigh, filterFqOp.getLevelsAttr(),
                filterFqOp.getLowFpTypeAttr(), filterFqOp.getAutoBroadcastAttr());
    }

    if (filterConvertOp != nullptr) {
        newFilter = rewriter.createOrFold<IE::ConvertOp>(filterConvertOp.getLoc(), newFilter,
                                                         filterConvertOp.getDstElemTypeAttr());
    }

    rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
            origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), /*bias*/ nullptr, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr);

    return mlir::success();
}

//
// GroupConvolutionBackpropDataConversion
//

class GroupConvolutionBackpropDataConversion final : public mlir::OpRewritePattern<IE::GroupConvolutionBackpropDataOp> {
public:
    GroupConvolutionBackpropDataConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionBackpropDataOp>(ctx), _log(log) {
        setDebugName("GroupConvolutionBackpropDataConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionBackpropDataOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupConvolutionBackpropDataConversion::matchAndRewrite(IE::GroupConvolutionBackpropDataOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::GroupConvolutionBackpropDataOp Operation '{0}'", origOp->getLoc());

    // Support filter pattern:
    //     Const::DeclareOp
    //             |
    //    (IE::FakeQuantizeOp)
    //             |
    //     (IE::ConvertOp)
    //             |
    // IE::ConvolutionBackpropDataOp

    auto filterTensor = origOp.getFilter();
    auto filterConvertOp = filterTensor.getDefiningOp<IE::ConvertOp>();
    if (filterConvertOp != nullptr) {
        filterTensor = filterConvertOp.getInput();
    }

    auto filterFqOp = filterTensor.getDefiningOp<IE::FakeQuantizeOp>();
    if (filterFqOp != nullptr) {
        filterTensor = filterFqOp.getInput();
    }

    auto filterOp = filterTensor.getDefiningOp<Const::DeclareOp>();
    if (filterOp == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to find filter constant operation");
    }

    // Reverse IC and OC dimensions in filter constant:
    //   [GROUPS, IC, OC, X] -> [GROUPS, OC, IC, X]
    //   [GROUPS, IC, OC, Y, X] -> [GROUPS, OC, IC, Y, X]
    //   [GROUPS, IC, OC, Z, Y, X] -> [GROUPS, OC, IC, Z, Y, X]
    const auto getNewFilterDimsOrder = [](const int64_t rank) {
        SmallVector<vpux::Dim> permutation{};
        if (rank == 4) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3)};
        } else if (rank == 5) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3), Dim(4)};
        } else if (rank == 6) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3), Dim(4), Dim(5)};
        }
        return DimsOrder::fromPermutation(permutation);
    };
    auto filterType = filterOp.getType().cast<NDTypeInterface>();
    auto newDimsOrder = getNewFilterDimsOrder(filterType.getRank());
    auto filterDimOC = Dim(2);
    auto contentAttr = filterOp.getContentAttr().reverse(filterDimOC).transpose(newDimsOrder);
    auto content = contentAttr.fold();
    auto contentType = content.getType();

    // It is necessary to copy the contents of the folded constant into a new buffer since the data conversion
    // does not take place until the values are extracted (e.g. inside `Const::Content::copyTo`)
    const auto contentSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
    std::vector<char> newContent(contentSize);
    content.copyTo(MutableArrayRef(newContent.data(), contentSize));
    const auto foldedBaseContent =
            mlir::DenseElementsAttr::getFromRawBuffer(mlir::cast<mlir::ShapedType>(contentType), ArrayRef(newContent));
    auto newContentAttr = Const::ContentAttr::get(foldedBaseContent);

    auto filterShape = to_small_vector(filterType.getShape());
    std::swap(filterShape[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
              filterShape[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX]);
    auto newFilterType = filterType.changeShape(Shape(filterShape));

    auto newFilterConstant = rewriter.create<Const::DeclareOp>(filterOp.getLoc(), newFilterType, newContentAttr);
    auto newFilter = newFilterConstant.getOutput();

    const auto transposeFqInput = [&](mlir::Value fqInput) -> mlir::Value {
        auto fqInputType = fqInput.getType().cast<NDTypeInterface>();
        if (fqInputType.getNumElements() == 1) {
            return fqInput;
        }

        auto permutation = to_small_vector(fqInputType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
                  permutation[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(filterOp.getLoc(), fqInput, /*order=*/nullptr, orderAttr);
        return transposeOp.getOutput();
    };

    if (filterFqOp != nullptr) {
        // In case the filter is quantized per-axis, make sure the axes are also correct for the new filter
        auto inputLow = transposeFqInput(filterFqOp.getInputLow());
        auto inputHigh = transposeFqInput(filterFqOp.getInputHigh());
        auto outputLow = transposeFqInput(filterFqOp.getOutputLow());
        auto outputHigh = transposeFqInput(filterFqOp.getOutputHigh());
        newFilter = rewriter.createOrFold<IE::FakeQuantizeOp>(
                filterFqOp.getLoc(), newFilter, inputLow, inputHigh, outputLow, outputHigh, filterFqOp.getLevelsAttr(),
                filterFqOp.getLowFpTypeAttr(), filterFqOp.getAutoBroadcastAttr());
    }

    if (filterConvertOp != nullptr) {
        newFilter = rewriter.createOrFold<IE::ConvertOp>(filterConvertOp.getLoc(), newFilter,
                                                         filterConvertOp.getDstElemTypeAttr());
    }

    rewriter.replaceOpWithNewOp<IE::GroupTransposedConvolutionOp>(
            origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr);

    return mlir::success();
}

//
// ConvertConvBackpropDataToTransposedConvPass
//

class ConvertConvBackpropDataToTransposedConvPass final :
        public IE::ConvertConvBackpropDataToTransposedConvBase<ConvertConvBackpropDataToTransposedConvPass> {
public:
    explicit ConvertConvBackpropDataToTransposedConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertConvBackpropDataToTransposedConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::ConvolutionBackpropDataOp>();
    target.addIllegalOp<IE::GroupConvolutionBackpropDataOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::GroupTransposedConvolutionOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::TransposedConvolutionOp>();
    target.addLegalOp<IE::ConvertOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvolutionBackpropDataConversion>(&ctx, _log);
    patterns.add<GroupConvolutionBackpropDataConversion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertConvBackpropDataToTransposedConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertConvBackpropDataToTransposedConvPass(Logger log) {
    return std::make_unique<ConvertConvBackpropDataToTransposedConvPass>(log);
}

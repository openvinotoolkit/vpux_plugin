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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// getAxes
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getAxes(IE::UnsqueezeOpAdaptor unsqueeze, mlir::Location loc) {
    if (unsqueeze.axes() != nullptr && unsqueeze.axes_value() != nullptr) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (unsqueeze.axes() == nullptr && unsqueeze.axes_value() == nullptr) {
        return errorAt(loc, "Missed axes representation");
    }

    if (unsqueeze.axes_value() != nullptr) {
        return parseIntArrayAttr<int64_t>(unsqueeze.axes_value());
    }

    auto axesConst = unsqueeze.axes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.content();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());
    std::sort(axes.begin(), axes.end());

    const auto inType = unsqueeze.input().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();
    const auto numAxes = checked_cast<int64_t>(axes.size());

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank + numAxes;
        }
    }

    return axes;
}

//
// inferOutputLayout
//

DimsOrder inferOutputLayout(const DimArr& inPerm, const SmallVector<int64_t>& axes) {
    SmallVector<vpux::Dim> perm;

    // Iterate over input dims in the given order and push back corresponding output dims.
    for (const auto& p : inPerm) {
        auto dim = p.ind();
        for (const auto& unsqueezedAxis : axes) {
            if (dim > unsqueezedAxis) {
                dim++;
            } else if (dim == unsqueezedAxis) {
                perm.push_back(vpux::Dim(dim));
                dim++;
            }
        }

        perm.push_back(vpux::Dim(dim));
    }

    // If unsqueezed 1s are at the end, push their corresponding axes in the perm vec
    const auto sz = static_cast<int64_t>(perm.size());
    for (const auto& unsqueezedAxis : axes) {
        if (unsqueezedAxis >= sz) {
            perm.push_back(vpux::Dim(unsqueezedAxis));
        }
    }

    return DimsOrder::fromPermutation(makeArrayRef(perm));
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::UnsqueezeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::UnsqueezeOpAdaptor unsqueeze(operands, attrs);
    if (mlir::failed(unsqueeze.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = getAxes(unsqueeze, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto input = unsqueeze.input();
    const auto inType = input.getType().cast<mlir::RankedTensorType>();
    const auto inShape = inType.getShape();
    const auto inOrder = DimsOrder::fromValue(input);

    SmallVector<int64_t> outShape(inShape.size() + axes->size());

    size_t inInd = 0;
    size_t axesInd = 0;
    for (auto outInd : irange(outShape.size())) {
        if (axesInd < axes.getValue().size()) {
            const auto nextAxisInd = checked_cast<size_t>(axes.getValue()[axesInd]);

            if (nextAxisInd < outInd) {
                return errorAt(loc, "Axis '{0}' was occured twice", nextAxisInd);
            }

            if (nextAxisInd == outInd) {
                outShape[outInd] = 1;
                ++axesInd;
                continue;
            }
        }

        if (inInd < inShape.size()) {
            outShape[outInd] = inShape[inInd];
            ++inInd;
            continue;
        }
    }
    if (inInd != inShape.size() || axesInd != axes->size()) {
        return errorAt(loc, "Inconsistent parameters");
    }

    const auto outDesc = IE::getTensorAttr(ctx, inferOutputLayout(inOrder.toPermutation(), axes.getValue()),
                                           IE::getMemorySpace(inType), IE::isSparse(inType));

    inferredReturnShapes.emplace_back(makeArrayRef(outShape), inType.getElementType(), outDesc);
    return mlir::success();
}

//
// inferLayoutInfo
//

void vpux::IE::UnsqueezeOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto axes = parseIntArrayAttr<int64_t>(axes_value().getValue());
    const auto inOrder = info.getInput(0);
    const auto inPermutation = inOrder.toPermutation();

    info.setInput(0, inOrder);
    info.setOutput(0, inferOutputLayout(inPermutation, axes));
}

//
// fold
//

mlir::OpFoldResult vpux::IE::UnsqueezeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.reshape(getShape(output()));
    }

    return nullptr;
}

//
// FuseWithReshape
//

namespace {

class FuseWithReshape final : public mlir::OpRewritePattern<IE::UnsqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::UnsqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseWithReshape::matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.input().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::UnsqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::UnsqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.axes_value().hasValue()) {
        return mlir::failure();
    }

    const auto axes = getAxes(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto axesAttr = getIntArrayAttr(getContext(), axes.getValue());

    rewriter.replaceOpWithNewOp<IE::UnsqueezeOp>(origOp, origOp.input(), nullptr, axesAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::UnsqueezeOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                        mlir::MLIRContext* context) {
    patterns.insert<FuseWithReshape>(context);
    patterns.insert<ConvertConstToAttr>(context);
}

std::unique_ptr<ngraph::Node> vpux::IE::UnsqueezeOp::toNgraph(ngraph::OutputVector &outputs)
{
    return std::make_unique<opset_latest::Unsqueeze>(outputs.at(0), outputs.at(1));
}

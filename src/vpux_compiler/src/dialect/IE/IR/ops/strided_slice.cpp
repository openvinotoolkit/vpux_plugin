//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <optional>

using namespace vpux;

namespace {

struct StridedSliceInputData final {
    SmallVector<int64_t> begins;
    SmallVector<int64_t> ends;
    SmallVector<int64_t> strides;
};

StridedSliceInputData extractData(mlir::Location loc, IE::StridedSliceOpAdaptor stridedSlice) {
    if (stridedSlice.getBegins() != nullptr) {
        auto begins = IE::constInputToData(loc, stridedSlice.getBegins());
        auto ends = IE::constInputToData(loc, stridedSlice.getEnds());
        auto strides = IE::constInputToData(loc, stridedSlice.getStrides());

        return StridedSliceInputData{mlir::succeeded(begins) ? begins.value() : SmallVector<int64_t>{},
                                     mlir::succeeded(ends) ? ends.value() : SmallVector<int64_t>{},
                                     mlir::succeeded(strides) ? strides.value() : SmallVector<int64_t>{}};
    }

    if (stridedSlice.getBeginsAttr().has_value()) {
        return StridedSliceInputData{stridedSlice.getBeginsAttr().has_value()
                                             ? parseIntArrayAttr<int64_t>(stridedSlice.getBeginsAttr().value())
                                             : SmallVector<int64_t>{},
                                     stridedSlice.getEndsAttr().has_value()
                                             ? parseIntArrayAttr<int64_t>(stridedSlice.getEndsAttr().value())
                                             : SmallVector<int64_t>{},
                                     stridedSlice.getStridesAttr().has_value()
                                             ? parseIntArrayAttr<int64_t>(stridedSlice.getStridesAttr().value())
                                             : SmallVector<int64_t>{}};
    }

    return StridedSliceInputData{{}, {}, {}};
}

}  // namespace

mlir::LogicalResult vpux::IE::StridedSliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::StridedSliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = slice.getInput().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();

    const auto inputData = extractData(loc, slice);
    const auto beginsShape =
            slice.getBegins() != nullptr
                    ? SmallVector<int64_t>(slice.getBegins().getType().cast<mlir::ShapedType>().getShape())
                    : SmallVector<int64_t>{};
    const auto endsShape = slice.getEnds() != nullptr
                                   ? SmallVector<int64_t>(slice.getEnds().getType().cast<mlir::ShapedType>().getShape())
                                   : SmallVector<int64_t>{};
    const auto stridesShape =
            slice.getStrides() != nullptr
                    ? SmallVector<int64_t>(slice.getStrides().getType().cast<mlir::ShapedType>().getShape())
                    : SmallVector<int64_t>{};

    const auto beginMask = parseIntArrayAttr<int64_t>(slice.getBeginMask());
    const auto endMask = parseIntArrayAttr<int64_t>(slice.getEndMask());
    const auto newAxisMask = parseIntArrayAttr<int64_t>(slice.getNewAxisMask());
    const auto shrinkAxisMask = parseIntArrayAttr<int64_t>(slice.getShrinkAxisMask());
    const auto ellipsisMask = parseIntArrayAttr<int64_t>(slice.getEllipsisMask());

    auto outputShapeInfo = inferStridedSliceOutputShape(inDataShape, inputData.begins, inputData.ends,
                                                        inputData.strides, beginsShape, endsShape, stridesShape,
                                                        beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    auto outputShape = outputShapeInfo.shape;
    if (outputShape.empty()) {
        outputShape.push_back(1);
    }

    auto inType = slice.getInput().getType().cast<NDTypeInterface>();
    const auto outType = inType.changeShape(Shape(outputShape)).cast<mlir::RankedTensorType>();

    const auto inDataTensorType = slice.getInput().getType().cast<mlir::RankedTensorType>();
    mlir::ArrayAttr outBoundsAttr =
            !outputShapeInfo.bounds.empty() ? getIntArrayAttr(ctx, outputShapeInfo.bounds) : nullptr;
    const auto outDesc = vpux::getTensorAttr(vpux::getOrder(inDataTensorType), /*memSpace=*/nullptr, outBoundsAttr);

    inferredReturnShapes.emplace_back(outType.getShape(), outType.getElementType(), outDesc);

    return mlir::success();
}

bool vpux::IE::StridedSliceOp::isSimplified() {
    auto isZero = [](auto val) {
        return val == 0;
    };
    auto isPositive = [](auto val) {
        return val >= 0;
    };

    return (llvm::all_of(parseIntArrayAttr<int64_t>(getNewAxisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getShrinkAxisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEllipsisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginsAttr().value()), isPositive) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndsAttr().value()), isPositive));
}

bool hasNegativeStride(ArrayRef<int64_t> strides) {
    auto isNegative = [](auto val) {
        return val < 0;
    };
    return (llvm::any_of(strides, isNegative));
}
//
// fold
//

mlir::OpFoldResult vpux::IE::StridedSliceOp::fold(FoldAdaptor) {
    auto strides = getStridesAttr();
    if (!strides.has_value()) {
        return nullptr;
    }
    // In case StridedSlice op has negative value in stride input, which means reversing
    // the data along the axis, we can not fold for such case.
    if (!hasNegativeStride(parseIntArrayAttr<int64_t>(strides.value())) &&
        getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    // TODO E#22568: attempt const folding but only if slice isSimplified()

    return nullptr;
}

//
// ComposeStridedSlice
//

namespace {

class ComposeStridedSlice final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeStridedSlice::matchAndRewrite(IE::StridedSliceOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto producerSliceOp = origOp.getInput().getDefiningOp<IE::StridedSliceOp>();
    if (producerSliceOp == nullptr) {
        return mlir::failure();
    }

    if (!(origOp.isSimplified() && producerSliceOp.isSimplified())) {
        return mlir::failure();
    }

    const auto firstBegin = parseIntArrayAttr<int64_t>(producerSliceOp.getBeginsAttr().value());
    const auto nextBegin = parseIntArrayAttr<int64_t>(origOp.getBeginsAttr().value());
    auto resultBegin = SmallVector<int64_t>(nextBegin.size());

    const auto firstEnd = parseIntArrayAttr<int64_t>(producerSliceOp.getEndsAttr().value());
    const auto nextEnd = parseIntArrayAttr<int64_t>(origOp.getEndsAttr().value());
    auto resultEnd = SmallVector<int64_t>(nextEnd.size());

    const auto firstStride = parseIntArrayAttr<int64_t>(producerSliceOp.getStridesAttr().value());
    const auto nextStride = parseIntArrayAttr<int64_t>(origOp.getStridesAttr().value());
    auto resultStride = SmallVector<int64_t>(nextStride.size());

    for (auto i : irange(firstBegin.size())) {
        resultBegin[i] = firstBegin[i] + nextBegin[i] * firstStride[i];
        resultEnd[i] = firstBegin[i] + nextEnd[i] * firstStride[i];
        resultStride[i] = firstStride[i] * nextStride[i];
    }

    // Stride on more than 2 axis is not supported
    const auto greaterThanOne = [](auto ele) {
        return ele > 1;
    };
    auto stridesGreaterThanOne = llvm::count_if(resultStride, greaterThanOne);
    if (stridesGreaterThanOne > 2) {
        return mlir::failure();
    }

    const auto beginsAttr = getIntArrayAttr(getContext(), resultBegin);
    const auto endsAttr = getIntArrayAttr(getContext(), resultEnd);
    const auto stridesAttr = getIntArrayAttr(getContext(), resultStride);

    const auto fusedLoc =
            mlir::FusedLoc::get(producerSliceOp->getLoc().getContext(), {producerSliceOp->getLoc(), origOp->getLoc()});
    const auto newOp = rewriter.create<IE::StridedSliceOp>(
            fusedLoc, producerSliceOp.getInput(), origOp.getBegins(), origOp.getEnds(), origOp.getStrides(), beginsAttr,
            endsAttr, stridesAttr, origOp.getBeginMask(), origOp.getEndMask(), origOp.getNewAxisMask(),
            origOp.getShrinkAxisMask(), origOp.getEllipsisMask());
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using mlir::OpRewritePattern<IE::StridedSliceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp slice, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::StridedSliceOp slice,
                                                        mlir::PatternRewriter& rewriter) const {
    if (!slice.getBegins() || !slice.getEnds() || !slice.getStrides()) {
        return mlir::failure();
    }

    const auto inputData = extractData(slice.getLoc(), slice);
    const auto beginsAttr = !inputData.begins.empty() ? getIntArrayAttr(getContext(), inputData.begins) : nullptr;
    const auto endsAttr = !inputData.ends.empty() ? getIntArrayAttr(getContext(), inputData.ends) : nullptr;
    const auto stridesAttr = !inputData.strides.empty() ? getIntArrayAttr(getContext(), inputData.strides) : nullptr;

    const auto begins = beginsAttr == nullptr ? slice.getBegins() : mlir::TypedValue<mlir::RankedTensorType>{nullptr};
    const auto ends = endsAttr == nullptr ? slice.getEnds() : mlir::TypedValue<mlir::RankedTensorType>{nullptr};
    const auto strides =
            stridesAttr == nullptr ? slice.getStrides() : mlir::TypedValue<mlir::RankedTensorType>{nullptr};

    rewriter.replaceOpWithNewOp<IE::StridedSliceOp>(
            slice, slice.getInput(), begins, ends, strides, beginsAttr, endsAttr, stridesAttr, slice.getBeginMask(),
            slice.getEndMask(), slice.getNewAxisMask(), slice.getShrinkAxisMask(), slice.getEllipsisMask());
    return mlir::success();
}

Const::DeclareOp createSeqLenConst(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, mlir::Location loc,
                                   ArrayRef<int64_t> seqLen) {
    auto intType = getSInt64Type(ctx);
    const auto dataStorageType = mlir::RankedTensorType::get({static_cast<int64_t>(seqLen.size())}, intType);
    const auto dataDenseAttr = mlir::DenseElementsAttr::get(dataStorageType, seqLen);
    auto newContentAttr = Const::ContentAttr::get(dataDenseAttr).convertElemType(getSInt32Type(ctx));
    return rewriter.create<Const::DeclareOp>(loc, dataStorageType, newContentAttr);
}

//
// ConvertNegStrideStridedSlice2Reverse
//

class ConvertNegStrideStridedSlice2Reverse final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using mlir::OpRewritePattern<IE::StridedSliceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp stridedSliceOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertNegStrideStridedSlice2Reverse::matchAndRewrite(IE::StridedSliceOp slice,
                                                                          mlir::PatternRewriter& rewriter) const {
    if (!slice.getBegins() || !slice.getEnds() || !slice.getStrides()) {
        return mlir::failure();
    }

    const auto inputData = extractData(slice.getLoc(), slice);

    auto strides = inputData.strides;
    if (!hasNegativeStride(strides)) {
        return mlir::failure();
    }

    // If the data is reversed all along 1 axis, the StridedSlice can be replaced by ReverseSequenceOp
    const auto inDataType = slice.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inDataShape = inDataType.getShape().raw();
    const auto begins = inputData.begins;
    const auto ends = inputData.ends;
    SmallVector<int64_t> seqLen;
    int64_t seqAxis = 0;
    for (const auto& p : strides | indexed) {
        if (p.value() < 0) {
            if (std::abs(ends[p.index()]) < inDataShape[p.index()] || begins[p.index()] != -1) {
                return mlir::failure();
            }
            seqAxis = p.index();
            seqLen.push_back(inDataShape[p.index()]);
        }
    }

    auto seqLenData = createSeqLenConst(rewriter, rewriter.getContext(), slice.getLoc(), seqLen);
    rewriter.replaceOpWithNewOp<IE::ReverseSequenceOp>(slice, slice.getInput(), seqLenData,
                                                       getIntAttr(rewriter.getContext(), seqAxis),
                                                       getIntAttr(rewriter.getContext(), 0));
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::StridedSliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                           mlir::MLIRContext* context) {
    patterns.add<ConvertNegStrideStridedSlice2Reverse>(context);
    patterns.add<ConvertConstToAttr>(context);
    patterns.add<ComposeStridedSlice>(context);
}

//
// inferElemTypeInfo
//

void vpux::IE::StridedSliceOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeDown(info);
}

void vpux::IE::StridedSliceOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeUp(info);
}

//
// verify
//

mlir::LogicalResult vpux::IE::StridedSliceOp::verify() {
    const auto arch = VPU::getArch(*this);
    const auto isConst = [](mlir::Value value) {
        return value == nullptr || value.getDefiningOp<Const::DeclareOp>() != nullptr;
    };
    // [E#103473]: extract the arch check from verify
    const auto nonConstIndeces = !isConst(getBegins()) || !isConst(getEnds()) || !isConst(getStrides());
    if (arch == VPU::ArchKind::NPU30XX && nonConstIndeces) {
        return errorAt(*this, "Non-constant begins, ends, strides are not supported for VPUX30XX");
    }

    auto beginsAttr = getBeginsAttr();
    auto endsAttr = getEndsAttr();
    auto stridesAttr = getStridesAttr();

    if (beginsAttr.has_value() && endsAttr.has_value()) {
        if (beginsAttr.value().size() != endsAttr.value().size()) {
            return errorAt(*this, "Lower bounds and Upper bounds needs to have same number of values");
        }
    }

    if (beginsAttr.has_value() && stridesAttr.has_value()) {
        if (beginsAttr.value().size() != stridesAttr.value().size()) {
            return errorAt(*this, "Lower bounds and strides needs to have same number of values");
        }
    }

    if (endsAttr.has_value() && stridesAttr.has_value()) {
        if (endsAttr.value().size() != stridesAttr.value().size()) {
            return errorAt(*this, "Upper bounds and strides needs to have same number of values");
        }
    }

    return mlir::success();
}

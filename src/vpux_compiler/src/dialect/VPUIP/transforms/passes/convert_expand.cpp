//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

namespace {

// Helper class to wrap the arguments for ExpandConverter::applyPadding
class PaddingContext {
public:
    PaddingContext(const mlir::Location loc, const ShapeRef inShape, const mlir::Value expandedBuffer,
                   const mlir::Value constantBuffer)
            : _loc(loc), _inShape(inShape), _expandedBuffer(expandedBuffer), _constantBuffer(constantBuffer){};
    PaddingContext(const PaddingContext&) = delete;
    PaddingContext(const PaddingContext&&) = delete;
    PaddingContext& operator=(const PaddingContext&) = delete;
    PaddingContext& operator=(const PaddingContext&&) = delete;

    const mlir::Location _loc;
    ShapeRef _inShape;
    const mlir::Value _expandedBuffer;
    const mlir::Value _constantBuffer;
};

//
// ConvertExpandPass
//

class ConvertExpandPass final : public VPUIP::ConvertExpandBase<ConvertExpandPass> {
public:
    explicit ConvertExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    mlir::Value applyPadding(const int64_t padAxis, const int64_t padValue, ArrayRef<int64_t> inSubViewOffsets,
                             const PaddingContext& padCtx, mlir::OpBuilder& builder) const;

    std::pair<int64_t, int64_t> getMaxExpandConstShapeForFp16AndU8(mlir::func::FuncOp func, Logger log);
    std::pair<Const::DeclareOp, Const::DeclareOp> getZeroConstOpsForFp16AndU8(mlir::func::FuncOp func,
                                                                              mlir::MLIRContext& ctx,
                                                                              mlir::OpBuilder& builder);

    Dim getPadDim(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType);
};

mlir::Value ConvertExpandPass::applyPadding(const int64_t padAxis, const int64_t padValue,
                                            ArrayRef<int64_t> inSubViewOffsets, const PaddingContext& padCtx,
                                            mlir::OpBuilder& builder) const {
    const auto& location = padCtx._loc;
    const auto& inShape = padCtx._inShape;
    const auto& expandedBuffer = padCtx._expandedBuffer;
    const auto& constantBuffer = padCtx._constantBuffer;
    SmallVector<int64_t> subViewOffsets;
    std::copy(inSubViewOffsets.begin(), inSubViewOffsets.end(), std::back_inserter(subViewOffsets));

    auto constantOp = constantBuffer.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(constantOp != nullptr, "Can not get constant Op");

    const auto constShapeType = constantOp.getOutput().getType().cast<NDTypeInterface>();
    const auto constOuputShape = constShapeType.getShape();
    Shape subViewShape;
    std::copy(inShape.begin(), inShape.end(), std::back_inserter(subViewShape));
    subViewShape[Dim(padAxis)] = padValue;
    VPUX_THROW_UNLESS(subViewShape.totalSize() <= constOuputShape.totalSize(),
                      "Constant subview shape size '{0}' large than full size '{1}'", subViewShape.totalSize(),
                      constOuputShape.totalSize());

    // Step 1: Create SubView Op to get the right constant size
    VPUX_THROW_UNLESS(constOuputShape.size() == 1, "Constant Op unexpect shape size '{0}'", constOuputShape);
    const auto constSubviewOffset = SmallVector<int64_t>(1, 0);
    const auto constSubviewShape = SmallVector<int64_t>(1, subViewShape.totalSize());
    auto constSubView =
            builder.create<VPUIP::SubViewOp>(appendLoc(location, "_constant_subview_{0}_{1}", padAxis, padValue),
                                             constantOp, constSubviewOffset, constSubviewShape);

    // Step 2: Create Reshape Op to match concat shape
    const auto shapeType = constSubView.getType().cast<vpux::NDTypeInterface>();
    const auto newShapeType = shapeType.changeShape(subViewShape);
    auto reshapeOp =
            builder.create<VPUIP::GenericReshapeOp>(appendLoc(location, "_constant_reshape_{0}_{1}", padAxis, padValue),
                                                    newShapeType, constSubView.getResult());

    // Step 3: Creat PermuteCast Op to match concat layout
    const auto expandOutBufferType = expandedBuffer.getType().cast<NDTypeInterface>();
    const auto newLayoutType = newShapeType.changeDimsOrder(expandOutBufferType.getDimsOrder());
    const auto dstOrderAttr =
            mlir::AffineMapAttr::get(expandOutBufferType.getDimsOrder().toAffineMap(reshapeOp.getContext()));
    const auto memPermAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(reshapeOp.getContext()));
    auto permuteCastOp =
            builder.create<VPUIP::PermuteCastOp>(appendLoc(location, "_constant_permute_{0}_{1}", padAxis, padValue),
                                                 newLayoutType, reshapeOp.getOutput(), dstOrderAttr, memPermAttr);

    // Step 4: Create Copy Op for concat concatant input
    auto subView = builder.create<VPUIP::SubViewOp>(appendLoc(location, "_expand_subview_{0}_{1}", padAxis, padValue),
                                                    expandedBuffer, Shape(subViewOffsets), subViewShape);
    auto subViewCopy = builder.create<VPUIP::CopyOp>(appendLoc(location, "_expand_copy_{0}_{1}", padAxis, padValue),
                                                     permuteCastOp.getResult(), subView);

    return subViewCopy.getOutput();
}

std::pair<int64_t, int64_t> ConvertExpandPass::getMaxExpandConstShapeForFp16AndU8(mlir::func::FuncOp func, Logger log) {
    int64_t maxFP16ShapeSize = 0;
    int64_t maxINT8ShapeSize = 0;

    func->walk([&](VPUIP::ExpandOp origOp) {
        auto inShape = getShape(origOp.getInput());
        auto outShape = getShape(origOp.getOutput());
        VPUX_THROW_UNLESS(outShape.totalSize() > inShape.totalSize(),
                          "Unexpect Expand input shape '{0}' output shape '{1}'", inShape, outShape);
        const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
        auto diffShapeSize = outShape.totalSize() - inShape.totalSize();
        if (inputType.getElementType().isa<mlir::Float16Type>()) {
            maxFP16ShapeSize = std::max(checked_cast<int64_t>(diffShapeSize), maxFP16ShapeSize);
        } else if (inputType.getElemTypeSize().count() == 8) {
            maxINT8ShapeSize = std::max(checked_cast<int64_t>(diffShapeSize), maxINT8ShapeSize);
        } else {
            log.trace("Unexpected Expand '{0}' with input type '{1}'", origOp->getLoc(), inputType.getElementType());
        }
        log.trace("Found Expand Operation '{0}' inshape '{1}' outshape '{2}', FP16 constant maxShapeSize is '{3}', "
                  "INT8 constant maxShapeSize is '{4}'",
                  origOp->getLoc(), inShape, outShape, maxFP16ShapeSize, maxINT8ShapeSize);
    });

    return std::make_pair(maxFP16ShapeSize, maxINT8ShapeSize);
}

std::pair<Const::DeclareOp, Const::DeclareOp> ConvertExpandPass::getZeroConstOpsForFp16AndU8(mlir::func::FuncOp func,
                                                                                             mlir::MLIRContext& ctx,
                                                                                             mlir::OpBuilder& builder) {
    const auto constantShapeSize = getMaxExpandConstShapeForFp16AndU8(func, _log);
    Const::DeclareOp constantFP16Op = nullptr;
    Const::DeclareOp constantQuantizeOp = nullptr;
    if (constantShapeSize.first != 0) {
        const auto dataFP16StorageType =
                mlir::RankedTensorType::get({constantShapeSize.first}, mlir::Float16Type::get(&ctx));
        const auto denseFP16ElementVal =
                mlir::DenseElementsAttr::get(dataFP16StorageType, checked_cast<vpux::type::float16>(0.f));

        constantFP16Op = builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(&ctx),
                                                          vpux::convertToMemRef(dataFP16StorageType),
                                                          Const::ContentAttr::get(denseFP16ElementVal));
    }

    if (constantShapeSize.second != 0) {
        const auto dataQuantizeStorageType = mlir::RankedTensorType::get(constantShapeSize.second, getUInt8Type(&ctx));
        const auto denseINT8ElementVal =
                mlir::DenseElementsAttr::get(dataQuantizeStorageType, checked_cast<uint8_t>(0));

        constantQuantizeOp = builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(&ctx),
                                                              vpux::convertToMemRef(dataQuantizeStorageType),
                                                              Const::ContentAttr::get(denseINT8ElementVal));
    }

    return std::make_pair(constantFP16Op, constantQuantizeOp);
}

Dim ConvertExpandPass::getPadDim(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto ioShapes = zip(inShape, outShape);
    const auto dimDiffPredicate = [](const std::tuple<int64_t, int64_t>& ioDims) -> bool {
        const auto& inDim = std::get<0>(ioDims);
        const auto& outDim = std::get<1>(ioDims);
        return inDim != outDim;
    };

    const auto diffAxisIter = std::find_if(ioShapes.begin(), ioShapes.end(), dimDiffPredicate);
    VPUX_THROW_UNLESS(diffAxisIter != ioShapes.end(), "Expand inShape '{0}' same with the outShape '{1}'", inShape,
                      outShape);

    const auto padAxis = std::distance(ioShapes.begin(), diffAxisIter);
    return Dim(padAxis);
}

//
// safeRunOnFunc
//

void ConvertExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::OpBuilder builder(&func.getBody().front().front());

    // For Expand(FP16) and Expand(INT8) with PadsBegin, replace the op with concat a const op
    //     input                input      const
    //       |                    \          /
    //     Expand         =>         Concat
    //       |                         |
    // Note that only create one largest Constant Op and reuse for all Expand layers in the model
    // Always Set this Constant with 1D shape size, it is convenient to reshape for specific Expand
    // For Expand(U8) without PadsBegin, the op will be replaced by single DMA directly in later
    // pass(ConvertToDMA). The DMA solution does not support PadsBegin.

    auto constOp = getZeroConstOpsForFp16AndU8(func, ctx, builder);

    func->walk([&](VPUIP::ExpandOp origOp) {
        _log.trace("Found Expand Operation '{0}'", origOp.getLoc());

        const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
        const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
        const auto elemType = inputType.getElementType();

        auto padBeginCheck = llvm::any_of(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()), [](auto padValue) {
            return padValue != 0;
        });
        if (!elemType.isa<mlir::Float16Type>() && !padBeginCheck) {
            _log.nest().trace(
                    "ExpandOp type should with FP16 precision or INT8 precision with PadsBegin, but got '{0}'",
                    elemType);
            return;
        }

        mlir::Value constOutput;
        if (elemType.isa<mlir::quant::QuantizedType>() && constOp.second != nullptr) {
            constOutput = constOp.second.getOutput();
        } else if (elemType.isa<mlir::Float16Type>() && constOp.first != nullptr) {
            constOutput = constOp.first.getOutput();
        } else {
            _log.nest().trace("unsupported ExpandOp type : '{0}'", elemType);
            return;
        }

        mlir::OpBuilder builder(origOp.getOperation());
        auto newMemRefOutputType = outputType;
        auto expandedBuffer =
                builder.create<mlir::memref::AllocOp>(origOp->getLoc(), newMemRefOutputType.cast<mlir::MemRefType>());

        const auto nonZeroAxisPredicate = [](const int64_t dim) -> bool {
            return dim > 0;
        };

        SmallVector<mlir::Value> concatInputs;
        const auto inShape = inputType.getShape();
        auto subViewOffsets = SmallVector<int64_t>(inShape.size(), 0);
        PaddingContext padCtx(origOp->getLoc(), ShapeRef(inShape), expandedBuffer, constOutput);

        // Apply pads_begin
        _log.nest().trace("Process Expand Operation '{0}' for pads begin", origOp->getLoc());
        const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
        const auto padBeginAxisIter = std::find_if(padsBegin.begin(), padsBegin.end(), nonZeroAxisPredicate);
        if (padBeginAxisIter != padsBegin.end()) {
            const auto padBeginAxis = std::distance(padsBegin.begin(), padBeginAxisIter);
            const auto padValue = padsBegin[padBeginAxis];
            const auto padOut = applyPadding(padBeginAxis, padValue, subViewOffsets, padCtx, builder);
            concatInputs.push_back(padOut);
            subViewOffsets[padBeginAxis] += padValue;
        }

        // Copy the input with offset according to the padding in the beginning
        _log.nest().trace("Process Expand Operation '{0}' for real input data", origOp->getLoc());
        builder.setInsertionPoint(origOp);
        const auto tensorShape = to_small_vector(inShape);
        auto tensorSubView =
                builder.create<VPUIP::SubViewOp>(origOp.getLoc(), expandedBuffer, subViewOffsets, tensorShape);
        auto tensorSubViewCopy = builder.create<VPUIP::CopyOp>(origOp->getLoc(), origOp.getInput(), tensorSubView);

        concatInputs.push_back(tensorSubViewCopy.getOutput());

        // Increment offsets
        const auto padAxis = getPadDim(inputType, outputType);
        subViewOffsets[padAxis.ind()] += tensorShape[padAxis.ind()];

        // Apply pads_end
        _log.nest().trace("Process Expand Operation '{0}' for pads end", origOp->getLoc());
        const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
        const auto padEndAxisIter = std::find_if(padsEnd.begin(), padsEnd.end(), nonZeroAxisPredicate);
        if (padEndAxisIter != padsEnd.end()) {
            const auto padEndAxis = std::distance(padsEnd.begin(), padEndAxisIter);
            const auto padValue = padsEnd[padEndAxis];
            const auto padOut = applyPadding(padEndAxis, padValue, subViewOffsets, padCtx, builder);
            concatInputs.push_back(padOut);
        }

        auto concatViewOp = builder.create<VPUIP::ConcatViewOp>(origOp->getLoc(), concatInputs, expandedBuffer);
        _log.nest().trace("Create ConcatViewOp '{0}'", concatViewOp->getLoc());

        origOp->replaceAllUsesWith(concatViewOp);
        origOp->erase();
    });
}

}  // namespace

//
// createConvertExpandPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertExpandPass(Logger log) {
    return std::make_unique<ConvertExpandPass>(log);
}

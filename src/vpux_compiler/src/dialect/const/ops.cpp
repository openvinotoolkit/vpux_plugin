//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// ConstDialect::materializeConstant
//

mlir::Operation* vpux::Const::ConstDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                                mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType, mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}

//
// ConstDialect: bufferize Const::DeclareOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, Const::DeclareOp origOp, Const::DeclareOp::Adaptor,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-ConstDeclareOp", 0);
    log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    const auto newType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newType, origOp.getContentAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

void vpux::registerConstDeclareBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, vpux::Const::ConstDialect*) {
        Const::DeclareOp::attachInterface<VpuGenericOneShotBufferizeModel<Const::DeclareOp>>(*ctx);
    });
}

//
// DeclareOp::fold
//

mlir::OpFoldResult vpux::Const::DeclareOp::fold(FoldAdaptor adaptor) {
    VPUX_THROW_UNLESS(adaptor.getOperands().empty(), "constant has no operands");
    return getContentAttr();
}

//
// DeclareOp::serialize
//

void vpux::Const::DeclareOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    vpux::Const::Content cnt = getContent();
    // int64_t typeTotalSize = cnt.getRawStorageBuf().size();

    auto ptr = binDataSection.expandData(getBinarySize());
    MutableArrayRef<char> tempBuf(reinterpret_cast<char*>(ptr), reinterpret_cast<char*>(ptr) + getBinarySize());
    cnt.copyTo(tempBuf);
}

//
// DeclareOp::getBinarySize
//

size_t vpux::Const::DeclareOp::getBinarySize() {
    vpux::Const::Content cnt = getContent();

    return cnt.getType().getTotalAllocSize().count();
}

//
// DeclareOp::getAlignmentRequirements
//

size_t vpux::Const::DeclareOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_NO_ALIGNMENT;
}

//
// DeclareOp::getMemorySpace
//

vpux::VPURT::BufferSection vpux::Const::DeclareOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::Constant;
}

//
// DeclareOp::getAccessingProcs
//

vpux::ELFNPU37XX::SectionFlagsAttr vpux::Const::DeclareOp::getAccessingProcs() {
    auto tempFlagsVal = vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE;

    for (auto user : getResult().getUsers()) {
        if (auto binaryIface = mlir::dyn_cast<vpux::ELFNPU37XX::BinaryOpInterface>(user)) {
            tempFlagsVal = tempFlagsVal | binaryIface.getUserProcs();
        }
    }

    return tempFlagsVal;
}

//
// DeclareOp::getUserProcs
//

vpux::ELFNPU37XX::SectionFlagsAttr vpux::Const::DeclareOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

//
// DeclareOp::verify
//

mlir::LogicalResult vpux::Const::DeclareOp::verify() {
    const auto op = getOperation();
    const auto attrType = getContentAttr().getType();
    const auto opType = getType().cast<vpux::NDTypeInterface>();
    // For type with swizzling skip the shape check as the content
    // might have been flattened to accomodate swizzled buffer.
    if (!vpux::getSwizzlingSchemeAttr(opType)) {
        if (opType.getShape() != attrType.getShape()) {
            return errorAt(op, "'Const.Declare' has mismatch in value shape '{0}' and result shape '{1}'",
                           attrType.getShape(), opType.getShape());
        }
    }
    if (opType.getElementType() != attrType.getElementType()) {
        if (!opType.getElementType().isa<mlir::quant::QuantizedType>() &&
            !attrType.getElementType().isa<mlir::IntegerType>()) {
            return errorAt(op, "'Const.Declare' has mismatch in value element type '{0}' and result element type '{1}'",
                           attrType.getElementType(), opType.getElementType());
        }
    }

    const auto attrOrder = attrType.getDimsOrder();
    const auto opOrder = opType.getDimsOrder();

    if (opOrder != attrOrder) {
        return errorAt(op, "'Const.Declare' has mismatch in value DimsOrder '{0}' and result DimsOrder '{1}'",
                       attrOrder, opOrder);
    }

    return mlir::success();
}

//
// DeclareOp::canonicalizer
//

void sendEquivalenceRequest([[maybe_unused]] Const::ContentAttr originalAttr,
                            [[maybe_unused]] Const::ContentAttr newAttr) {
#ifdef BACKGROUND_FOLDING_ENABLED
    auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
    auto ctx = originalAttr.getContext();
    if (cacheManager.contains(ctx)) {
        auto& cache = cacheManager.get(ctx);
        auto request = Const::EquivalenceRequestAttr::get(ctx, originalAttr, newAttr);
        cache.enqueueRequest(Const::FoldingRequest{request, /*newTransformation=*/nullptr});
    }
#endif
}

/**
 * Fuses consecutive SubView transformations into a single transformation.
 *   SubView + SubView ---> SubView
 */
class FuseConsecutiveSubViews final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    FuseConsecutiveSubViews(mlir::MLIRContext* ctx): mlir::OpRewritePattern<Const::DeclareOp>(ctx) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseConsecutiveSubViews::matchAndRewrite(Const::DeclareOp constOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto contentAttr = constOp.getContentAttr();
    auto transformations = contentAttr.getTransformations();

    if (transformations.empty()) {
        return mlir::failure();
    }

    const auto hasConsecutiveSubViews = [&transformations]() {
        for (auto it = transformations.begin(); it != transformations.end() - 1; it++) {
            if (it->isa<Const::SubViewAttr>() && (it + 1)->isa<Const::SubViewAttr>()) {
                return true;
            }
        }
        return false;
    };

    if (!hasConsecutiveSubViews()) {
        return mlir::failure();
    }

    SmallVector<Const::TransformAttrInterface> fusedTransformations;
    for (const auto& attr : transformations) {
        if (attr.isa<Const::SubViewAttr>() && !fusedTransformations.empty() &&
            fusedTransformations.back().isa<Const::SubViewAttr>()) {
            auto subViewAttrFirst = fusedTransformations.back().cast<Const::SubViewAttr>();
            auto subViewAttrSecond = attr.cast<Const::SubViewAttr>();

            auto firstOffset = parseIntArrayAttr<int64_t>(subViewAttrFirst.getOffset());
            auto newOffset = parseIntArrayAttr<int64_t>(subViewAttrSecond.getOffset());

            for (auto i : irange(newOffset.size())) {
                newOffset[i] += firstOffset[i];
            }
            auto newSubViewAttr =
                    Const::SubViewAttr::get(getIntArrayAttr(getContext(), newOffset), subViewAttrSecond.getShape());

            fusedTransformations.pop_back();
            fusedTransformations.push_back(newSubViewAttr);
        } else {
            fusedTransformations.push_back(attr);
        }
    }

    const auto newContentAttr = Const::ContentAttr::get(contentAttr.getBaseContent(), fusedTransformations);
    rewriter.replaceOpWithNewOp<vpux::Const::DeclareOp>(constOp, constOp.getType(), newContentAttr);

    sendEquivalenceRequest(contentAttr, newContentAttr);

    return mlir::success();
}

/**
 * Swaps Reorder and SubView transformations, in order to reduce the amount of data movement
 * when performing the permutation
 *   Reorder + SubView ---> SubView ---> Reorder
 */
class SwapReorderAndSubView final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    SwapReorderAndSubView(mlir::MLIRContext* ctx): mlir::OpRewritePattern<Const::DeclareOp>(ctx) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult SwapReorderAndSubView::matchAndRewrite(Const::DeclareOp constOp,
                                                           mlir::PatternRewriter& rewriter) const {
    auto contentAttr = constOp.getContentAttr();
    auto transformations = contentAttr.getTransformations();

    if (transformations.empty()) {
        return mlir::failure();
    }

    const auto hasSubViewAfterReorder = [&transformations]() {
        for (auto it = transformations.begin(); it != transformations.end() - 1; it++) {
            if (it->isa<Const::ReorderAttr>() && (it + 1)->isa<Const::SubViewAttr>()) {
                return true;
            }
        }
        return false;
    };

    if (!hasSubViewAfterReorder()) {
        return mlir::failure();
    }

    auto newTransformations = to_small_vector(transformations);
    for (size_t idx = 0; idx < transformations.size() - 1; idx++) {
        if (transformations[idx].isa<Const::ReorderAttr>() && transformations[idx + 1].isa<Const::SubViewAttr>()) {
            std::swap(newTransformations[idx], newTransformations[idx + 1]);
        }
    }

    const auto newContentAttr = Const::ContentAttr::get(contentAttr.getBaseContent(), newTransformations);
    rewriter.replaceOpWithNewOp<vpux::Const::DeclareOp>(constOp, constOp.getType(), newContentAttr);

    sendEquivalenceRequest(contentAttr, newContentAttr);

    return mlir::success();
}

/**
 * Swaps Transpose and SubView transformations, in order to reduce the amount of data movement
 * when performing the permutation
 *   Transpose + SubView ---> SubView ---> Transpose
 * Since Transpose will change the input shape, we need to reconstruct the SubView and then swap them.
 * e.g.
 *   Shape:[3, 5] -> const.Transpose -> Shape: [5, 3] -> const.SubView<[0, 0], [2, 1]> -> Shape:[2, 1]
 *   After SwapTransposeAndSubView:
 *   Shape:[3, 5] -> const.SubView<[0, 0], [1, 2]> -> Shape: [1, 2] -> const.Transpose -> Shape:[2, 1]
 */
class SwapTransposeAndSubView final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    SwapTransposeAndSubView(mlir::MLIRContext* ctx): mlir::OpRewritePattern<Const::DeclareOp>(ctx) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult SwapTransposeAndSubView::matchAndRewrite(Const::DeclareOp constOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto contentAttr = constOp.getContentAttr();
    auto transformations = contentAttr.getTransformations();

    if (transformations.empty()) {
        return mlir::failure();
    }

    const auto hasSubViewAfterTranspose = [&transformations]() {
        for (auto it = transformations.begin(); it != transformations.end() - 1; it++) {
            if (it->isa<Const::TransposeAttr>() && (it + 1)->isa<Const::SubViewAttr>()) {
                return true;
            }
        }
        return false;
    };

    if (!hasSubViewAfterTranspose()) {
        return mlir::failure();
    }

    auto newTransformations = to_small_vector(transformations);
    for (size_t idx = 0; idx < transformations.size() - 1; idx++) {
        if (transformations[idx].isa<Const::TransposeAttr>() && transformations[idx + 1].isa<Const::SubViewAttr>()) {
            auto subViewAttr = mlir::dyn_cast_or_null<Const::SubViewAttr>(transformations[idx + 1]);
            VPUX_THROW_WHEN(subViewAttr == nullptr, "Expected SubViewAttr, got {0}",
                            transformations[idx + 1].getTransformationName());

            auto transposeAttr = mlir::dyn_cast_or_null<Const::TransposeAttr>(transformations[idx]);
            VPUX_THROW_WHEN(transposeAttr == nullptr, "Expected TransposeAttr, got {0}",
                            transformations[idx].getTransformationName());

            auto offset = parseIntArrayAttr<int64_t>(subViewAttr.getOffset());
            auto shape = parseIntArrayAttr<int64_t>(subViewAttr.getShape());

            SmallVector<int64_t> newOffset(offset.size());
            SmallVector<int64_t> newShape(shape.size());
            VPUX_THROW_WHEN(newOffset.size() != newShape.size(), "offset size is not equal to shape size");

            const auto order = DimsOrder::fromAffineMap(transposeAttr.getOrder().getValue());

            for (size_t idx = 0; idx < newShape.size(); idx++) {
                newOffset[order.dimAt(idx).ind()] = offset[idx];
                newShape[order.dimAt(idx).ind()] = shape[idx];
            }

            auto newSubViewAttr = Const::SubViewAttr::get(getIntArrayAttr(getContext(), newOffset),
                                                          getIntArrayAttr(getContext(), newShape));
            newTransformations[idx + 1] = newSubViewAttr;
            std::swap(newTransformations[idx], newTransformations[idx + 1]);
        }
    }

    rewriter.replaceOpWithNewOp<vpux::Const::DeclareOp>(
            constOp, constOp.getType(), Const::ContentAttr::get(contentAttr.getBaseContent(), newTransformations));
    return mlir::success();
}

void vpux::Const::DeclareOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseConsecutiveSubViews>(ctx);
    patterns.add<SwapReorderAndSubView>(ctx);
    patterns.add<SwapTransposeAndSubView>(ctx);
}

//
// setupExtraInterfaces
//

void Const::ConstDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::BuiltinDialect*) {
        mlir::RankedTensorType::attachInterface<vpux::TensorNDTypeInterface>(*ctx);
        mlir::RankedTensorType::attachInterface<vpux::TensorBoundedTypeInterface>(*ctx);
        mlir::UnrankedTensorType::attachInterface<vpux::TensorNDTypeInterface>(*ctx);
        mlir::MemRefType::attachInterface<vpux::MemRefNDTypeInterface>(*ctx);
        mlir::UnrankedMemRefType::attachInterface<vpux::MemRefNDTypeInterface>(*ctx);
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/const/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/const/ops.cpp.inc>

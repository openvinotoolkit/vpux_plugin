//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/SymbolTable.h>

namespace vpux::Const {

namespace {
// Note: some users create *memref* constants; however, as per DenseElementsAttr
// documentation, one cannot create dense<> memrefs, so this set of functions
// ensures we deal with tensors where necessary -- making this internally
// simplifies the job for the user as one does not have to specify custom
// conversions or supply 2 types instead of 1.
mlir::RankedTensorType ensureRankedTensor(mlir::RankedTensorType type) {
    return type;
}
mlir::RankedTensorType ensureRankedTensor(mlir::MemRefType type) {
    return mlir::cast<mlir::RankedTensorType>(reconstructTensorType(type));
}

template <typename TensorOrMemref>
mlir::Value createZerosConstImpl(mlir::OpBuilder& builder, mlir::Location loc, TensorOrMemref type) {
    const auto elemType = type.getElementType();

    mlir::DenseElementsAttr denseElementVal = nullptr;
    if (const auto uniformElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
        const auto quantizedType = type.cloneWith(type.getShape(), normalizeQuantStorageType(uniformElemType));
        const auto quantizedTensorType = ensureRankedTensor(quantizedType);
        const auto zeroPoint = uniformElemType.getZeroPoint();
        if (uniformElemType.isSigned()) {
            denseElementVal = mlir::DenseElementsAttr::get(quantizedTensorType, checked_cast<int8_t>(zeroPoint));
        } else {
            denseElementVal = mlir::DenseElementsAttr::get(quantizedTensorType, checked_cast<uint8_t>(zeroPoint));
        }
    } else {
        denseElementVal = wrapData(ensureRankedTensor(type), 0.f);
    }

    VPUX_THROW_WHEN(
            denseElementVal == nullptr,
            "Upsampling has incompatible data type {0}, only float16, float32 or uniform quantized type are supported",
            elemType);

    return builder.create<Const::DeclareOp>(loc, type, Const::ContentAttr::get(denseElementVal)).getOutput();
}
}  // namespace

mlir::Value createZerosConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type) {
    return createZerosConstImpl(builder, loc, type);
}

mlir::Value createZerosConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::MemRefType type) {
    return createZerosConstImpl(builder, loc, type);
}

mlir::Value createFloatConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type,
                             ArrayRef<float> values) {
    const auto constShape = type.getShape();
    const auto shapeTotalSize =
            std::accumulate(constShape.begin(), constShape.end(), int64_t(1), std::multiplies<int64_t>());
    VPUX_THROW_UNLESS(values.size() == 1 || shapeTotalSize == checked_cast<int64_t>(values.size()),
                      "Create float Const failed with unexpect data size");

    const auto denseElementVal = wrapData(type, values);
    VPUX_THROW_UNLESS(denseElementVal != nullptr, "Incompatible data type {0}, only float16 or float32 are supported",
                      type.getElementType());

    return builder.create<Const::DeclareOp>(loc, type, Const::ContentAttr::get(denseElementVal)).getOutput();
}

bool hasNegativeValues(const Const::Content& content) {
    if (content.isSplat()) {
        return content.getSplatValue<double>() < 0.0;
    }

    const auto vals = content.getValues<double>();
    return std::any_of(vals.begin(), vals.end(), [](double val) {
        return val < 0.0;
    });
}

mlir::FailureOr<mlir::Value> updateConstStorageValues(const Logger& log, mlir::OpBuilder& builder,
                                                      Const::DeclareOp origConst, ArrayRef<float> values) {
    const auto contentAttr = origConst.getContentAttr();
    const auto origTransAttrs = contentAttr.getTransformations();
    const auto baseContentType = mlir::cast<NDTypeInterface>(contentAttr.getBaseContent().getType());
    const auto origOutType = mlir::cast<NDTypeInterface>(origConst.getOutput().getType());

    SmallVector<Const::TransformAttrInterface> reserveTransAttrs;
    auto newBaseContentType = baseContentType;
    if (checked_cast<int64_t>(values.size()) == baseContentType.getShape().totalSize()) {
        reserveTransAttrs = to_small_vector(origTransAttrs);
    } else if (checked_cast<int64_t>(values.size()) == origOutType.getShape().totalSize()) {
        newBaseContentType = newBaseContentType.changeShape(origOutType.getShape());
        for (const auto& attr : origTransAttrs) {
            if (attr.isa<Const::ConvertElemTypeAttr>()) {
                reserveTransAttrs.push_back(attr);
            } else if (attr.isa<Const::ReshapeAttr, Const::BroadcastAttr, Const::SubViewAttr,
                                Const::PadWithZeroAttr>()) {
                continue;
            } else {
                // There are many constant transformation attributions
                // It is possible to consider all attributions, but great effort for all corner cases
                log.trace("Unexpected constant transformation attribution '{0}'", attr);
                return mlir::failure();
            }
        }
    } else {
        log.trace("Unexpected values size '{0}' that mismatch with constant base type '{1}' and output type '{2}'",
                  values.size(), baseContentType, origOutType);
        return mlir::failure();
    }

    const auto denseElementVal = wrapData(mlir::cast<mlir::RankedTensorType>(newBaseContentType), values);
    VPUX_THROW_WHEN(denseElementVal == nullptr, "Incompatible data type {0}, only float16 or float32 are supported",
                    newBaseContentType.getElementType());

    auto newContentAttr = Const::ContentAttr::get(denseElementVal);
    for (const auto& attr : reserveTransAttrs) {
        newContentAttr = Const::ContentAttr::addTransformation(newContentAttr, attr);
    }

    return builder.create<Const::DeclareOp>(origConst.getLoc(), origOutType, newContentAttr).getOutput();
}

mlir::Value buildWeightsConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type,
                              ArrayRef<float> values) {
    const auto ctx = builder.getContext();
    const auto origElemType = type.getElementType();

    mlir::Type filterElemType = mlir::Float16Type::get(ctx);
    if (auto qInputElemType = origElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto scale = 1.0f;
        const auto zeroPoint = 0;
        filterElemType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), mlir::Float16Type::get(ctx),
                                                                scale, zeroPoint, std::numeric_limits<uint8_t>::min(),
                                                                std::numeric_limits<uint8_t>::max());
    }

    const auto dataType = mlir::RankedTensorType::get(type.getShape(), mlir::Float32Type::get(ctx));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataType, values);

    auto contentAttr = Const::ContentAttr::get(dataAttr);
    VPUX_THROW_WHEN(!(mlir::isa<mlir::quant::QuantizedType, mlir::Float16Type>(origElemType)), "Unsupported type {0}",
                    origElemType);
    if (auto qElemType = filterElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        contentAttr = contentAttr.convertElemType(getUInt8Type(ctx));
        contentAttr = contentAttr.quantCast(qElemType);
    } else if (origElemType.isa<mlir::Float16Type>()) {
        contentAttr = contentAttr.convertElemType(mlir::Float16Type::get(ctx));
    }
    contentAttr = contentAttr.reorder(mlir::cast<NDTypeInterface>(type).getDimsOrder());

    return builder.create<Const::DeclareOp>(loc, contentAttr.getType(), contentAttr).getOutput();
}

SmallVector<Const::DeclareOp> getDeclareOpsUses(Const::RodataOp rodataOp, mlir::Operation* from) {
    auto usesOpt = rodataOp.getSymbolUses(from);

    if (!usesOpt.has_value()) {
        return {};
    }

    auto uses = usesOpt.value();

    return to_small_vector(uses | transformed([](auto use) {
                               return mlir::dyn_cast_or_null<Const::DeclareOp>(use.getUser());
                           }) |
                           filtered([](Const::DeclareOp declareOp) {
                               return declareOp != nullptr;
                           }));
}

SmallVector<Const::DeclareOp> getDeclareOpsUses(mlir::SymbolRefAttr symbol, mlir::ModuleOp from) {
    auto op = mlir::SymbolTable::lookupSymbolIn(from, symbol);
    auto rodataOp = llvm::dyn_cast_or_null<Const::RodataOp>(op);

    if (rodataOp == nullptr) {
        return {};
    }

    return getDeclareOpsUses(rodataOp, from);
}

}  // namespace vpux::Const

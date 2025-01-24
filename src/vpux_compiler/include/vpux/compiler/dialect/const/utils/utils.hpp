//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/array_ref.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Types.h>

#include <numeric>

namespace vpux::Const {

/// Returns a new zero-value constant of the specified **tensor** type. Supports
/// float and quantized types.
mlir::Value createZerosConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type);

/// Returns a new zero-value constant of the specified **memref** type. Supports
/// float and quantized types.
mlir::Value createZerosConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::MemRefType type);

/// Returns a new constant of the specified **tensor** type.
mlir::Value createFloatConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type,
                             ArrayRef<float> values);
Const::ContentAttr createFloatContentAttr(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type,
                                          ArrayRef<float> values);

/// Returns a new constant of the specified **tensor** type. This is a generic
/// version that could be used to create any constant when the data matches the
/// type. One could also optionally supply a transformation function that
/// applies arbitrary transformations to a raw content attribute.
template <typename In>
mlir::Value createConst(
        mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type, ArrayRef<In> values,
        FuncRef<Const::ContentSetup(Const::ContentSetup&)> transform = [](Const::ContentSetup& setup) {
            return std::move(setup);
        }) {
    const auto constShape = type.getShape();
    const auto shapeTotalSize =
            std::accumulate(constShape.begin(), constShape.end(), int64_t(1), std::multiplies<int64_t>());
    VPUX_THROW_UNLESS(values.size() == 1 || shapeTotalSize == checked_cast<int64_t>(values.size()),
                      "data size != shape size: {0} vs {1}", values.size(), shapeTotalSize);

    const auto dataAttr = createConstContent(type, values);
    VPUX_THROW_UNLESS(dataAttr != nullptr, "Data is incompatible with the supplied type {0}", type.getElementType());

    Const::ContentSetup setup(mlir::cast<mlir::Type>(dataAttr.getType()));
    setup = transform(setup);

    auto contentAttr = Const::ContentAttr::get(dataAttr, setup);
    return builder.create<Const::DeclareOp>(loc, contentAttr.getType(), std::move(contentAttr)).getOutput();
}

mlir::Value buildWeightsConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type,
                              ArrayRef<float> values);

/// Returns whether constant content has negative values.
bool hasNegativeValues(const Const::Content& content);

// Returns all Const::DeclareOp operations nested in 'from' that use the symbol defined by 'rodataOp'.
SmallVector<Const::DeclareOp> getDeclareOpsUses(Const::RodataOp rodataOp, mlir::Operation* from);
// Returns all Const::DeclareOp operations nested in 'from' that use the symbol 'symbol'.
SmallVector<Const::DeclareOp> getDeclareOpsUses(mlir::SymbolRefAttr symbol, mlir::ModuleOp from);

}  // namespace vpux::Const

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes_properties_conversion.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/algo.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace IE {

static constexpr float QUANT_RANGE_RATIO = 5.0;

std::optional<int64_t> getQuantAxisIndex(mlir::Operation* fq, Logger log = Logger::global());
bool areAnyUserQuantizeOps(mlir::Operation* op);
bool areAllUsersQuantized(mlir::Operation* op);
bool isPerAxisQuant(mlir::Value val);
bool checkQuantApproximation(mlir::Operation* op);
bool isPerTensorFQ(ArrayRef<IE::FakeQuantizeOp> fqOps);
IE::FakeQuantizeOp createFQ(mlir::PatternRewriter& rewriter, mlir::Value inputOp, IE::FakeQuantizeOp fq,
                            mlir::Location loc);
Const::DeclareOp createFQConst(mlir::MLIRContext* ctx, mlir::Location loc, float val, mlir::RankedTensorType argType,
                               mlir::PatternRewriter& rewriter);
mlir::Value createFQScaling(mlir::Location loc, mlir::Value input, float scaleFactor, mlir::Type elemType,
                            std::optional<int64_t> levels, std::optional<mlir::Type> lowFpType,
                            vpux::IE::AutoBroadcastTypeAttr autoBroadcast, mlir::PatternRewriter& rewriter);
SmallVector<float> getConst(Const::DeclareOp declOp);
mlir::Value findQuantizedInput(mlir::Value opInput, bool allowPerAxisQuantize);
bool isSymmetricQuantType(mlir::quant::QuantizedType type);
bool hasLeakyReLUPostOp(mlir::Operation* op);
mlir::quant::UniformQuantizedType getQuantizedTypeFromFakeQuantize(IE::FakeQuantizeOp fqOp);
bool hasFQSameZeroPoint(IE::FakeQuantizeOp fqOp);

bool checkRescaledQuantApproximationForConvBasedOp(mlir::Operation* op);

mlir::Type composeWeightsExpressedType(const mlir::Type convolutionInputType);

/*
 *  Bias will be rescaled for mixed precision and written in weight table later, so need to check whether the
 *  rescaled bias range exceeds or not
 */
template <class ConcreteOp>
mlir::LogicalResult checkRescaledBiasRange(ConcreteOp op) {
    auto inputDequantizeOp = op.getInput().template getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = op.getFilter().template getDefiningOp<IE::DequantizeOp>();
    if (!inputDequantizeOp || !filterDequantizeOp) {
        return mlir::failure();
    }

    if (auto biasAttr = op.getBias()) {
        const auto inElemType =
                inputDequantizeOp.getInput().getType().template cast<vpux::NDTypeInterface>().getElementType();
        const auto filterElemType =
                filterDequantizeOp.getInput().getType().template cast<vpux::NDTypeInterface>().getElementType();

        Const::ContentAttr bias;
        if (auto biasConstOp = biasAttr.template getDefiningOp<Const::DeclareOp>()) {
            bias = biasConstOp.getContentAttr();
        } else {
            auto biasDequantOp = biasAttr.template getDefiningOp<IE::DequantizeOp>();
            if (!biasDequantOp) {
                return mlir::failure();
            }
            if (auto inputConst = biasDequantOp.getInput().template getDefiningOp<Const::DeclareOp>()) {
                bias = inputConst.transformContentAttr().dequantize().get();
            } else {
                return mlir::failure();
            }
        }
        const auto OC = getShape(op.getFilter())[Dims4D::Filter::OC];
        if (mlir::failed(VPU::NCESparsity::getRescaledBias(bias, inElemType, filterElemType, OC))) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

}  // namespace IE
}  // namespace vpux

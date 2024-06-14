//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   double clamp_low, double clamp_high, double quant_scale) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getF64FloatAttr(clamp_low), builder.getF64FloatAttr(clamp_high), nullptr, nullptr, nullptr, nullptr,
          nullptr, builder.getF64ArrayAttr({quant_scale}), nullptr, nullptr, nullptr,
          /*ppe_fp_scale=*/nullptr,
          /*ppe_fp_bias=*/nullptr);
}

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   double clamp_low, double clamp_high, double quant_scale, double fp_prelu_alpha) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getF64FloatAttr(clamp_low), builder.getF64FloatAttr(clamp_high), nullptr, nullptr, nullptr, nullptr,
          nullptr, builder.getF64ArrayAttr({quant_scale}), nullptr, nullptr, builder.getF64FloatAttr(fp_prelu_alpha),
          /*ppe_fp_scale=*/nullptr,
          /*ppe_fp_bias=*/nullptr);
}

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   int64_t clamp_low, int64_t clamp_high, int64_t lrelu_mult, int64_t lrelu_shift) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getI64IntegerAttr(clamp_low), builder.getI64IntegerAttr(clamp_high),
          builder.getI64IntegerAttr(lrelu_mult), builder.getI64IntegerAttr(lrelu_shift), nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr,
          /*ppe_fp_scale=*/nullptr,
          /*ppe_fp_bias=*/nullptr);
}

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   int64_t clamp_low, int64_t clamp_high, int64_t lrelu_mult, int64_t lrelu_shift,
                                   int64_t quant_mult, int64_t quant_shift) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getI64IntegerAttr(clamp_low), builder.getI64IntegerAttr(clamp_high),
          builder.getI64IntegerAttr(lrelu_mult), builder.getI64IntegerAttr(lrelu_shift),
          builder.getI64ArrayAttr({quant_mult}), builder.getI64ArrayAttr({quant_shift}), builder.getI64IntegerAttr(0),
          builder.getF64ArrayAttr({1.0}), nullptr, nullptr, nullptr,
          /*ppe_fp_scale=*/nullptr,
          /*ppe_fp_bias=*/nullptr);
}

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   int64_t clamp_low, int64_t clamp_high, int64_t lrelu_mult, int64_t lrelu_shift,
                                   int64_t quant_mult, int64_t quant_shift, int64_t quant_post_shift,
                                   double quant_scale) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getI64IntegerAttr(clamp_low), builder.getI64IntegerAttr(clamp_high),
          builder.getI64IntegerAttr(lrelu_mult), builder.getI64IntegerAttr(lrelu_shift),
          builder.getI64ArrayAttr({quant_mult}), builder.getI64ArrayAttr({quant_shift}),
          builder.getI64IntegerAttr(quant_post_shift), builder.getF64ArrayAttr({quant_scale}), nullptr, nullptr,
          nullptr,
          /*ppe_fp_scale=*/nullptr,
          /*ppe_fp_bias=*/nullptr);
}

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   int64_t clamp_low, int64_t clamp_high, int64_t lrelu_mult, int64_t lrelu_shift,
                                   int64_t quant_mult, int64_t quant_shift, int64_t quant_post_shift,
                                   double quant_scale, double ppe_fp_scale, double ppe_fp_bias) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getI64IntegerAttr(clamp_low), builder.getI64IntegerAttr(clamp_high),
          builder.getI64IntegerAttr(lrelu_mult), builder.getI64IntegerAttr(lrelu_shift),
          builder.getI64ArrayAttr({quant_mult}), builder.getI64ArrayAttr({quant_shift}),
          builder.getI64IntegerAttr(quant_post_shift), builder.getF64ArrayAttr({quant_scale}), nullptr, nullptr,
          nullptr, builder.getF64FloatAttr(ppe_fp_scale), builder.getF64FloatAttr(ppe_fp_bias));
}

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   int64_t clamp_low, int64_t clamp_high, int64_t lrelu_mult, int64_t lrelu_shift,
                                   int64_t quant_mult, int64_t quant_shift, int64_t quant_post_shift,
                                   double quant_scale, int64_t in1_quant_mult, int64_t in2_quant_mult,
                                   double fp_prelu_alpha) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getI64IntegerAttr(clamp_low), builder.getI64IntegerAttr(clamp_high),
          builder.getI64IntegerAttr(lrelu_mult), builder.getI64IntegerAttr(lrelu_shift),
          builder.getI64ArrayAttr({quant_mult}), builder.getI64ArrayAttr({quant_shift}),
          builder.getI64IntegerAttr(quant_post_shift), builder.getF64ArrayAttr({quant_scale}),
          builder.getI64ArrayAttr({in1_quant_mult}), builder.getI64ArrayAttr({in2_quant_mult}),
          builder.getF64FloatAttr(fp_prelu_alpha),
          /*ppe_fp_scale=*/nullptr,
          /*ppe_fp_bias=*/nullptr);
}

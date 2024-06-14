//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

namespace vpux {
namespace VPUIP {

// TODO: E60214, need support more sw kernel task type. Currently only enable MVN
const SmallVector<StringLiteral> SW_KERNELS_SUPPORTING_TILING = {"mvn1",
                                                                 "mvn6",
                                                                 "interpolate",
                                                                 "activation_swish",
                                                                 "activation_gelu",
                                                                 "softmax",
                                                                 "matmul",
                                                                 "activation_hswish",
                                                                 "eltwise_mul",
                                                                 "activation_hardsigmoid",
                                                                 "convert",
                                                                 "activation_tanh",
                                                                 "topk",
                                                                 "gather",
                                                                 "activation_sigmoid",
                                                                 "depth_to_space",
                                                                 "activation_clamp",
                                                                 "eltwise_min",
                                                                 "eltwise_max",
                                                                 "eltwise_power",
                                                                 "activation_abs",
                                                                 "eltwise_div",
                                                                 "prelu_fp16",
                                                                 "normalize_l2",
                                                                 "reduce_l1",
                                                                 "reduce_l2",
                                                                 "reduce_logical_and",
                                                                 "reduce_logical_or",
                                                                 "reduce_max",
                                                                 "reduce_mean",
                                                                 "reduce_min",
                                                                 "reduce_prod",
                                                                 "reduce_sum",
                                                                 "eltwise_greater",
                                                                 "eltwise_less",
                                                                 "eltwise_sub",
                                                                 "eltwise_add",
                                                                 "gru_sequence",
                                                                 "gru_sequence_last_part",
                                                                 "activation_floor",
                                                                 "activation_log",
                                                                 "activation_sqrt",
                                                                 "fake_quantize",
                                                                 "detection_output_sort",
                                                                 "eltwise_select"};

const SmallVector<StringLiteral> SW_KERNELS_SUPPORTING_STRIDE = {"mvn1"};

// TODO: E#117136, use heuristic for tile dim
const SmallVector<StringLiteral> SW_ACTIVATION_KERNELS = {
        "activation_swish", "activation_gelu", "activation_hardsigmoid", "activation_tanh", "activation_sigmoid",
        "activation_clamp", "activation_abs",  "activation_floor",       "hswish_fp16",     "prelu_fp16"};

constexpr StringLiteral SW_KERNEL_NAME_PREFIX = "builtin_";

SmallVector<mlir::Attribute> kernelArgsRange(VPUIP::SwKernelOp swKernelOp);

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, mlir::StringRef builtInFunctionName,
                                          const ArrayRef<mlir::Type> inputTypes, mlir::StringRef kernelEntryName,
                                          mlir::StringRef kernelSourceFileName, const vpux::Logger& log);

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log, vpux::VPU::ArchKind arch);

void initSwKernel(vpux::VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  mlir::ArrayRef<mlir::Attribute> args, const vpux::Logger& log);

void initSwKernel(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRunOp, const vpux::Logger& log);

SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp);
mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log);
bool isActivationSwKernelOp(VPUIP::SwKernelOp swKernelOp);
bool isSwKernelTilingSupported(VPUIP::SwKernelOp swKernelOp);
bool isStridedDataAccessSupported(VPUIP::SwKernelOp swKernelOp);

InputTiling backInferSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const SmallVector<vpux::TileInfo>& outputTiles,
                                       int tileId, Logger log);

SmallVector<mlir::Attribute> getSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                            ArrayRef<mlir::Attribute> origAttr,
                                                            const TilingInfo& inputTiling, const TileInfo& outTile,
                                                            Logger log);

SmallVector<vpux::NDTypeInterface> getSwKernelTiledTypes(VPUIP::SwKernelOp swKernelOp);

bool isCacheOpTaskType(std::optional<::mlir::SymbolRefAttr> kernelTaskType, bool includePrefetch = true);
bool isCacheOpTaskType(mlir::SymbolRefAttr kernelTaskType, bool includePrefetch = true);

bool isCacheHandlingOp(VPUIP::SwKernelOp swKernelOp);

mlir::SmallVector<mlir::Value> getDDRBuffers(mlir::ValueRange buffers);
bool hasInputsInDDR(VPUIP::SwKernelOp swKernelTask);
}  // namespace VPUIP
}  // namespace vpux

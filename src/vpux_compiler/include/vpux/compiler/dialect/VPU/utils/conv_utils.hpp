//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux::VPU {

bool isNCEConvSupported(VPU::ArchKind arch, NDTypeInterface inputType, NDTypeInterface filterType,
                        NDTypeInterface outputType, ArrayRef<int64_t> dilations, int64_t KY, int64_t KX, int64_t SY,
                        int64_t SX, PadInfo pads, bool checkLayout, bool checkChannelAlignment, LogCb logCb,
                        bool supportsInputActCompression = false);

bool isSupportedConv(IE::ConvolutionOp op, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                     bool supportsInputActCompression = false);

bool isSupportedSEPTransposedConv(IE::TransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                  bool checkChannelAlignment, bool supportsInputActCompression = false);

bool isSupportedSEPTransposedConv(IE::GroupTransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                  bool checkChannelAlignment, bool supportsInputActCompression = false);

bool isSupportedSEPTransposedConv(VPU::TransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                  bool checkChannelAlignment, bool supportsInputActCompression = false);

std::optional<bool> isSEPConvCompatibleWithClusterStrategy(VPU::NCEConvolutionOp nceConv,
                                                           VPU::MultiClusterStrategy strategy);

mlir::LogicalResult verifyConvUtil(mlir::Location loc, VPU::ArchKind arch, Shape filterShape, Shape kernelStrides,
                                   PaddingAttr padAttr, ShapeRef weightsTableShape, mlir::Value output);
}  // namespace vpux::VPU

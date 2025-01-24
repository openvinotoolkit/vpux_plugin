//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux {
namespace NPUReg40XX {

namespace DMADescriptorComposer {

struct DMATransactionConfig {
    std::array<size_t, npu40xx::DMA_NUM_DIM_MAX> srcDimSizes = {};
    std::array<size_t, npu40xx::DMA_NUM_DIM_MAX> srcStrides = {};
    std::array<size_t, npu40xx::DMA_NUM_DIM_MAX> dstDimSizes = {};
    std::array<size_t, npu40xx::DMA_NUM_DIM_MAX> dstStrides = {};

    uint64_t numDims = {};
};

DMATransactionConfig configurePatternFromDescriptorAttr(VPUASM::NNDMAOp op);
DMATransactionConfig configurePatternFromTransactionAttr(DMATransaction& transaction);

Descriptors::DMARegister compose(VPUASM::NNDMAOp origOp, ELF::SymbolReferenceMap& symRefMap);

}  // namespace DMADescriptorComposer
}  // namespace NPUReg40XX
}  // namespace vpux

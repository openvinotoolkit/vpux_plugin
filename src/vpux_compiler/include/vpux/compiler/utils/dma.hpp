//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux {

int64_t getDMAPortValue(mlir::Operation* wrappedTaskOp);

SmallVector<VPUIP::DmaChannelType> getDMAChannelsWithIndependentLinkAgents(VPU::ArchKind arch);

// Encode DMA port and channel setting into a single integer for convenient usage by scheduling modules
int64_t getDMAQueueIdEncoding(int64_t port, int64_t channelIdx);
int64_t getDMAQueueIdEncoding(int64_t port, std::optional<vpux::VPUIP::DmaChannelType> channel);
int64_t getDMAQueueIdEncoding(std::optional<vpux::VPUIP::DmaChannelType> channel);
int64_t getDMAQueueIdEncoding(VPU::MemoryKind srcMemKind, VPU::ArchKind arch);

VPUIP::DmaChannelType getDMAQueueTypeFromEncodedId(int64_t dmaQueueIdEncoding, VPU::ArchKind arch);
std::string getDMAChannelTypeAsString(VPUIP::DmaChannelType channelType, VPU::ArchKind arch);
std::string getDMAChannelTypeAsString(int64_t dmaQueueIdEncoding, VPU::ArchKind arch);

}  // namespace vpux

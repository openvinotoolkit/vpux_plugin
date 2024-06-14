//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux::VPU {

bool isSupportedSEPPadOp(IE::PadOp padOp, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                         bool supportsInputActCompression = false);

bool isSupportedSEPPadOp(VPU::PadOp padOp, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                         bool supportsInputActCompression = false);

}  // namespace vpux::VPU

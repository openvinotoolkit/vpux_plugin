//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Dialect.h>

namespace vpux::VPUIP::arch37xx {

void registerAlignedChannelsOpInterfaces(mlir::DialectRegistry& registry);
void registerAlignedWorkloadChannelsOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::VPUIP::arch37xx

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"

using namespace vpux;
using namespace VPUMI40XX;

bool ActKernelRangeOp::supportsTaskLink() {
    return false;
}

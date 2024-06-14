//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux {

struct GraphWriterParams final {
    std::string startAfter;
    std::string stopBefore;
    bool printConst = false;
    bool printDeclarations = false;
    bool printOnlyDotInterfaces = false;
    bool printOnlyTaskAndBarrier = false;
    bool printOnlyAsyncExec = false;
    bool htmlLike = true;
};

mlir::LogicalResult writeDotGraph(mlir::func::FuncOp func, StringRef fileName, const GraphWriterParams& params);

}  // namespace vpux

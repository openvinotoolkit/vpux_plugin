//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/Transforms/InliningUtils.h>

namespace vpux {
namespace VPUIP {

//
// FuncInlinerInterface
//
// Custom implementation of the DialectInlinerInterface for FuncDialect
// Original one can be attached via func::registerInlinerExtension
// but only one needs to be registered,
// so don't use them implementation together
//
// Rationale:
// The function call is located in the VPURT::TaskOp region
// VPURT.Task updates(%bar : !VPURT.Barrier) {
//     func.call @foo1(%in, %tmp)
// }
// VPURT.Task waits(%bar : !VPURT.Barrier) {
//     func.call @foo2(%tmp, %uot)
// }
//
// This means that after inserting operations into main function,
// we must update the barriers for the first and last operation
// in order to restore the dependency between the list of operations:
// processInlinedCallBlocks is used for this.
//
// Even if we have a VPUIP::CallOp,
// we still need to implement the DialectInlinerInterface.
// So it's easier not to add our own operation yet -- less code
//

struct FuncInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final;

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final;

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final;

    void handleTerminator(mlir::Operation*, mlir::ValueRange) const final;

    void processInlinedCallBlocks(mlir::Operation* call,
                                  mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const final;

    std::tuple<mlir::Block*, mlir::Block::iterator> getInlineBlockAndPoint(mlir::Operation* call) const final;

    void eraseCall(mlir::Operation* call) const final;
};

}  // namespace VPUIP
}  // namespace vpux

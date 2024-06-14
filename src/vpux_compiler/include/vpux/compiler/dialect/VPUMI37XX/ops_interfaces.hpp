//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPUMI37XX {

//
// SingleOutputAsIndexOp
//

mlir::LogicalResult verifySingleOutputAsIndexOp(mlir::Operation* op);

template <typename ConcreteOp>
class SingleOutputAsIndexOp : public mlir::OpTrait::TraitBase<ConcreteOp, SingleOutputAsIndexOp> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySingleOutputAsIndexOp(op);
    }
};

}  // namespace VPUMI37XX
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPUMI37XX/ops_interfaces.hpp.inc>

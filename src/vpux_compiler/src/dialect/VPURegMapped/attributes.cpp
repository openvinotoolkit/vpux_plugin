//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPURegMapped/attributes.cpp.inc>

#include <vpux/compiler/dialect/VPURegMapped/enums.cpp.inc>

//
// Dialect hooks
//

void vpux::VPURegMapped::VPURegMappedDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPURegMapped/attributes.cpp.inc>
            >();
}

mlir::Attribute getVPURegMapped_RegisterFieldAttr(::mlir::MLIRContext* context,
                                                  vpux::VPURegMapped::RegFieldType value) {
    return vpux::VPURegMapped::RegisterFieldAttr::get(context, value);
}

mlir::ArrayAttr getVPURegMapped_RegisterFieldArrayAttr(mlir::OpBuilder builder,
                                                       mlir::ArrayRef<vpux::VPURegMapped::RegFieldType> values) {
    auto attrs = llvm::to_vector<8>(llvm::map_range(values, [&](vpux::VPURegMapped::RegFieldType v) -> mlir::Attribute {
        return getVPURegMapped_RegisterFieldAttr(builder.getContext(), v);
    }));
    return builder.getArrayAttr(attrs);
}

mlir::Attribute getVPURegMapped_RegisterAttr(::mlir::MLIRContext* context, vpux::VPURegMapped::RegisterType value) {
    return vpux::VPURegMapped::RegisterAttr::get(context, value);
}

mlir::ArrayAttr getVPURegMapped_RegisterArrayAttr(mlir::OpBuilder builder,
                                                  mlir::ArrayRef<vpux::VPURegMapped::RegisterType> values) {
    auto attrs = llvm::to_vector<8>(llvm::map_range(values, [&](vpux::VPURegMapped::RegisterType v) -> mlir::Attribute {
        return getVPURegMapped_RegisterAttr(builder.getContext(), v);
    }));
    return builder.getArrayAttr(attrs);
}

//
// RegisterFieldAttr::verify
//

mlir::LogicalResult vpux::VPURegMapped::RegisterFieldAttr::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, vpux::VPURegMapped::RegFieldType regField) {
    if (regField == nullptr) {
        return printTo(emitError(), "Got NULL 'regField' in 'RegisterFieldAttr'");
    }

    if (mlir::failed(VPURegMapped::RegFieldType::verify(emitError, regField.getWidth(), regField.getPos(),
                                                        regField.getValue(), regField.getName(),
                                                        regField.getDataType()))) {
        return printTo(emitError(), "RegisterFieldAttr - invalid.");
    }

    return mlir::success();
}

//
// RegisterAttr::verify
//

mlir::LogicalResult vpux::VPURegMapped::RegisterAttr::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, vpux::VPURegMapped::RegisterType reg) {
    if (reg == nullptr) {
        return printTo(emitError(), "Got NULL 'regField' in 'RegisterFieldAttr'");
    }

    if (mlir::failed(VPURegMapped::RegisterType::verify(emitError, reg.getSize(), reg.getName(), reg.getAddress(),
                                                        reg.getRegFields(), reg.getAllowOverlap()))) {
        return printTo(emitError(), "RegisterAttr - invalid.");
    }

    return mlir::success();
}

//
// RegisterMappedAttr::verify
//

mlir::LogicalResult vpux::VPURegMapped::RegisterMappedAttr::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, vpux::VPURegMapped::RegMappedType regMapped) {
    if (regMapped == nullptr) {
        return printTo(emitError(), "Got NULL 'regMapped' in 'RegisterMappedAttr'");
    }

    if (mlir::failed(VPURegMapped::RegMappedType::verify(emitError, regMapped.getName(), regMapped.getRegs()))) {
        return printTo(emitError(), "RegisterMappedAttr - invalid.");
    }

    return mlir::success();
}

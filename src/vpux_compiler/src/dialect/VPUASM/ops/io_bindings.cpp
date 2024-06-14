//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

size_t VPUASM::IOBindingsOp::getNetInputsCount() {
    return getInputDeclarations().front().getOperations().size();
}

SmallVector<VPUASM::DeclareBufferOp, 1> VPUASM::IOBindingsOp::getInputDeclarationsOps() {
    return to_vector<1>(getInputDeclarations().getOps<VPUASM::DeclareBufferOp>());
}

size_t VPUASM::IOBindingsOp::getNetOutputsCount() {
    return getOutputDeclarations().front().getOperations().size();
}

SmallVector<VPUASM::DeclareBufferOp, 1> VPUASM::IOBindingsOp::getOutputDeclarationsOps() {
    return to_vector<1>(getOutputDeclarations().front().getOps<VPUASM::DeclareBufferOp>());
}

size_t VPUASM::IOBindingsOp::getProfilingBuffsCount() {
    return getProfilingBuffDeclarations().front().getOperations().size();
}

SmallVector<VPUASM::DeclareBufferOp, 1> VPUASM::IOBindingsOp::getProfilingBuffDeclarationsOps() {
    return to_vector<1>(getProfilingBuffDeclarations().front().getOps<VPUASM::DeclareBufferOp>());
}

VPUASM::IOBindingsOp VPUASM::IOBindingsOp::getFromModule(mlir::ModuleOp module) {
    auto bindingOps = to_small_vector(module.getOps<VPUASM::IOBindingsOp>());

    VPUX_THROW_UNLESS(bindingOps.size() <= 1,
                      "Can't have more than one 'VPUASM::IOBindingsOp' Operation in Module, got `{0}`",
                      bindingOps.size());

    VPUASM::IOBindingsOp bindingOp = bindingOps.size() == 1 ? bindingOps.front() : nullptr;

    return bindingOp;
}

void VPUASM::IOBindingsOp::build(::mlir::OpBuilder&, ::mlir::OperationState& odsState) {
    auto inputsRegion = odsState.addRegion();
    auto outputsRegion = odsState.addRegion();
    auto profilingBuffsRegion = odsState.addRegion();

    inputsRegion->emplaceBlock();
    outputsRegion->emplaceBlock();
    profilingBuffsRegion->emplaceBlock();
}

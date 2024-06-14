//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_params_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::LogicalResult KernelParamsRewriter::symbolize(VPUMI40XX::KernelParamsOp op, SymbolMapper&,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();

    llvm::SmallVector<mlir::Attribute> inputSyms(op.getInputs().size());
    llvm::SmallVector<mlir::Attribute> outputSyms(op.getOutputs().size());

    for (auto inputIt : llvm::enumerate(op.getInputs())) {
        auto inputIdx = inputIt.index();
        auto symVal = findSym(inputIt.value());

        inputSyms[inputIdx] = symVal;
    }
    for (auto outputIt : llvm::enumerate(op.getOutputs())) {
        auto outputIdx = outputIt.index();
        auto symVal = findSym(outputIt.value());

        outputSyms[outputIdx] = symVal;
    }

    auto inputsAttr = mlir::ArrayAttr::get(getContext(), inputSyms);
    auto outputsAttr = mlir::ArrayAttr::get(getContext(), outputSyms);

    rewriter.create<VPUASM::KernelParamsOp>(op.getLoc(), symName, inputsAttr, outputsAttr, op.getKernelTypeAttr(),
                                            op.getKernelParamsAttr());

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux

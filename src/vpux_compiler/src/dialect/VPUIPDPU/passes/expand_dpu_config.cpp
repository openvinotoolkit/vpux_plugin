//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_variant_rewriter.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;
using namespace VPUIPDPU;

namespace {

//
// ExpandDPUConfigPass
//

class ExpandDPUConfigPass final : public VPUIPDPU::ExpandDPUConfigBase<ExpandDPUConfigPass> {
public:
    explicit ExpandDPUConfigPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ExpandDPUConfigPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<VPUASM::VPUASMDialect>();
    target.addLegalDialect<VPUIPDPU::VPUIPDPUDialect>();

    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain);

    mlir::RewritePatternSet patternsVar(&ctx);
    patternsVar.add<DPUVariantRewriter>(&ctx, _log, symRefMap);
    target.addIllegalOp<VPUASM::DPUVariantOp>();
    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patternsVar)))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet patternsInv(&ctx);
    patternsInv.add<DPUInvariantRewriter>(&ctx, _log, symRefMap);
    target.addIllegalOp<VPUASM::DPUInvariantOp>();
    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patternsInv)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createExpandDPUConfigPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIPDPU::createExpandDPUConfigPass(Logger log) {
    return std::make_unique<ExpandDPUConfigPass>(log);
}

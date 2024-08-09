//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"
#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_type_converter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/act_shave_runtime_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/barrier_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/bootstrap_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_const_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_task_addr_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_task_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dma_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dpu_variant_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/enqueue_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_data_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_entry_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_invocation_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_params_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_range_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_text_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/m2i_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/profiling_metadata_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/task_sink_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/view_task_range_rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/symbolization.hpp"

using namespace vpux;
using namespace vpumi40xx2vpuasm;

namespace {

//
// ConvertVPUMI40XX2VPUASMPass
//

class ConvertVPUMI40XX2VPUASMPass final : public ConvertVPUMI40XX2VPUASMBase<ConvertVPUMI40XX2VPUASMPass> {
public:
    explicit ConvertVPUMI40XX2VPUASMPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertVPUMI40XX2VPUASMPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp cnnOp;

    IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, netFunc);

    llvm::DenseMap<mlir::Value, mlir::FlatSymbolRefAttr> symbolNameMappings;

    SymbolizationTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<VPUMI40XX::VPUMI40XXDialect>();
    target.addIllegalDialect<Const::ConstDialect>();
    target.addIllegalDialect<VPURT::VPURTDialect>();

    target.addLegalDialect<VPUASM::VPUASMDialect>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addLegalOp<VPURegMapped::TaskBufferLayoutOp>();

    {
        // don't use rewriter infrastructure here as it's not "regular" lowering
        // instead of lowering some operation VPUMI40XX -> VPUASM with symbol name
        // lower VPUMI40XX::OpRanges -> mlir::func::ReturnOp with no arguments
        auto opRanges = mlir::cast<VPUMI40XX::OpRanges>(netFunc.getBlocks().front().getTerminator());
        mlir::OpBuilder(opRanges).create<mlir::func::ReturnOp>(opRanges.getLoc());
        opRanges.erase();
    }

    SymbolizationPatternSet patterns(&ctx);

    patterns.add<DeclareBufferRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<DeclareConstBufferRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<DeclareTaskBufferRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<NNDMARewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<M2IRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<KernelTextRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<KernelDataRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<KernelEntryRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<KernelParamsRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<ActShaveRtRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<KernelRangeRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<KernelInvocationRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<DPUVariantRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<DPUInvariantRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<TaskSinkRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<ViewTaskRangeRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<EnqueueRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<DeclareTaskAddrBuffRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<BarrierRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<MappedInferenceRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<ProfilingMetadataRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);
    patterns.add<BootstrapRewriter>(netFunc, typeConverter, symbolNameMappings, &ctx, _log);

    if (mlir::failed(
                mlir::applyFullConversion(netFunc, target, SymbolizationPatternSet::freeze(std::move(patterns))))) {
        signalPassFailure();
    }

    return;
}

}  // namespace

//
// createConvertVPUMI40XX2VPUASMPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUMI40XX2VPUASMPass(Logger log) {
    return std::make_unique<ConvertVPUMI40XX2VPUASMPass>(log);
}

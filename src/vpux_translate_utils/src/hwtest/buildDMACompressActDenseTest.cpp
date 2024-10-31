//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

//                         comp       decomp
//       [CMXbuf0Uncomp] --------   --------- [CMXbuf1Uncomp]
//              |               |   |                |
//              | 1:1           |   |                | 1:1
//              |               |   |                |
//      [DDRinput_uncomp]  [DDRspilledComp]  [DDRoutput_uncomp]
//

void buildDMACompressActDense(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                              mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType) {
    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);

    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    auto* ctx = builder.getContext();

    auto input = testDesc.getInputLayerList().front();
    auto dmaParams = testDesc.getDMAparams();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    if (testDesc.getArchitecture() == vpux::VPU::ArchKind::NPU40XX) {
        VPUX_THROW_UNLESS(dmaParams.engine == 0, "buildDMACompressActDense: DMA on NPU40XX should have 1 engine");
    }

    // Activation compression enabled - Dense Mode
    // Compiler must ensure that the DDR allocation is capable of handling the worst case compressed size (which can be
    // more than the source)
    // DTS = X * Y * Z * (element size in bytes)
    // denseSize = (DTS * (65/64)) + 1
    // DDR Allocation (32B aligned) = denseSize + ( (denseSize % 32) ? (32 – (denseSize % 32) : 0)
    const auto alignment = Byte(32);
    const auto elementSizeBytes =
            getElemTypeSize(inputType).count() < CHAR_BIT ? 1 : getElemTypeSize(inputType).count() / CHAR_BIT;
    const auto denseTensorSize = inShape[vpux::Dims4D::Act::C.ind()] * inShape[vpux::Dims4D::Act::W.ind()] *
                                 inShape[vpux::Dims4D::Act::H.ind()] * elementSizeBytes;
    const auto denseSize = static_cast<int64_t>(denseTensorSize * (static_cast<float>(65) / 64) + 1);
    const SmallVector<int64_t> DDRspilledCompShape = {1, 1, 1, vpux::alignValUp(denseSize, alignment.count())};

    VPUX_THROW_UNLESS(!inShape.empty(), "buildDMACompressActDense: Input rank is 0");
    VPUX_THROW_UNLESS(inShape == outShape, "buildDMACompressActDense: in_shape and out_shape don't match");
    VPUX_THROW_UNLESS(inputType == outputType, "buildDMACompressActDense: inputType and outputType don't match");

    auto inputTotalSize = totalTensorSize(inShape, inputType);

    SmallVector<int64_t> flatShape{1, 1, 1, denseTensorSize};
    auto innerType = getUInt8Type(ctx);

    const auto inType = getMemRefType(VPURT::BufferSection::NetworkInput, flatShape, innerType, DimsOrder::NHWC);
    const auto outType = getMemRefType(VPURT::BufferSection::NetworkOutput, flatShape, innerType, DimsOrder::NHWC);
    auto DDRspilledCompType =
            getMemRefType(VPURT::BufferSection::NetworkOutput, DDRspilledCompShape, innerType, DimsOrder::NHWC);
    DDRspilledCompType = setCompressionState(DDRspilledCompType, VPUIP::CompressionState::RuntimeCompressed)
                                 .cast<mlir::MemRefType>();

    const auto funcType =
            builder.getFunctionType(ArrayRef(std::vector<mlir::Type>{inType, outType, DDRspilledCompType}),
                                    ArrayRef(std::vector<mlir::Type>{outType, DDRspilledCompType}));

    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), printToString("dma_compress_activations"),
                                                   funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr,
                                                   /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcbuilder, testDesc.getWLMParams().isWLMPartialEnabled);
    int barrierNumber = freeBarrierId++;

    size_t CMX0_AVAILABLE_OFFSET = 0;

    // DDRinput_uncomp - CMXbuf0Uncomp
    auto DDRinput_uncomp = func.getArgument(0);

    const auto sectionIdx = 0;
    auto CMXbuf0UncompType =
            getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, flatShape, innerType, DimsOrder::NHWC);
    auto CMXbuf0Uncomp = createDeclareTensorOp(funcbuilder, CMXbuf0UncompType, VPURT::BufferSection::CMX_NN, sectionIdx,
                                               CMX0_AVAILABLE_OFFSET);
    CMX0_AVAILABLE_OFFSET += inputTotalSize;

    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, waitWLMBarrier, mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), DDRinput_uncomp,
                                          CMXbuf0Uncomp.getOperation()->getResult(0), dmaParams.engine);

    // act_compression_entry
    enum { actCompressionEntrySize = 32 };
    const auto elemType = getUInt8Type(ctx);

    auto actCompressionEntryType =
            getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, ShapeRef({1, 1, 1, actCompressionEntrySize}),
                          elemType, DimsOrder::NHWC);
    auto actCompressionEntry = createDeclareTensorOp(funcbuilder, actCompressionEntryType, VPURT::BufferSection::CMX_NN,
                                                     sectionIdx, CMX0_AVAILABLE_OFFSET);

    CMX0_AVAILABLE_OFFSET += actCompressionEntrySize;
    // CMXbuf0Uncomp - DDRspilledComp
    auto DDRspilledComp = func.getArgument(2);

    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    VPURT::wrapIntoTaskOp<VPUIP::CompressDMAOp>(
            funcbuilder, mlir::ValueRange(barrier0.getBarrier()), mlir::ValueRange(barrier1.getBarrier()),
            builder.getUnknownLoc(), CMXbuf0Uncomp.getOperation()->getResult(0),
            actCompressionEntry.getOperation()->getResult(0), DDRspilledComp, dmaParams.engine);

    // DDRspilledComp - CMXbuf1Uncomp
    auto CMXbuf1UncompType =
            getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, flatShape, innerType, DimsOrder::NHWC);
    auto CMXbuf1Uncomp = createDeclareTensorOp(funcbuilder, CMXbuf1UncompType, VPURT::BufferSection::CMX_NN, sectionIdx,
                                               CMX0_AVAILABLE_OFFSET);
    CMX0_AVAILABLE_OFFSET += inputTotalSize;

    auto barrier2 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    VPURT::wrapIntoTaskOp<VPUIP::DecompressDMAOp>(funcbuilder, mlir::ValueRange(barrier1.getBarrier()),
                                                  mlir::ValueRange(barrier2.getBarrier()), builder.getUnknownLoc(),
                                                  DDRspilledComp, actCompressionEntry.getOperation()->getResult(0),
                                                  CMXbuf1Uncomp.getOperation()->getResult(0), dmaParams.engine);

    // CMXbuf1Uncomp - DDRoutput_uncomp
    auto DDRoutput_uncomp = func.getArgument(1);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++,
                                                                      testDesc.getWLMParams().isWLMPartialEnabled);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            funcbuilder, mlir::ValueRange(barrier2.getBarrier()), mlir::ValueRange(finalBarrier.getBarrier()),
            builder.getUnknownLoc(), CMXbuf1Uncomp.getOperation()->getResult(0), DDRoutput_uncomp, dmaParams.engine);

    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                             mlir::ValueRange{DDRoutput_uncomp, DDRspilledComp});

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(flatShape), innerType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(flatShape), innerType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(DDRspilledCompShape), innerType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux

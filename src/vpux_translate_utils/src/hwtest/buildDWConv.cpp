//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

unsigned round_up(unsigned x, unsigned mult) {
    return ((x + mult - 1) / mult) * mult;
}

SmallVector<int64_t> getWeightsPaddedShape(ArrayRef<int64_t> wt_shape) {
    auto kernelWidth = wt_shape[3];
    auto kernelHeight = wt_shape[2];

    // Initializions are done assuming regular convolution and then eventually modified for depthwise
    auto inputChannels = wt_shape[1];
    auto outputChannels = wt_shape[0];

    inputChannels = outputChannels;

    auto weightSetDimension = kernelWidth * kernelHeight * inputChannels;

    weightSetDimension = kernelWidth * kernelHeight;

    auto weightSetDimensionPadded = round_up(static_cast<unsigned int>(weightSetDimension), 16);

    return SmallVector<int64_t>{outputChannels, 1, 1, weightSetDimensionPadded};
}

void buildDWConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                 Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();

    const auto arch = testDesc.getArchitecture();
    auto input = testDesc.getInputLayerList().front();
    auto weight = testDesc.getWeightLayers().front();
    auto conv = testDesc.getConvLayer();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!in_shape.empty(), "buildDWConv: Got empty inputShape");
    VPUX_THROW_UNLESS(!out_shape.empty(), "buildDWConv: Got empty outputShape");

    VPUX_THROW_UNLESS(conv.group == in_shape[1],
                      "For Depthwise convolution group should be equal to no. of input channels");

    std::vector<int64_t> filter_size{weight.shape[2], weight.shape[3]};
    std::vector<int64_t> stried_vec(conv.stride.begin(), conv.stride.end());
    std::vector<int64_t> padding_vec = convertNBPadtoNCETaskPad(conv.pad);

    VPUX_THROW_UNLESS(stried_vec.size() == 2, "Strides vector has inappropriate size");

    SmallVector<int64_t> wt_data_shape{weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]};

    const char* weight_file_name = "weights.dat";

    auto output_totalsize = totalTensorSize(out_shape, outputType);
    auto input_totalsize = totalTensorSize(in_shape, inputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto weightsElementTypeBitSize = static_cast<Bit>(getElemTypeSize(weightsType)).count();
    const auto alignment = (16 * weightsElementTypeBitSize) / CHAR_BIT;
    const auto WEIGHTS_CMX_OFFSET =
            vpux::alignValUp(INPUT_CMX_OFFSET + input_totalsize, static_cast<std::uint64_t>(alignment));

    SmallVector<mlir::Type> inputTypes;

    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, in_shape, inputType, DimsOrder::NHWC));
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    // TODO: Func should not return
    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), printToString("dw_conv_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // weights data
    auto wt_data_shape_padded = getWeightsPaddedShape(ArrayRef(wt_data_shape));
    auto weightData_ddr_type =
            getMemRefType(VPURT::BufferSection::Constant, wt_data_shape_padded, weightsType, DimsOrder::NHWC);

    auto wt_data_vals = generateWeights(wt_data_shape_padded, weightsType, builder.getContext(), weight_file_name);
    auto wt_data_attr = Const::ContentAttr::get(wt_data_vals);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }

    auto weight_data_ddr = funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightData_ddr_type,
                                                                wt_data_attr.reorder(DimsOrder::NHWC));

    // weights cmx tensor
    auto wtData_cmx_type =
            getMemRefType(VPURT::BufferSection::CMX_NN, 0, wt_data_shape_padded, weightsType, DimsOrder::NHWC);
    auto wtData_cmx = createDeclareTensorOp(funcbuilder, wtData_cmx_type, VPURT::BufferSection::CMX_NN,
                                            /*locale index=*/0,
                                            /*data idx=*/WEIGHTS_CMX_OFFSET);

    auto weight_padded_totalsize = totalTensorSize(wt_data_shape_padded, weightsType);
    const auto ACTIVATIONWINDOW_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weight_padded_totalsize;

    // input - output cmx tensors
    auto inputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, in_shape, inputType, DimsOrder::NHWC);
    auto inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);

    auto outputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, out_shape, outputType, DimsOrder::NHWC);
    auto outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parent_outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, 1);
    // finalBarrier passed as production barrier to last DMA task
    auto barrier2 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, 2);

    // DMAs
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(),
                                                mlir::ValueRange(barrier0.getBarrier()), loc, funcinput,
                                                inputcmx.getOperation()->getResult(0), 0);
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), loc,
            weight_data_ddr.getOperation()->getResult(0), wtData_cmx.getOperation()->getResult(0), 0);

    // Activation Window ddr
    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(
            VPU::NCESparsity::Mode::DW_CONV, ShapeRef(filter_size), stried_vec[1],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType,
            in_shape[1]);
    mlir::IntegerAttr actChannelLength = funcbuilder.getI32IntegerAttr(checked_cast<int32_t>(bitPatternSize));

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(
            VPU::NCESparsity::Mode::DW_CONV, ShapeRef(filter_size), stried_vec[1],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType,
            in_shape[1]);

    const auto sparsity_type = getUInt8Type(ctx);
    SmallVector<int64_t> sparsity_shape{1, 1, 1, static_cast<int64_t>(fakeSparsity.size())};

    const auto dataStorageType = mlir::RankedTensorType::get(sparsity_shape, sparsity_type);
    const auto sparsityAttr = mlir::DenseElementsAttr::get(dataStorageType, ArrayRef(fakeSparsity));

    auto activationWindow_ddr_type =
            getMemRefType(VPURT::BufferSection::Constant, sparsity_shape, sparsity_type, DimsOrder::NHWC);
    auto activationWindow_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), activationWindow_ddr_type,
                                                 Const::ContentAttr::get(sparsityAttr).reorder(DimsOrder::NHWC));

    auto activationwindow_totalsize = totalTensorSize(sparsity_shape, sparsity_type);
    auto activationwindow_totalsize_bytes = activationwindow_totalsize * sparsity_type.getIntOrFloatBitWidth() / 8;

    // Activation Window cmx
    auto actWindow_cmx_type =
            getMemRefType(VPURT::BufferSection::CMX_NN, 0, sparsity_shape, sparsity_type, DimsOrder::NHWC);
    auto actWindow_cmx =
            createDeclareTensorOp(funcbuilder, actWindow_cmx_type, VPURT::BufferSection::CMX_NN, /*locale index=*/0,
                                  /*data idx=*/ACTIVATIONWINDOW_CMX_OFFSET);

    // activation window dma ddr->cmx
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), loc,
            activationWindow_ddr.getOperation()->getResult(0), actWindow_cmx.getOperation()->getResult(0), 0);

    // weights table ddr tensor
    auto weights_outChannel = wtData_cmx_type.getShape()[0];
    SmallVector<int64_t> wtTbl_data_shape{weights_outChannel, 1, 1, 4};
    auto weightTblData_ddr_type = getMemRefType(VPURT::BufferSection::Constant, wtTbl_data_shape,
                                                builder.getIntegerType(32, /*isSigned=*/true), DimsOrder::NHWC);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    auto weights_set_size =
            wtData_cmx_type.getShape()[1] * wtData_cmx_type.getShape()[2] * wtData_cmx_type.getShape()[3];
    size_t elementsize_bytes = 0;
    if (auto qType = wtData_cmx_type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        elementsize_bytes = qType.getStorageType().getIntOrFloatBitWidth() / CHAR_BIT;

    } else {
        elementsize_bytes = (wtData_cmx_type.getElementType().getIntOrFloatBitWidth()) / CHAR_BIT;
    }
    auto weights_set_nbytes = weights_set_size * elementsize_bytes;

    const auto sparsityPtrStep = 0;
    const std::vector<int32_t> wtTbl_data_values_vec = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<int32_t>(WEIGHTS_CMX_OFFSET), static_cast<int32_t>(weights_set_nbytes),
            static_cast<int32_t>(ACTIVATIONWINDOW_CMX_OFFSET), sparsityPtrStep, arch, weights_outChannel, weightsType);

    auto wtTbl_data_values = ArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor
    auto wtTbl_cmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, wtTbl_data_shape,
                                        builder.getIntegerType(32, /*isSigned=*/true), DimsOrder::NHWC);
    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + activationwindow_totalsize_bytes;

    auto wtTbl_cmx =
            createDeclareTensorOp(funcbuilder, wtTbl_cmx_type, VPURT::BufferSection::CMX_NN, /*locale index=*/0,
                                  /*data idx=*/WEIGHTSTABLE_CMX_OFFSET);

    // weights table dma ddr->cmx
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), loc,
            weightTbl_data_ddr.getOperation()->getResult(0), wtTbl_cmx.getOperation()->getResult(0), 0);

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stried_vec);
    auto kernel_padding = VPU::getPaddingAttr(ctx, padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                              padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    auto nceTask = vpux::VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcbuilder, mlir::ValueRange(barrier0.getBarrier()), mlir::ValueRange(barrier1.getBarrier()), loc,
            outputcmx_type, inputcmx.getOperation()->getResult(0), wtData_cmx.getOperation()->getResult(0),
            wtTbl_cmx.getOperation()->getResult(0), /*instruction_table_list*/ nullptr, /*spr_lookup_table*/ nullptr,
            actWindow_cmx.getOperation()->getResult(0), parent_inputcmx.getOperation()->getResult(0),
            parent_outputcmx.getOperation()->getResult(0), outputcmx.getOperation()->getResult(0),
            VPUIP::NCETaskType::DWCONV, filtersize, strides, kernel_padding, actChannelLength, /*is_continued*/ nullptr,
            /*sp_pattern*/ nullptr);

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t bypassMult = 1;
    int64_t bypassShift = 0;

    if (auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }
    nceTask.addPPETask(funcbuilder, VPU::PPEMode::NOOP, clampLow, clampHigh, bypassMult, bypassShift);

    // Create DPU task for NCE task

    const std::vector<int32_t> startVec{0, 0, 0};
    auto start = getIntArrayAttr(builder, startVec);
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{out_shape[3] - 1, out_shape[2] - 1, out_shape[1] - 1});
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{in_shape[3] - 1, in_shape[2] - 1, in_shape[1] - 1});
    auto pad = VPU::getPaddingAttr(ctx, padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                   padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    nceTask.addDPUTask(funcbuilder, start, outEnd, start, inEnd, pad, VPU::MPEMode::CUBOID_8x16);

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.getBarrier()),
                                                mlir::ValueRange(barrier2.getBarrier()), loc,
                                                outputcmx.getOperation()->getResult(0), funcoutput, 0);

    // TODO : return empty as func does not return anything
    /* auto returnOp = */ funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux

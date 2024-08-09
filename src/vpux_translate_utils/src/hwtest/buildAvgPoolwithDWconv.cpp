//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

#include <climits>

namespace vpux {
namespace hwtest {

void buildAvgpoolWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                            Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto input = testDesc.getInputLayerList().front();
    auto pool_op = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!in_shape.empty(), "buildAvgpoolWithDwConv: Got empty inputShape");
    VPUX_THROW_UNLESS(!out_shape.empty(), "buildAvgpoolWithDwConv: Got empty outputShape");

    std::vector<int64_t> filter_size{pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};
    std::vector<int64_t> stride_vec(pool_op.stride.begin(), pool_op.stride.end());
    std::vector<int64_t> padding_vec = convertNBPadtoNCETaskPad(pool_op.pad);

    auto input_totalsize = totalTensorSize(in_shape, inputType);
    auto output_totalsize = totalTensorSize(out_shape, outputType);

    SmallVector<int64_t> wt_data_shape{in_shape[1], 1, pool_op.kernel_shape.at(0), pool_op.kernel_shape.at(1)};

    auto scaleValue = 1 / double(pool_op.kernel_shape.at(0) * pool_op.kernel_shape.at(1));

    mlir::Type weightsType = inputType;

    if (auto qtype = inputType.dyn_cast<mlir::quant::QuantizedType>()) {
        auto inputStorageType = mlir::quant::QuantizedType::castToStorageType(qtype);
        int64_t zeroPoint = 0;

        if (inputStorageType.isUnsignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else if (inputStorageType.isSignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::FlagValue::Signed,
                                                                 getSInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else {
            VPUX_THROW("Unsupported storage type for input quantized type. I8 or U8 is supported only");
        }
    }

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto WEIGHTS_CMX_OFFSET = INPUT_CMX_OFFSET + input_totalsize;

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, in_shape, inputType, DimsOrder::NHWC));
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::func::FuncOp>(loc, printToString("avgPool_{0}_{1}", inputType, outputType),
                                                   funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr,
                                                   /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // weights data

    // Generate weights for kh x kw DW conv

    auto weightData_ddr_type2 =
            getMemRefType(VPURT::BufferSection::Constant, wt_data_shape, weightsType, DimsOrder::NHWC);
    size_t weightDataSize = static_cast<size_t>(std::accumulate(wt_data_shape.begin(), wt_data_shape.end(),
                                                                static_cast<int64_t>(1), std::multiplies<int64_t>()));

    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_data_shape, weightsType);
    if (auto qtype = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_data_shape, getSInt8Type(ctx));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_data_shape, getUInt8Type(ctx));
        }
    }
    mlir::DenseElementsAttr wt_data_valss;
    if (weightsType.isF16()) {
        std::vector<vpux::type::float16> wt_vec(weightDataSize, static_cast<float>(scaleValue));
        wt_data_valss = mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef<vpux::type::float16>(wt_vec));
    } else if (weightsType.isBF16()) {
        std::vector<vpux::type::bfloat16> wt_vec(weightDataSize, static_cast<float>(scaleValue));
        wt_data_valss = mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef<vpux::type::bfloat16>(wt_vec));
    } else {
        scaleValue = 1;
        if (weightsType.dyn_cast<mlir::quant::QuantizedType>().getFlags() & mlir::quant::QuantizationFlags::Signed) {
            std::vector<int8_t> wt_vec(weightDataSize, static_cast<int8_t>(scaleValue));
            wt_data_valss = mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef<int8_t>(wt_vec));
        } else {
            std::vector<uint8_t> wt_vec(weightDataSize, static_cast<uint8_t>(scaleValue));
            wt_data_valss = mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef<uint8_t>(wt_vec));
        }
    }
    auto wt_data_attr = Const::ContentAttr::get(wt_data_valss);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }
    auto weight =
            funcbuilder.create<Const::DeclareOp>(loc, weightData_ddr_type2, wt_data_attr.reorder(DimsOrder::NHWC));

    auto weight_data_ddr = VPUIP::alignDepthWiseWeightsTensor(funcbuilder, loc, weight.getResult());

    auto wt_data_shape_padded = weight_data_ddr.getType().cast<vpux::NDTypeInterface>().getShape().raw();

    // weights cmx tensor
    auto wtData_cmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, to_vector<4>(wt_data_shape_padded),
                                         weightsType, DimsOrder::NHWC);
    auto wtData_cmx =
            createDeclareTensorOp(funcbuilder, wtData_cmx_type, VPURT::BufferSection::CMX_NN, 0, WEIGHTS_CMX_OFFSET);

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

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcbuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    // barrier config
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, freeBarrierId++,
                                                                      testDesc.getWLMParams().isWLMPartialEnabled);

    // DMAs
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, waitWLMBarrier, mlir::ValueRange(barrier0.getBarrier()), loc,
                                          funcinput, inputcmx.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), loc,
                                          weight_data_ddr, wtData_cmx.getOperation()->getResult(0), 0);

    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(
            VPU::NCESparsity::Mode::DW_CONV, ShapeRef(filter_size), stride_vec[1],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType,
            in_shape[1]);
    mlir::IntegerAttr actChannelLength = funcbuilder.getI32IntegerAttr(checked_cast<int32_t>(bitPatternSize));

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(
            VPU::NCESparsity::Mode::DW_CONV, ShapeRef(filter_size), stride_vec[1],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType,
            in_shape[1]);

    const auto sparsity_type = getUInt8Type(ctx);
    SmallVector<int64_t> sparsity_shape{1, 1, 1, static_cast<int64_t>(fakeSparsity.size())};

    const auto dataStorageType = mlir::RankedTensorType::get(sparsity_shape, sparsity_type);
    const auto sparsityAttr = mlir::DenseElementsAttr::get(dataStorageType, ArrayRef(fakeSparsity));

    auto activationWindow_ddr_type =
            getMemRefType(VPURT::BufferSection::Constant, sparsity_shape, sparsity_type, DimsOrder::NHWC);
    auto activationWindow_ddr = funcbuilder.create<Const::DeclareOp>(
            loc, activationWindow_ddr_type, Const::ContentAttr::get(sparsityAttr).reorder(DimsOrder::NHWC));

    auto activationwindow_totalsize = totalTensorSize(sparsity_shape, sparsity_type);
    auto activationwindow_totalsize_bytes =
            activationwindow_totalsize * sparsity_type.getIntOrFloatBitWidth() / CHAR_BIT;

    // Activation Window cmx
    auto actWindow_cmx_type =
            getMemRefType(VPURT::BufferSection::CMX_NN, 0, sparsity_shape, sparsity_type, DimsOrder::NHWC);
    auto actWindow_cmx = createDeclareTensorOp(funcbuilder, actWindow_cmx_type, VPURT::BufferSection::CMX_NN, 0,
                                               ACTIVATIONWINDOW_CMX_OFFSET);

    // activation window dma ddr->cmx
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), loc,
                                          activationWindow_ddr.getOperation()->getResult(0),
                                          actWindow_cmx.getOperation()->getResult(0), 0);

    // weights table ddr tensor
    auto weights_outChannel = wtData_cmx_type.getShape()[0];
    SmallVector<int64_t> wtTbl_data_shape{weights_outChannel, 1, 1, 4};
    auto weightTblData_ddr_type = getMemRefType(VPURT::BufferSection::Constant, wtTbl_data_shape,
                                                builder.getIntegerType(32, true), DimsOrder::NHWC);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, true));

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
            static_cast<int32_t>(ACTIVATIONWINDOW_CMX_OFFSET), sparsityPtrStep, testDesc.getArchitecture(),
            weights_outChannel, weightsType);

    auto wtTbl_data_values = ArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr = funcbuilder.create<Const::DeclareOp>(
            loc, weightTblData_ddr_type, Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor

    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + activationwindow_totalsize_bytes;
    auto wtTbl_cmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, 0, wtTbl_data_shape,
                                        builder.getIntegerType(32, true), DimsOrder::NHWC);
    auto wtTbl_cmx = createDeclareTensorOp(funcbuilder, wtTbl_cmx_type, VPURT::BufferSection::CMX_NN, 0,
                                           WEIGHTSTABLE_CMX_OFFSET);

    // weights table dma ddr->cmx
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), loc,
                                          weightTbl_data_ddr.getOperation()->getResult(0),
                                          wtTbl_cmx.getOperation()->getResult(0), 0);

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stride_vec);

    auto kernel_padding = VPU::getPaddingAttr(ctx, padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                              padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcbuilder, mlir::ValueRange(barrier0.getBarrier()), mlir::ValueRange(barrier1.getBarrier()), loc,
            outputcmx_type, inputcmx.getOperation()->getResult(0), wtData_cmx.getOperation()->getResult(0),
            wtTbl_cmx.getOperation()->getResult(0), /*instruction_table_list*/ nullptr, /*spr_lookup_table*/ nullptr,
            actWindow_cmx.getOperation()->getResult(0), parent_inputcmx.getOperation()->getResult(0),
            parent_outputcmx.getOperation()->getResult(0), outputcmx.getOperation()->getResult(0),
            VPUIP::NCETaskType::DWCONV, filtersize, strides, kernel_padding, actChannelLength, nullptr,
            /*sp_pattern*/ nullptr);

    nceTask.addPPETask(funcbuilder);

    // Create DPU task for NCE task
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.getVariants().front(), builder.getListener());
    createDPUTaskOp(funcbuilder, variantbuilder, out_shape, in_shape, padding_vec, VPU::MPEMode::CUBOID_16x16);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                          outputcmx.getOperation()->getResult(0), funcoutput, 0);

    funcbuilder.create<mlir::func::ReturnOp>(loc, funcoutput);

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
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

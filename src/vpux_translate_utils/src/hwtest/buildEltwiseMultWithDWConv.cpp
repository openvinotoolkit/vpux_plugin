//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

mlir::DenseElementsAttr generateZeroPadForEltwiseMultWeights(ArrayRef<int64_t> wt_shape_padded, mlir::Type dtype,
                                                             mlir::MLIRContext* ctx) {
    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_shape_padded, dtype);

    if (auto qtype = dtype.dyn_cast<mlir::quant::QuantizedType>()) {
        wtData_ddr_valueType = (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed)
                                       ? mlir::RankedTensorType::get(wt_shape_padded, getSInt8Type(ctx))
                                       : mlir::RankedTensorType::get(wt_shape_padded, getUInt8Type(ctx));
    }

    auto vecSize = static_cast<size_t>(std::accumulate(wt_shape_padded.begin(), wt_shape_padded.end(),
                                                       static_cast<int64_t>(1), std::multiplies<int64_t>()));

    if (dtype.isF16()) {
        std::vector<vpux::type::float16> wt_vec(vecSize, 0);
        return mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef<vpux::type::float16>(wt_vec));
    } else if (dtype.isBF16()) {
        std::vector<vpux::type::bfloat16> wt_vec(vecSize, 0);
        return mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef<vpux::type::bfloat16>(wt_vec));
    } else {
        if (dtype.dyn_cast<mlir::quant::QuantizedType>().getFlags() & mlir::quant::QuantizationFlags::Signed) {
            std::vector<int8_t> wt_vec(vecSize, 0);
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef<int8_t>(wt_vec));
        } else {
            std::vector<uint8_t> wt_vec(vecSize, 0);
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef<uint8_t>(wt_vec));
        }
    }
}

void buildEltwiseMultWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                mlir::Type outputType) {
    using namespace VPUIP;
    auto* ctx = builder.getContext();

    const size_t num_func_args = 3;

    const auto arch = testDesc.getArchitecture();
    auto input = testDesc.getInputLayerList().front();
    auto weight = testDesc.getWeightLayers().front();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> weights_shape(weight.shape.begin(), weight.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(in_shape.size() >= 4, "buildEltwiseMultWithDwConv: Got input with rank less than 4");
    VPUX_THROW_UNLESS(out_shape.size() >= 4, "buildEltwiseMultWithDwConv: Got output with rank less than 4");
    VPUX_THROW_UNLESS(weights_shape.size() >= 4, "buildEltwiseMultWithDwConv: Got weights with rank less than 4");

    auto output_totalsize = totalTensorSize(out_shape, outputType);
    auto input_totalsize = totalTensorSize(in_shape, inputType);

    /*
        Notes on shapes
        ----------------
        - if non-flat input/output shapes are used in future test cases, ImplicitReshapes are needed to flatten the
       shapes beforehand.
        - NCE DWConv as EltwiseMult requires the input/output shapes w/ the elements in the C dim...  e.g., (1,32,1,1)
        - However, the ImplicitConcat to the zero-pad weights requires the elements to be in the W dim
        - Therefore, I changed the testcase input/output shapes to (1,1,1,32) & used DeclareTensor ops to
       implicitly-reshape the input/output of NCE

        Summary of Topology
        -------------------

        input           weights             zero_pad
        (1,1,1,32)      (1,1,1,32)          (1,15,1,32)
        INPUT           INPUT               GRAPHFILE
            |               |                    |
          NNDMA           NNDMA                NNDMA
            |                \                  /
            |                   ImplicitConcat
            |                         |
        input_cmx               weights_cmx             weightstable_ddr        activation_window_ddr
        (1,1,1,32)              (1,16,1,32)             (32,1,1,4)              (32,1,1,16)
        VPU_CMX_NN              VPU_CMX_NN              GRAPHFILE               GRAPHFILE
             |                       |                        |                       |
        ImplicitReshape         ImplicitReshape             NNDMA                   NNDMA
             |                       |                        |                       |
        input_nce               eweights_nc             weightstable_nce        activation_window_nce
        (1,32,1,1)              (32,1,1,16)             (32,1,1,4)              (32,1,1,16)
        VPU_CMX_NN              VPU_CMX_NN              VPU_CMX_NN               VPU_CMX_NN
           \                       \                         /                       /
            \_______________________\_______________________/_______________________/
                                                |
                                            NCEClusterTask
                                                |
                                            output_nce
                                            (1,32,1,1)
                                            VPU_CMX_NN
                                                |
                                            ImplicitReshape
                                                |
                                            output_cmx
                                            (1,1,1,32)
                                            VPU_CMX_NN
                                                |
                                              NNDMA
                                                |
                                            output
                                            (1,1,1,32)
                                            OUTPUT

    */
    // Weights concat
    SmallVector<int64_t> zero_pad_shape({1, 15, 1, in_shape[3]});
    SmallVector<int64_t> weights_pad_shape({1, 16, 1, in_shape[3]});

    // NCE input/output
    SmallVector<int64_t> input_nce_shape({1, in_shape[3], 1, 1});
    SmallVector<int64_t> weights_nce_shape({in_shape[3], 1, 1, 16});
    SmallVector<int64_t> output_nce_shape(input_nce_shape.begin(), input_nce_shape.end());

    std::vector<int64_t> filter_size({1, 1});
    std::vector<int64_t> stride_vec({1, 1});
    std::vector<int64_t> padding_vec({0, 0, 0, 0});

    auto weights_nce_totalsize = totalTensorSize(weights_nce_shape, weightsType);
    auto input1_leadingoffset = totalTensorSize({weights_shape[1]}, inputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto INPUT1_CMX_OFFSET = INPUT0_CMX_OFFSET + input_totalsize;
    const auto ZERO_PAD_CMX_OFFSET = INPUT1_CMX_OFFSET + input1_leadingoffset;
    const auto WEIGHTS_PAD_CMX_OFFSET = INPUT1_CMX_OFFSET;
    const auto ACTIVATIONWINDOW_CMX_OFFSET = WEIGHTS_PAD_CMX_OFFSET + weights_nce_totalsize;

    SmallVector<mlir::Type> inputTypes;
    inputTypes.reserve(num_func_args);
    auto inputParamType = getMemRefType(VPURT::BufferSection::NetworkInput, in_shape, inputType, DimsOrder::NHWC);
    auto weightsParamType = getMemRefType(VPURT::BufferSection::Constant, weights_shape, weightsType, DimsOrder::NHWC);
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(inputParamType);
    inputTypes.push_back(weightsParamType);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), printToString("eltwise_mult_{0}_{1}_{2}", inputType, weightsType, outputType),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcweights = func.getArgument(1);
    auto funcoutput = func.getArgument(2);

    // Tensor - input cmx
    auto input_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, in_shape, inputType,
                                           DimsOrder::NHWC, 0, INPUT0_CMX_OFFSET);

    auto padded_weights_type =
            getMemRefType(VPURT::BufferSection::CMX_NN, 0, weights_pad_shape, weightsType, DimsOrder::NHWC);
    auto padded_weights_strides = padded_weights_type.cast<vpux::NDTypeInterface>().getStrides();
    // Tensors - concat input/output
    auto weights_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, weights_shape, weightsType,
                                             DimsOrder::NHWC, padded_weights_strides, 0, WEIGHTS_PAD_CMX_OFFSET);
    auto zero_pad_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, zero_pad_shape, weightsType,
                                              DimsOrder::NHWC, padded_weights_strides, 0, ZERO_PAD_CMX_OFFSET);

    // Tensors - NCE input/output
    auto input_nce_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, input_nce_shape, inputType,
                                               DimsOrder::NHWC, 0, INPUT0_CMX_OFFSET);
    auto weights_nce_cmx =
            createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, weights_nce_shape, weightsType,
                                  DimsOrder::NHWC, padded_weights_strides, 0, WEIGHTS_PAD_CMX_OFFSET);
    auto output_nce_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, output_nce_shape, outputType,
                                                DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET);
    auto parent_input_nce_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, input_nce_shape,
                                                      inputType, DimsOrder::NHWC, 0, INPUT0_CMX_OFFSET);
    auto parent_output_nce_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, output_nce_shape,
                                                       outputType, DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET);

    // Tensor - output cmx
    auto output_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, out_shape, outputType,
                                            DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcbuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    // Barriers
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    auto finalBarrier = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++,
                                                                      testDesc.getWLMParams().isWLMPartialEnabled);

    auto wt_data_vals = generateZeroPadForEltwiseMultWeights(zero_pad_shape, weightsType, ctx);
    auto wt_data_attr = Const::ContentAttr::get(wt_data_vals);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }
    auto zero_pad_type = getMemRefType(VPURT::BufferSection::Constant, zero_pad_shape, weightsType, DimsOrder::NHWC);
    auto zero_pad_data = funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), zero_pad_type,
                                                              wt_data_attr.reorder(DimsOrder::NHWC));

    // Input DMAs
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, waitWLMBarrier, barrier0.getBarrier(), builder.getUnknownLoc(),
                                          funcinput, getTensorResult(input_cmx), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), barrier0.getBarrier(),
                                          builder.getUnknownLoc(), zero_pad_data, getTensorResult(zero_pad_cmx), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), barrier0.getBarrier(),
                                          builder.getUnknownLoc(), funcweights, getTensorResult(weights_cmx), 0);

    // Activation Window
    const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(
            VPU::NCESparsity::Mode::DW_CONV, ShapeRef(filter_size), stride_vec[1],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType,
            input_nce_shape[1]);
    mlir::IntegerAttr actChannelLength = funcbuilder.getI32IntegerAttr(checked_cast<int32_t>(bitPatternSize));

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(
            VPU::NCESparsity::Mode::DW_CONV, ShapeRef(filter_size), stride_vec[1],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType,
            input_nce_shape[1]);

    const auto sparsity_type = getUInt8Type(ctx);
    SmallVector<int64_t> sparsity_shape{1, 1, 1, static_cast<int64_t>(fakeSparsity.size())};

    const auto dataStorageType = mlir::RankedTensorType::get(sparsity_shape, sparsity_type);
    const auto sparsityAttr = mlir::DenseElementsAttr::get(dataStorageType, ArrayRef(fakeSparsity));

    auto act_window_ddr_memreftype =
            getMemRefType(VPURT::BufferSection::Constant, sparsity_shape, sparsity_type, DimsOrder::NHWC);

    auto act_window_totalsize = totalTensorSize(sparsity_shape, sparsity_type);
    auto act_window_totalsize_bytes = act_window_totalsize * sparsity_type.getIntOrFloatBitWidth() / CHAR_BIT;

    auto act_window_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), act_window_ddr_memreftype,
                                                 Const::ContentAttr::get(sparsityAttr).reorder(DimsOrder::NHWC));
    auto act_window_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, sparsity_shape,
                                                sparsity_type, DimsOrder::NHWC, 0, ACTIVATIONWINDOW_CMX_OFFSET);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), barrier0.getBarrier(),
                                          builder.getUnknownLoc(), getConstResult(act_window_ddr),
                                          getTensorResult(act_window_cmx), 0);

    // weights table ddr
    SmallVector<int64_t> weightstable_data_shape{output_nce_shape[1], 1, 1, 4};
    auto weightstable_ddr_memreftype = getMemRefType(VPURT::BufferSection::Constant, weightstable_data_shape,
                                                     builder.getIntegerType(32, /*isSigned=*/true), DimsOrder::NHWC);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(weightstable_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    auto weights_cmx_memreftype = getMemRefType(VPURT::BufferSection::CMX_NN, 0, weights_nce_shape, weightsType,
                                                DimsOrder::NHWC, padded_weights_strides);
    auto output_cmx_memreftype =
            getMemRefType(VPURT::BufferSection::CMX_NN, 0, output_nce_shape, outputType, DimsOrder::NHWC);

    auto weights_set_size = weights_cmx_memreftype.getShape()[1] * weights_cmx_memreftype.getShape()[2] *
                            weights_cmx_memreftype.getShape()[3];
    size_t elementsize_bytes = 0;
    if (auto qType = weights_cmx_memreftype.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        elementsize_bytes = qType.getStorageType().getIntOrFloatBitWidth() / CHAR_BIT;

    } else {
        elementsize_bytes = (weights_cmx_memreftype.getElementType().getIntOrFloatBitWidth()) / CHAR_BIT;
    }
    auto weights_set_nbytes = weights_set_size * elementsize_bytes;
    const auto sparsityPtrStep = 0;
    const std::vector<int32_t> weightstable_data_values_vec = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<int32_t>(WEIGHTS_PAD_CMX_OFFSET),
            static_cast<int32_t>(weights_set_nbytes), static_cast<int32_t>(ACTIVATIONWINDOW_CMX_OFFSET),
            sparsityPtrStep, arch, output_nce_shape[1], weightsType);

    auto weightstable_data_values = ArrayRef<int32_t>(weightstable_data_values_vec);
    auto weightstable_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, weightstable_data_values);

    auto weightstable_data_ddr = funcbuilder.create<Const::DeclareOp>(
            builder.getUnknownLoc(), weightstable_ddr_memreftype,
            Const::ContentAttr::get(weightstable_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor
    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + act_window_totalsize_bytes;
    auto weightstable_cmx = createDeclareTensorOp(funcbuilder, VPURT::BufferSection::CMX_NN, weightstable_data_shape,
                                                  builder.getIntegerType(32, /*isSigned=*/true), DimsOrder::NHWC, 0,
                                                  WEIGHTSTABLE_CMX_OFFSET);

    // weights table dma ddr->cmx
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), barrier0.getBarrier(),
                                          builder.getUnknownLoc(), getConstResult(weightstable_data_ddr),
                                          getTensorResult(weightstable_cmx), 0);

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stride_vec);
    auto kernel_padding = VPU::getPaddingAttr(ctx, padding_vec[PAD_NCETASK_LEFT], padding_vec[PAD_NCETASK_RIGHT],
                                              padding_vec[PAD_NCETASK_TOP], padding_vec[PAD_NCETASK_BOTTOM]);

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcbuilder, barrier0.getBarrier(), barrier1.getBarrier(), builder.getUnknownLoc(), output_cmx_memreftype,
            getTensorResult(input_nce_cmx), getTensorResult(weights_nce_cmx), getTensorResult(weightstable_cmx),
            /*instruction_table_list=*/nullptr, /*spr_lookup_table=*/nullptr, getTensorResult(act_window_cmx),
            getTensorResult(parent_input_nce_cmx), getTensorResult(parent_output_nce_cmx),
            getTensorResult(output_nce_cmx), NCETaskType::DWCONV, filtersize, strides, kernel_padding, actChannelLength,
            /*is_continued*/ nullptr, /*sp_pattern*/ nullptr);

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

    // DPU task for NCE task
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.getVariants().front(), builder.getListener());
    createDPUTaskOp(builder, variantbuilder, output_nce_shape, input_nce_shape, padding_vec,
                    VPU::MPEMode::CUBOID_16x16);

    // Output DMA
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, barrier1.getBarrier(), finalBarrier.getBarrier(),
                                          builder.getUnknownLoc(), getTensorResult(output_cmx), funcoutput, 0);

    // Return op
    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    // Compile
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(),
               {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(weights_shape), weightsType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(in_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux

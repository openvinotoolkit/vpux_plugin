//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/hwtest/hwtest_utils.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/custom_float.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <numeric>

namespace vpux {
namespace hwtest {

namespace {

mlir::Type parseType(mlir::OpBuilder builder, mlir::Type ty, const nb::QuantParams& qp) {
    if (!qp.present) {
        return ty;
    }

    auto intTy = ty.dyn_cast<mlir::IntegerType>();
    auto float8E5M2Ty = ty.dyn_cast<mlir::Float8E5M2Type>();
    auto float8E4M3Ty = ty.dyn_cast<mlir::Float8E4M3FNType>();
    if (!intTy && !float8E5M2Ty && !float8E4M3Ty) {
        return ty;
    }

    auto lowRange = qp.low_range;
    auto highRange = qp.high_range;
    auto outputType = builder.getF32Type();
    auto flags = intTy && intTy.isSigned() ? mlir::quant::QuantizationFlags::Signed : 0;
    if (float8E5M2Ty != nullptr) {
        lowRange = std::numeric_limits<vpux::type::float8_e5m2>::lowest();
        highRange = std::numeric_limits<vpux::type::float8_e5m2>::max();
        flags = mlir::quant::QuantizationFlags::Signed;
    } else if (float8E4M3Ty != nullptr) {
        lowRange = std::numeric_limits<vpux::type::float8_e4m3>::lowest();
        highRange = std::numeric_limits<vpux::type::float8_e4m3>::max();
        flags = mlir::quant::QuantizationFlags::Signed;
    }

    if (qp.scale.size() == 1) {
        return mlir::quant::UniformQuantizedType::get(flags, ty, outputType, qp.scale.front(), qp.zeropoint, lowRange,
                                                      highRange);
    }

    std::vector<int64_t> zeropoint(qp.scale.size(), qp.zeropoint);
    return mlir::quant::UniformQuantizedPerAxisType::get(flags, ty, outputType, qp.scale, zeropoint,
                                                         (DimsOrder::NCHW).dimPos(vpux::Dims4D::Act::C), lowRange,
                                                         highRange);
}

mlir::Type convertToMLIRType(mlir::OpBuilder builder, nb::DType dtype) {
    auto ctx = builder.getContext();
    switch (dtype) {
    case nb::DType::U4:
        return getUInt4Type(ctx);
    case nb::DType::U8:
        return getUInt8Type(ctx);
    case nb::DType::I4:
        return getSInt4Type(ctx);
    case nb::DType::I8:
        return getSInt8Type(ctx);
    case nb::DType::I32:
        return getSInt32Type(ctx);
    case nb::DType::BF8:
        return builder.getFloat8E5M2Type();
    case nb::DType::HF8:
        return builder.getFloat8E4M3FNType();
    case nb::DType::FP16:
        return builder.getF16Type();
    case nb::DType::FP32:
        return builder.getF32Type();
    case nb::DType::BF16:
        return builder.getBF16Type();
    default:
        throw std::domain_error{"Expected a valid data type"};
    }
}

template <class StorageType>
mlir::DenseElementsAttr generateWeights(std::ifstream& stream, mlir::RankedTensorType type, std::size_t elementsCount) {
    if (!stream) {
        auto generatedElements = std::vector<StorageType>(elementsCount);
        // have to add at least one non-zero element to make attribute non-splat. BitPack can't
        // work with splat tensors
        generatedElements[0] = 1;
        return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<StorageType>(generatedElements));
    }

    std::vector<StorageType> buffer(elementsCount);
    const auto expectedBytesCountToRead = elementsCount * sizeof(StorageType);
    // read as bytes since FP16/BFP16 are not supported by C++ standard
    stream.read(reinterpret_cast<char*>(buffer.data()), expectedBytesCountToRead);

    const auto actualBytesCountRead = static_cast<std::size_t>(stream.gcount());
    const auto state = stream.rdstate();

    if (expectedBytesCountToRead == actualBytesCountRead) {
        return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<StorageType>(buffer));
    }

    VPUX_THROW_UNLESS((state & std::ifstream::eofbit) == 0,
                      "Failed to read {0} bytes from weights file, read {1} bytes before EOF has been reached",
                      expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::failbit) == 0,
            "Failed to read {0} bytes from weights file, read {1} bytes before logical error on i/o operation occurred",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::badbit) == 0,
            "Failed to read {0} bytes from weights file, read {1} bytes before read error on i/o operation occurred",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW("Unexpected std::ifstream::rdstate value {}", state);
}

}  // namespace

mlir::DenseElementsAttr generateWeights(llvm::ArrayRef<int64_t> shape, mlir::Type type, mlir::MLIRContext* context,
                                        const char* weightsFileName) {
    VPUX_THROW_UNLESS(!shape.empty(), "generateWeights: Got empty shape");
    auto wtData_ddr_valueType = mlir::RankedTensorType::get(shape, type);
    const auto vecSize = static_cast<std::size_t>(
            std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

    if (auto qtype = type.dyn_cast_or_null<mlir::quant::QuantizedType>()) {
        type = mlir::quant::QuantizedType::castToStorageType(qtype);
        if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(shape, getSInt8Type(context));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(shape, getUInt8Type(context));
        }
    }

    std::ifstream stream{weightsFileName, std::ios::in | std::ios::binary};
    if (!stream) {
        std::cerr << "Warning: Unable to open weight data file " << weightsFileName << '\n';
    }

    if (type.isInteger(4)) {
        std::vector<int64_t> uintWrapperShape{shape.begin(), shape.end()};
        VPUX_THROW_UNLESS(!uintWrapperShape.empty(), "generateWeights: Got empty shape");
        // in NHWC tensor two int4 neighboring elements by C axis will be united into one uint8 element. So we have to
        // recalculate shape for uint wrapper tensor
        uintWrapperShape[Dims4D::Filter::OC.ind()] /= 2;

        const auto wrapperVecSize = static_cast<std::size_t>(std::accumulate(
                uintWrapperShape.begin(), uintWrapperShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

        auto uintWrapperValueType = mlir::RankedTensorType::get(uintWrapperShape, getUInt8Type(context));
        const auto weightsPacked = generateWeights<uint8_t>(stream, uintWrapperValueType, wrapperVecSize);
        std::vector<std::uint8_t> weightsUnpacked;
        weightsUnpacked.reserve(weightsPacked.size() * 2);
        for (const auto& elemPacked : weightsPacked.getValues<uint8_t>()) {
            const int8_t msn = (elemPacked & 0xf0) >> 4;
            const int8_t lsn = (elemPacked & 0x0f) >> 0;
            weightsUnpacked.push_back(lsn);
            weightsUnpacked.push_back(msn);
        }
        VPUX_THROW_UNLESS(
                weightsUnpacked.size() == vecSize,
                "Warning: count of elements in weights file {0} doesn't match with provided weights shape {1}",
                weightsUnpacked.size(), shape);

        if (type.isSignedInteger(4)) {
            return mlir::DenseElementsAttr::get(
                    wtData_ddr_valueType,
                    ArrayRef(reinterpret_cast<const int8_t*>(weightsUnpacked.data()), weightsUnpacked.size()));
        } else {
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef(weightsUnpacked));
        }
    } else if (type.isSignedInteger(8)) {
        return generateWeights<std::int8_t>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isInteger(8)) {
        return generateWeights<std::uint8_t>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isF16()) {
        return generateWeights<vpux::type::float16>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isBF16()) {
        return generateWeights<vpux::type::bfloat16>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isF32()) {
        return generateWeights<float>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isFloat8E5M2()) {
        return generateWeights<vpux::type::float8_e5m2>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isFloat8E4M3FN()) {
        return generateWeights<vpux::type::float8_e4m3>(stream, wtData_ddr_valueType, vecSize);
    } else {
        VPUX_THROW("Unexpected weights data type: {0}", type);
    }
}

vpux::Const::ContentAttr generateDefaultWeightsAttr(mlir::DenseElementsAttr weights, mlir::Type type) {
    auto weightsAttribute = vpux::Const::ContentAttr::get(weights);
    weightsAttribute = weightsAttribute.reorder(vpux::DimsOrder::OYXI);

    if (auto qty = type.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto quantizedType = vpux::changeStorageType(qty, weightsAttribute.getType().getElementType());
        weightsAttribute = weightsAttribute.quantCast(quantizedType);
        if (qty.getStorageType().isInteger(4)) {
            weightsAttribute = weightsAttribute.bitPack(4);
        }
    }

    return weightsAttribute;
}

std::vector<uint16_t> computeRsqrtLookupTable() {
    // LUT Data; Following values (fp16) are synced with numerics-bench rsqrt LUT

    // Saturation/bypass regions, from l-to-r:
    // upper threshold (ru)
    // lower threshold (rl)
    // saturation value (y value to be set if x value in above interval)
    const std::vector<std::vector<uint16_t>> rsqrtSaturationTableLut = {
            {1024, 0, 65535},      {31744, 30719, 0},     {64512, 32768, 65535}, {65535, 65535, 65534},
            {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}};

    // Linear funcion approximations, from l-to-r:
    // slope
    // intercept
    const std::vector<std::vector<uint16_t>> rsqrtSlopeInterceptLut = {
            {51561, 16808}, {51443, 16765}, {51342, 16725}, {51253, 16689}, {51151, 16655}, {51013, 16624},
            {50890, 16595}, {50780, 16568}, {50682, 16542}, {50593, 16519}, {50512, 16496}, {50438, 16475},
            {50371, 16455}, {50310, 16436}, {50253, 16418}, {50201, 16400}, {53158, 17408}, {52993, 17347},
            {52849, 17291}, {52724, 17239}, {52613, 17192}, {52516, 17148}, {52429, 17107}, {52351, 17068},
            {52282, 17032}, {52213, 16998}, {52099, 16967}, {51995, 16937}, {51900, 16908}, {51813, 16881},
            {51733, 16856}, {51660, 16831}};

    // Table containing base addresses and exponents for LUT address generation
    // Each numer a single fp16 data point, represented by :|<- n->|<- base address ->|
    //                                                      |6 bits|<-   10 bits     >|
    //
    //  - base/start LUT address for each input exponent stored in LUT_CFG table
    //  - n: number of linear segments (l), a power of 2 i.e. l = 2^n^ , where n = 0, 1, 2, 3, 4,...

    // Address generation (S-sign, E-exponent, M-mantissa):
    //    fp32 input:     S    EEEEEEEE    MMMMMMMMMMMMMMMMMMMMMM
    //                         |      |    |...|- n MSB of mantissa
    //                     exp16=exp32-112    |
    //                             V          V
    //                          LUT_CFG ----> + -> LUT address
    //                                b. addr
    const std::vector<uint16_t> rsqrtCfgLUT = {137, 133, 133, 132, 132, 131, 131, 130, 130, 129, 129, 128, 128,
                                               127, 127, 126, 126, 125, 125, 124, 124, 123, 123, 122, 122, 121,
                                               121, 120, 120, 119, 119, 0,   255, 134, 133, 133, 132, 132, 131,
                                               131, 130, 130, 129, 129, 128, 128, 127, 127, 126, 126, 125, 125,
                                               124, 124, 123, 123, 122, 122, 121, 121, 120, 120, 119, 0};

    // LUT creation/formatting
    std::vector<uint16_t> rsqrtSprLookUpTableContent;

    // saturation regions
    for (uint32_t index = 0; index < NUM_LUT_SAT_REGS; ++index) {
        rsqrtSprLookUpTableContent.push_back(rsqrtSaturationTableLut[index][2]);
        rsqrtSprLookUpTableContent.push_back(rsqrtSaturationTableLut[index][1]);
        rsqrtSprLookUpTableContent.push_back(rsqrtSaturationTableLut[index][0]);
    }

    // SYM_B = 0 (pos 128)
    // RCP = 0 (pos 129)
    // RSQRT = 1 (pos 130)
    rsqrtSprLookUpTableContent.push_back(0X0004);

    // reserved region
    for (uint32_t index = 0; index < NUM_LUT_RESERVED_REGS; ++index) {
        rsqrtSprLookUpTableContent.push_back(0x0000);
    }

    // lut cfg region
    for (uint32_t index = 0; index < NUM_LUT_CFG_REGS; ++index) {
        rsqrtSprLookUpTableContent.push_back(rsqrtCfgLUT[index]);
    }

    // lut data region
    if (size(rsqrtSlopeInterceptLut) > NUM_LUT_DATA_REGS)
        VPUX_THROW("Size of lut data {0} is bigger than allowed size of {1}", size(rsqrtSlopeInterceptLut),
                   NUM_LUT_DATA_REGS);
    for (uint32_t index = 0; index < size(rsqrtSlopeInterceptLut); ++index) {
        rsqrtSprLookUpTableContent.push_back(rsqrtSlopeInterceptLut[index][0]);
        rsqrtSprLookUpTableContent.push_back(rsqrtSlopeInterceptLut[index][1]);
    }

    return rsqrtSprLookUpTableContent;
}

std::size_t totalTensorSize(llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
    auto qType = elementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
    auto qPerChType = elementType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();

    if (qType) {
        elementType = qType.getStorageType();
    } else if (qPerChType) {
        elementType = qPerChType.getStorageType();
    }

    size_t numBits = elementType.getIntOrFloatBitWidth();

    const auto totalSize =
            std::accumulate(shape.begin(), shape.end(), static_cast<std::int64_t>(1), std::multiplies<std::int64_t>());
    const auto totalBits = totalSize * numBits;
    VPUX_THROW_UNLESS(totalBits % CHAR_BIT == 0, "Tensors size is not allligned to Byte");
    return static_cast<std::size_t>(totalBits / CHAR_BIT);
}

std::vector<int64_t> convertNBPadtoNCETaskPad(const std::array<int64_t, 4>& nb_pad) {
    std::vector<int64_t> ncetask_pad(nb_pad.size());

    ncetask_pad[PAD_NCETASK_LEFT] = nb_pad[PAD_NB_LEFT];
    ncetask_pad[PAD_NCETASK_RIGHT] = nb_pad[PAD_NB_RIGHT];
    ncetask_pad[PAD_NCETASK_TOP] = nb_pad[PAD_NB_TOP];
    ncetask_pad[PAD_NCETASK_BOTTOM] = nb_pad[PAD_NB_BOTTOM];

    return ncetask_pad;
}

mlir::Type parseInputType(mlir::OpBuilder builder, const nb::InputLayer& input) {
    return parseType(builder, convertToMLIRType(builder, input.dtype), input.qp);
}

mlir::Type parseOutputType(mlir::OpBuilder builder, const nb::OutputLayer& output) {
    return parseType(builder, convertToMLIRType(builder, output.dtype), output.qp);
}

mlir::Type parseWeightsType(mlir::OpBuilder builder, const nb::WeightLayer& weight) {
    return parseType(builder, convertToMLIRType(builder, weight.dtype), weight.qp);
}

void buildCNNOp(mlir::OpBuilder& builder, llvm::StringRef mainFuncName, llvm::ArrayRef<mlir::Type> inputs,
                llvm::ArrayRef<mlir::Type> outputs) {
    return buildCNNOp(builder, mainFuncName, inputs, outputs, {});
}

void buildCNNOp(mlir::OpBuilder& builder, llvm::StringRef mainFuncName, llvm::ArrayRef<mlir::Type> inputs,
                llvm::ArrayRef<mlir::Type> outputs, llvm::ArrayRef<ProfilingDataSection> profilingSections) {
    const auto enableProfiling = !profilingSections.empty();
    const auto mainFuncNameAttr = mlir::SymbolRefAttr::get(builder.getContext(), mainFuncName);
    auto cnnOp = builder.create<IE::CNNNetworkOp>(builder.getUnknownLoc(), mainFuncNameAttr, enableProfiling);
    cnnOp.getInputsInfo().emplaceBlock();
    cnnOp.getOutputsInfo().emplaceBlock();
    if (enableProfiling) {
        for (auto& section : cnnOp.getProfilingOutputsInfo()) {
            section.emplaceBlock();
        }
    }

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.getInputsInfo().front(), builder.getListener());
    for (auto input : enumerate(inputs)) {
        auto inputType = input.value().cast<mlir::ShapedType>();
        auto quantized = inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
        auto quantizedPerCh = inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();

        if (quantized) {
            inputType = inputType.clone(quantized.getStorageType());
        } else if (quantizedPerCh) {
            inputType = inputType.clone(quantizedPerCh.getStorageType());
        }

        const auto inputName = printToString("input_{0}", input.index());
        const auto nameAttr = builder.getStringAttr(inputName);
        const auto userTypeAttr = mlir::TypeAttr::get(inputType);
        inputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr,
                                                 /*profilingSectionsCount=*/0);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.getOutputsInfo().front(), builder.getListener());
    for (auto output : enumerate(outputs)) {
        auto outputType = output.value().cast<mlir::ShapedType>();
        auto quantized = outputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
        auto quantizedPerCh = outputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();

        if (quantized) {
            outputType = outputType.clone(quantized.getStorageType());
        } else if (quantizedPerCh) {
            outputType = outputType.clone(quantizedPerCh.getStorageType());
        }

        const auto resultName = printToString("output_{0}", output.index());
        const auto nameAttr = builder.getStringAttr(resultName);
        const auto userTypeAttr = mlir::TypeAttr::get(outputType);
        outputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr,
                                                  /*profilingSectionsCount=*/0);
    }

    if (enableProfiling) {
        auto& profilingOutputsInfo = cnnOp.getProfilingOutputsInfo().front();
        auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&profilingOutputsInfo.front());

        const auto ctx = builder.getContext();

        const auto lastSection = profilingSections[profilingSections.size() - 1];
        const auto totalSize = lastSection.offset + lastSection.size;

        auto newOutputResult =
                mlir::MemRefType::get({static_cast<int64_t>(totalSize / sizeof(uint32_t))}, getUInt32Type(ctx));
        auto newOutputShapedType = newOutputResult.cast<vpux::NDTypeInterface>();
        auto outputUserResult = getTensorType(newOutputShapedType.getShape(), newOutputShapedType.getElementType(),
                                              newOutputShapedType.getDimsOrder(), nullptr);

        auto dataInfo = userInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx),
                                                               mlir::StringAttr::get(ctx, "profilingOutput"),
                                                               mlir::TypeAttr::get(outputUserResult),
                                                               /*profilingSectionsCount=*/1);
        dataInfo.getSections().front().emplaceBlock();

        auto sectionsBuilder =
                mlir::OpBuilder::atBlockBegin(&dataInfo.getSections().front().front(), builder.getListener());
        for (auto section : profilingSections) {
            sectionsBuilder.create<VPUIP::ProfilingSectionOp>(
                    mlir::UnknownLoc::get(ctx), getIntAttr(ctx, section.execType), getIntAttr(ctx, section.offset),
                    getIntAttr(ctx, section.size));
        }
    }
}

mlir::DenseElementsAttr splitWeightsOverC(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape, mlir::Type dtype,
                                          mlir::MLIRContext* ctx, size_t start_C, size_t end_C) {
    auto qType = dtype.dyn_cast<mlir::quant::UniformQuantizedType>();
    if (!((dtype.isF16() || (qType && qType.getStorageType().isUnsignedInteger(8)))))
        throw std::domain_error{
                printToString("splitWeightsOverC only supports weight data type fp16 or uint8; got {0}", dtype)};

    if (dtype.isF16()) {
        vpux::type::float16 elementType = 0;
        return splitWeightsOverCLoop(wt_vec, wt_shape, dtype, elementType, ctx, start_C, end_C);
    } else {
        uint8_t elementType = 0;
        return splitWeightsOverCLoop(wt_vec, wt_shape, dtype, elementType, ctx, start_C, end_C);
    }
}

template <typename T>
mlir::DenseElementsAttr splitWeightsOverCLoop(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape,
                                              mlir::Type dtype, T, mlir::MLIRContext* ctx, size_t start_C,
                                              size_t end_C) {
    SmallVector<int64_t> original_shape(wt_shape.begin(), wt_shape.end());

    // Weights from NumericsBench in KCHW
    // For stream-over-C, need to extract weights[startC:endC]
    int64_t K = original_shape[0];
    int64_t C = original_shape[1];
    int64_t H = original_shape[2];
    int64_t W = original_shape[3];
    int64_t new_C = end_C - start_C;

    auto wt_full_itr = wt_vec.getValues<T>();
    std::vector<T> wt_full(wt_full_itr.begin(), wt_full_itr.end());
    const llvm::SmallVector<int64_t> wt_partial_shape({K, new_C, H, W});
    size_t vecSize = static_cast<size_t>(std::accumulate(wt_partial_shape.begin(), wt_partial_shape.end(),
                                                         static_cast<int64_t>(1), std::multiplies<int64_t>()));
    std::vector<T> wt_partial(vecSize);

    for (int64_t k = 0; k < K; k++)
        for (int64_t c = 0; c < new_C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    // auto old_c = c+start_C;
                    auto old_offset = k * C * H * W + (c + start_C) * H * W + h * W + w;
                    auto new_offset = k * new_C * H * W + c * H * W + h * W + w;
                    wt_partial[new_offset] = wt_full[old_offset];
                }

    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, dtype);
    if (auto qtype = dtype.dyn_cast<mlir::quant::QuantizedType>()) {
        dtype = mlir::quant::QuantizedType::castToStorageType(qtype);
        if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, getSInt8Type(ctx));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, getUInt8Type(ctx));
        }
    }

    auto wt_data_values = ArrayRef<T>(wt_partial);
    auto wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
    return wt_data_vals;
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, ArrayRef<int64_t> shape, mlir::Type elemType,
                               DimsOrder order) {
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, VPURT::getMemoryKind(section));
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, symbolAttr);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ShapeRef shape, mlir::Type elemType,
                               DimsOrder order) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(shape, elemType, order, symbolAttr);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, ArrayRef<int64_t> shape, mlir::Type elemType,
                               DimsOrder order, StridesRef strides) {
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, VPURT::getMemoryKind(section), strides);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order, StridesRef strides) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, symbolAttr, strides);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order, StridesRef strides,
                               VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, symbolAttr, strides, swizzlingSchemeAttr);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   int64_t locale, size_t offset) {
    const auto type = getMemRefType(section, locale, shape, elemType, order);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, VPURT::BufferSection section,
                                                   ShapeRef shape, mlir::Type elemType, DimsOrder order, int64_t locale,
                                                   size_t offset) {
    const auto type = getMemRefType(section, locale, shape, elemType, order);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   StridesRef strides, int64_t locale, size_t offset) {
    const auto type = getMemRefType(section, locale, shape, elemType, order, strides);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   StridesRef strides, int64_t locale, size_t offset,
                                                   VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr) {
    const auto type = getMemRefType(section, locale, shape, elemType, order, strides, swizzlingSchemeAttr);
    const auto swizzlingKey = swizzlingSchemeAttr.getKey().getInt();
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset, swizzlingKey);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::Type type,
                                                   VPURT::BufferSection section, size_t offset) {
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::MemRefType type,
                                                   VPURT::BufferSection section, int64_t locale, size_t offset) {
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::Type type,
                                                   VPURT::BufferSection section, ArrayRef<int64_t> locale,
                                                   size_t offset) {
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

mlir::OpResult getTensorResult(VPURT::DeclareBufferOp op) {
    return op.getOperation()->getResult(0);
}

mlir::OpResult getConstResult(vpux::Const::DeclareOp op) {
    return op.getOperation()->getResult(0);
}

vpux::VPUIP::DPUTaskOp createDPUTaskOp(mlir::OpBuilder builder, mlir::OpBuilder variantbuilder,
                                       ArrayRef<int64_t> outputShape, ArrayRef<int64_t> inputShape,
                                       const std::vector<int64_t>& paddingVec, VPU::MPEMode mpeMode,
                                       int64_t clusterId) {
    std::vector<int32_t> startVec{0, 0, 0};
    auto start = getIntArrayAttr(builder, startVec);
    const auto outEnd = getIntArrayAttr(
            builder, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto inEnd = getIntArrayAttr(
            builder, std::vector<std::int64_t>{inputShape[3] - 1, inputShape[2] - 1, inputShape[1] - 1});
    auto pad = VPU::getPaddingAttr(builder.getContext(), paddingVec[PAD_NCETASK_LEFT], paddingVec[PAD_NCETASK_RIGHT],
                                   paddingVec[PAD_NCETASK_TOP], paddingVec[PAD_NCETASK_BOTTOM]);

    auto dpuTask =
            variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), start, outEnd, start, inEnd, pad, mpeMode,
                                                    getIntAttr<int64_t>(builder.getContext(), clusterId));

    return dpuTask;
}

vpux::DimsOrder oduPermutationToLayout(const MVCNN::Permutation oduPermutation) {
    switch (oduPermutation) {
    case MVCNN::Permutation::Permutation_ZXY:
        return vpux::DimsOrder::NHWC;
    case MVCNN::Permutation::Permutation_ZYX:
        return vpux::DimsOrder::fromCode(0x1432);  // NWHC
    case MVCNN::Permutation::Permutation_YZX:
        return vpux::DimsOrder::fromCode(0x1423);  // NWCH
    case MVCNN::Permutation::Permutation_YXZ:
        return vpux::DimsOrder::fromCode(0x1243);  // NCWH
    case MVCNN::Permutation::Permutation_XZY:
        return vpux::DimsOrder::NHCW;
    case MVCNN::Permutation::Permutation_XYZ:
        return vpux::DimsOrder::NCHW;
    default:
        return vpux::DimsOrder::NHWC;
    }
}

vpux::Dim getInnermostDim(const vpux::DimsOrder& order) {
    return order.toDim(MemDim(order.numDims() - 1));
}

// Get padding for each cluster based on global padding, segmentation type and current cluster
VPU::PaddingAttr getMulticlusteringPaddings(mlir::MLIRContext* ctx, const int64_t cluster, const int64_t numClusters,
                                            nb::SegmentationType segmentationType, VPU::PaddingAttr globalPadding,
                                            SmallVector<std::int64_t> clustersPerDim) {
    const auto noPad = getIntAttr<int64_t>(ctx, 0);
    switch (segmentationType) {
    case nb::SegmentationType::SOH: {
        if (cluster == 0) {
            return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), globalPadding.getRight(), globalPadding.getTop(),
                                         noPad);
        }

        if (cluster == numClusters - 1) {
            return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), globalPadding.getRight(), noPad,
                                         globalPadding.getBottom());
        }

        return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), globalPadding.getRight(), noPad, noPad);
    }
    case nb::SegmentationType::SOW: {
        if (cluster == 0) {
            return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), noPad, globalPadding.getTop(),
                                         globalPadding.getBottom());
        }

        if (cluster == numClusters - 1) {
            return VPU::PaddingAttr::get(ctx, noPad, globalPadding.getRight(), globalPadding.getTop(),
                                         globalPadding.getBottom());
        }

        return VPU::PaddingAttr::get(ctx, noPad, noPad, globalPadding.getTop(), globalPadding.getBottom());
    }
    case nb::SegmentationType::SOK:
        return globalPadding;
    case nb::SegmentationType::SOHW: {
        if (clustersPerDim.size() != 2) {
            VPUX_THROW("SOHW needs clustersPerDim size to pe 2");
        }
        const auto heightNumClusters = clustersPerDim[0];
        const auto widthNumClusters = clustersPerDim[1];

        const auto idxH = cluster / widthNumClusters;
        const auto idxW = cluster % widthNumClusters;

        const auto top = (idxH == 0) ? globalPadding.getTop() : noPad;
        const auto bottom = (idxH == heightNumClusters - 1) ? globalPadding.getBottom() : noPad;
        const auto left = (idxW == 0) ? globalPadding.getLeft() : noPad;
        const auto right = (idxW == widthNumClusters - 1) ? globalPadding.getRight() : noPad;

        return VPU::PaddingAttr::get(ctx, left, right, top, bottom);
    }
    case nb::SegmentationType::SOHK: {
        if (clustersPerDim.size() != 2) {
            VPUX_THROW("SOHW needs clustersPerDim size to pe 2");
        }

        const auto heightNumClusters = clustersPerDim[0];
        const auto channelsNumClusters = clustersPerDim[1];

        const auto idxH = cluster / channelsNumClusters;

        const auto top = (idxH == 0) ? globalPadding.getTop() : noPad;
        const auto bottom = (idxH == heightNumClusters - 1) ? globalPadding.getTop() : noPad;

        return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), globalPadding.getRight(), top, bottom);
    }
    default:
        VPUX_THROW("Segmentation type unsupported {0}", nb::to_string(segmentationType));
    };

    return VPU::PaddingAttr();
}

}  // namespace hwtest
}  // namespace vpux

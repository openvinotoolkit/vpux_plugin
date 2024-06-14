//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Dialect/Quant/QuantTypes.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/act_shave_op.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_string.hpp"

using namespace vpux;

namespace vpux {
namespace hwtest {

// without Tiling:
//                 [input]
//                    |
//               [inCMXBuff]
//                    |
//                (M2iTask)
//                    |
//              [outCMXBuff]
//                    |
//                 [output]

// with Tiling:
//    Run 4 M2i Ops for individual tensor tiles. Processing is not done in parralel/
//
//   tensor tiling configuration:
//       -------------
//       |tile0|tile1|
//       |-----|-----|
//       |tile2|tile3|
//       -------------
//
//                            [input]                     Splitting input into 1/4*Input tiles in networkInput buffers
//            /            /           \            \      ~(NNDMAOps with input strides)
//      [netInBuff0] [netInBuff1] [netInBuff2] [netInBuff3]
//            |            |            |            |
//      [inCMXBuff0] [inCMXBuff1] [inCMXBuff2] [inCMXBuff3]
//            |            |            |            |
//       (M2iTask0)   (M2iTask1)   (M2iTask2)   (M2iTask3)
//            |            |            |            |
//     [outCMXBuff0] [outCMXBuff1] [outCMXBuff2] [outCMXBuff3]
//            |            |            |            |
//      [netoutBuff0] [netOutBuff1] [netOutBuff2] [netoutBuff3]      Adding back outputs into single output (whole
//      tensor):
//            \            \            /            /      ~(NNDMAOps with output strides)
//                            [output]
//

VPU::M2iColorFmt getM2iFmt(nb::M2iFmt fmt) {
    if (fmt == nb::M2iFmt::SP_NV12_8)
        return VPU::M2iColorFmt::SP_NV12_8;
    if (fmt == nb::M2iFmt::PL_YUV420_8)
        return VPU::M2iColorFmt::PL_YUV420_8;
    if (fmt == nb::M2iFmt::IL_RGB888 || fmt == nb::M2iFmt::IL_BGR888)
        return VPU::M2iColorFmt::IL_RGB888;
    if (fmt == nb::M2iFmt::PL_RGB24)
        return VPU::M2iColorFmt::PL_RGB24;
    if (fmt == nb::M2iFmt::PL_FP16_RGB)
        return VPU::M2iColorFmt::PL_FP16_RGB;
    VPUX_THROW("getM2iFmt unsupported fmt: {0}", static_cast<uint32_t>(fmt));
}

VPU::M2iInterp getM2iInterp(nb::M2iInterp interp) {
    if (interp == nb::M2iInterp::NEAREST)
        return VPU::M2iInterp::NEAREST;
    if (interp == nb::M2iInterp::BILINEAR)
        return VPU::M2iInterp::BILINEAR;

    VPUX_THROW("getM2iInterp unsupported interpolation: {0}", static_cast<uint32_t>(interp));
}

DimsOrder getTensorOrder(nb::M2iFmt fmt) {
    if (fmt == nb::M2iFmt::PL_YUV420_8 || fmt == nb::M2iFmt::SP_NV12_8 || fmt == nb::M2iFmt::IL_RGB888 ||
        fmt == nb::M2iFmt::IL_BGR888)
        return DimsOrder::NHWC;
    if (fmt == nb::M2iFmt::PL_FP16_RGB || fmt == nb::M2iFmt::PL_RGB24)
        return DimsOrder::NCHW;
    VPUX_THROW("getM2iFmt unsupported fmt: {0}", static_cast<uint32_t>(fmt));
}

vpux::Dim getWidthFromOrder(DimsOrder order) {
    // In order to compute scale factor for resize, width and height must be extracted from input shape
    if (order == DimsOrder::NCHW)
        // NCHW = 0x1234 where W.ind() corresponds to W
        return vpux::Dims4D::Act::W;
    if (order == DimsOrder::NHWC)
        // NCHW = 0x1342 where H.ind() corresponds to W
        return vpux::Dims4D::Act::H;
    VPUX_THROW("Tensor order not supported by m2i");
}

vpux::Dim getHeightFromOrder(DimsOrder order) {
    // In order to compute scale factor for resize, width and height must be extracted from input shape
    if (order == DimsOrder::NCHW)
        // NCHW = 0x1234 where H.ind() corresponds to H
        return vpux::Dims4D::Act::H;
    if (order == DimsOrder::NHWC)
        // NCHW = 0x1342 where C.ind() corresponds to H
        return vpux::Dims4D::Act::C;
    VPUX_THROW("Tensor order not supported by m2i");
}

bool isM2iChromaOrderInverted(nb::M2iFmt fmt) {
    if (fmt == nb::M2iFmt::SP_NV12_8 || fmt == nb::M2iFmt::PL_YUV420_8 || fmt == nb::M2iFmt::IL_RGB888 ||
        fmt == nb::M2iFmt::PL_RGB24 || fmt == nb::M2iFmt::PL_FP16_RGB)
        return false;
    if (fmt == nb::M2iFmt::IL_BGR888)
        return true;
    VPUX_THROW("getM2iFmt unsupported fmt: {0}", static_cast<uint32_t>(fmt));
}

bool isM2iLumaOrderInverted(nb::M2iFmt fmt) {
    if (fmt == nb::M2iFmt::SP_NV12_8 || fmt == nb::M2iFmt::PL_YUV420_8 || fmt == nb::M2iFmt::IL_RGB888 ||
        fmt == nb::M2iFmt::PL_RGB24 || fmt == nb::M2iFmt::PL_FP16_RGB || fmt == nb::M2iFmt::IL_BGR888)
        return false;
    VPUX_THROW("getM2iFmt unsupported fmt: {0}", static_cast<uint32_t>(fmt));
}

size_t getCMXTileOffset(SmallVector<size_t>& offsets, size_t CMXTileIdx, size_t incrementValue) {
    if (CMXTileIdx >= offsets.size()) {
        VPUX_THROW("CMX tile index {0} is bigger than {1} available tiles", CMXTileIdx, offsets.size());
    }

    size_t offset = offsets[CMXTileIdx];
    offsets[CMXTileIdx] += incrementValue;
    return offset;
}

void buildM2iTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    auto input = testDesc.getInputLayerList().front();
    auto output = testDesc.getOutputLayers().front();
    auto params = testDesc.getM2iLayer();
    auto profilingParams = testDesc.getProfilingParams();

    // Drop quantization info
    if (inputType.dyn_cast<mlir::quant::QuantizedType>()) {
        inputType = mlir::quant::QuantizedType::castToStorageType(inputType);
    }
    if (outputType.dyn_cast<mlir::quant::QuantizedType>()) {
        outputType = mlir::quant::QuantizedType::castToStorageType(outputType);
    }

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildM2iTest: got empty inputShape");
    VPUX_THROW_UNLESS(!outShape.empty(), "buildM2iTest: got empty outputShape");

    if (params.doNorm) {
        VPUX_THROW_UNLESS(params.normCoefs.size() > 0, "buildM2iTest: norm coeffs missing");
    }

    int64_t m2iProfilingBufferSizeBytes = 0;
    int64_t totalProfilingBufferSizeBytes = 0;
    if (profilingParams.m2iProfilingEnabled) {
        m2iProfilingBufferSizeBytes = HWP_M2I_BYTES_PER_ENTRY;
        totalProfilingBufferSizeBytes += m2iProfilingBufferSizeBytes;
    }
    SmallVector<int64_t> profilingOutputShapeUI64{totalProfilingBufferSizeBytes / 8};
    SmallVector<int64_t> profilingOutputShapeUI8{totalProfilingBufferSizeBytes};

    auto inputTotalSize = totalTensorSize(inShape, inputType);

    const auto inputTensorOrder = getTensorOrder(params.iFmt);
    const auto outputTensorOrder = getTensorOrder(params.oFmt);

    const auto inWidthDim = getWidthFromOrder(inputTensorOrder);
    const auto inHeightDim = getHeightFromOrder(inputTensorOrder);
    const auto outWidthDim = getWidthFromOrder(outputTensorOrder);
    const auto outHeightDim = getHeightFromOrder(outputTensorOrder);

    const auto inWidthIndex = inWidthDim.ind();
    const auto inHeightIndex = inHeightDim.ind();
    const auto outWidthIndex = outWidthDim.ind();
    const auto outHeightIndex = outHeightDim.ind();

    const auto inputWidth = inShape[inWidthIndex];
    const auto inputHeight = inShape[inHeightIndex];
    const auto outputWidth = outShape[outWidthIndex];
    const auto outputHeight = outShape[outHeightIndex];

    const auto getActualImgHeight = [](size_t h, nb::M2iFmt fmt) -> size_t {
        if (fmt == nb::M2iFmt::PL_YUV420_8 || fmt == nb::M2iFmt::SP_NV12_8) {
            return (h * 2) / 3;
        }
        return h;
    };
    const auto actualInputHeight = getActualImgHeight(inputHeight, params.iFmt);
    const auto actualOutputHeight = getActualImgHeight(outputHeight, params.oFmt);

    auto scaleFactorWidthOrig =
            VPU::getM2iFixedPointScaleFactor(static_cast<uint32_t>(inputWidth), static_cast<uint32_t>(outputWidth),
                                             VPU::M2I_SCALE_FACTOR_FRACTIONAL_BITS);
    auto scaleFactorHeightOrig = VPU::getM2iFixedPointScaleFactor(static_cast<uint32_t>(actualInputHeight),
                                                                  static_cast<uint32_t>(actualOutputHeight),
                                                                  VPU::M2I_SCALE_FACTOR_FRACTIONAL_BITS);

    const auto maxM2iHwBlockInputSize = 2048 * 2048;
    auto nrOfInputTiles = 1;

    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Type> outputTypes;

    auto inputParamType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NCHW);
    inputTypes.push_back(inputParamType);

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NCHW);
    inputTypes.push_back(outputParamType);
    outputTypes.push_back(outputParamType);

    if (profilingParams.profilingEnabled()) {
        auto profParamType = getMemRefType(VPURT::BufferSection::ProfilingOutput, profilingOutputShapeUI64,
                                           getUInt64Type(ctx), DimsOrder::C);
        inputTypes.push_back(profParamType);
        outputTypes.push_back(profParamType);
    }

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), ArrayRef(outputTypes));

    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), llvm::formatv("m2i_test").str(), funcType,
                                                   builder.getStringAttr("private"), /*arg_attrs=*/nullptr,
                                                   /*res_attrs=*/nullptr);

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    auto normCoefs = params.doNorm ? getFPArrayAttr(funcBuilder, params.normCoefs) : nullptr;

    // Build VPUIP ops
    auto funcInput0 = func.getArgument(0);
    auto funcOutput = func.getArgument(1);
    auto funcProfOutput = profilingParams.profilingEnabled() ? func.getArgument(2) : nullptr;

    const auto origInputTypeIf = funcInput0.getType().cast<NDTypeInterface>();
    const auto inputStrides = origInputTypeIf.getStrides();

    const auto origOutputTypeIf = funcOutput.getType().cast<NDTypeInterface>();
    const auto outputStrides = origOutputTypeIf.getStrides();

    std::vector<SmallVector<int64_t>> inShapeTile{inShape};
    std::vector<SmallVector<int64_t>> outShapeTile{outShape};

    std::vector<uint32_t> offsetFixedPointsWidth;
    std::vector<uint32_t> offsetFixedPointsHeight;

    std::vector<Byte> inSliceOffsets{0};
    std::vector<Byte> outSliceOffsets{0};

    // Tiling set up
    SmallVector<size_t> CMXTileOffsets = {0};
    int availableCMXTiles = 1;

    if (params.doTiling) {
        nrOfInputTiles = 4;

        if (testDesc.getArchitecture() == vpux::VPU::ArchKind::NPU40XX) {
            availableCMXTiles = 4;
            CMXTileOffsets = SmallVector<size_t>{0, 0, 0, 0};
        }

        // individual tile shapes (WxH values) are based and calculated on the initial tensor shapes
        inShapeTile = {inShape, inShape, inShape, inShape};
        outShapeTile = {outShape, outShape, outShape, outShape};

        double scaleFactorWidth = (double)inputWidth / outputWidth;
        double scaleFactorHeight = (double)inputHeight / outputHeight;

        const auto outputWidthTile0 = outputWidth / 2;
        const auto outputHeightTile0 = outputHeight / 2;

        if (params.interp == nb::M2iInterp::BILINEAR) {
            double tile_offset_x = 0, tile_offset_y = 0, intpart = 0;

            const auto inputWidthTile0 = static_cast<int64_t>(std::ceil(scaleFactorWidth * outputWidthTile0));
            const auto inputHeightTile0 = static_cast<int64_t>(std::floor(scaleFactorHeight * outputHeightTile0));

            const auto outputWidthTile1 = outputWidth - outputWidthTile0;
            const auto outputHeightTile1 = outputHeight - outputHeightTile0;

            const auto tile1_x = static_cast<int>(std::floor(scaleFactorWidth * outputWidthTile1));
            const auto inputWidthTile1 = static_cast<int>(inputWidth - tile1_x);
            // Downsampling
            if (scaleFactorWidth >= 1) {
                tile_offset_x = std::modf((double)scaleFactorWidth * outputWidthTile0, &intpart);
            } else {
                // Upsampling
                tile_offset_x = std::modf((double)scaleFactorWidth * outputWidthTile0, &intpart) + 1;
            }
            const auto inputHeightTile1 = static_cast<int64_t>(inputHeight - inputHeightTile0);
            // Downsampling
            if (scaleFactorHeight >= 1) {
                tile_offset_y = std::modf((double)scaleFactorHeight * outputHeightTile0, &intpart);
            } else {
                // Upsampling
                tile_offset_y = std::modf((double)scaleFactorHeight * outputHeightTile0, &intpart) + 1;
            }

            inShapeTile.at(0)[inWidthIndex] = inputWidthTile0;
            inShapeTile.at(1)[inWidthIndex] = inputWidthTile1;
            inShapeTile.at(2)[inWidthIndex] = inputWidthTile0;
            inShapeTile.at(3)[inWidthIndex] = inputWidthTile1;

            inShapeTile.at(0)[inHeightIndex] = inputHeightTile0;
            inShapeTile.at(1)[inHeightIndex] = inputHeightTile1;
            inShapeTile.at(2)[inHeightIndex] = inShape[inHeightIndex] - inputHeightTile0;
            inShapeTile.at(3)[inHeightIndex] = inShape[inHeightIndex] - inputHeightTile1;

            outShapeTile.at(0)[outWidthIndex] = outputWidthTile0;
            outShapeTile.at(1)[outWidthIndex] = outputWidthTile1;
            outShapeTile.at(2)[outWidthIndex] = outputWidthTile0;
            outShapeTile.at(3)[outWidthIndex] = outputWidthTile1;

            outShapeTile.at(0)[outHeightIndex] = outputHeightTile0;
            outShapeTile.at(1)[outHeightIndex] = outputHeightTile1;
            outShapeTile.at(2)[outHeightIndex] = outShape[outHeightIndex] - outputHeightTile0;
            outShapeTile.at(3)[outHeightIndex] = outShape[outHeightIndex] - outputHeightTile1;

            for (auto tile = 0; tile < nrOfInputTiles; tile++) {
                if (totalTensorSize(inShapeTile.at(tile), inputType) > maxM2iHwBlockInputSize) {
                    VPUX_THROW("Input tensor size is bigger than m2i max supported size: {0}",
                               static_cast<uint32_t>(maxM2iHwBlockInputSize));
                }
            }
            const uint32_t offsetFixedPointWidth =
                    VPU::getM2iFixedPointTilingRegister(tile_offset_x, VPU::M2I_TILING_REG_FRACTIONAL_BITS);
            const uint32_t offsetFixedPointHeight =
                    VPU::getM2iFixedPointTilingRegister(tile_offset_y, VPU::M2I_TILING_REG_FRACTIONAL_BITS);

            // offsets vectors = {offsetTile0, offsetTile1, offsetTile2, offsetTile3}
            // For tiles 0 and 2, widths offset are 0 as tiles start at width index 0.
            // For tiles 0 and 1, heights offsets are 0 as tiles start at height index 0.
            // Rest of offsets need to be calculated. Tiling configuration is specified above
            offsetFixedPointsWidth = {0, offsetFixedPointWidth, 0, offsetFixedPointWidth};
            offsetFixedPointsHeight = {0, 0, offsetFixedPointHeight, offsetFixedPointHeight};

            // Input Offsets for tiling
            const Byte inSliceOffset0 =
                    0 * static_cast<Byte>(inputStrides[inHeightDim]) + 0 * static_cast<Byte>(inputStrides[inWidthDim]);

            const Byte inSliceOffset1 = 0 * static_cast<Byte>(inputStrides[inHeightDim]) +
                                        ((inShape[inWidthIndex] / 2)) * static_cast<Byte>(inputStrides[inWidthDim]);

            const Byte inSliceOffset2 = (inShape[inHeightIndex] / 2) * static_cast<Byte>(inputStrides[inHeightDim]) +
                                        0 * static_cast<Byte>(inputStrides[inWidthDim]);

            const Byte inSliceOffset3 = (inShape[inHeightIndex] / 2) * static_cast<Byte>(inputStrides[inHeightDim]) +
                                        (inShape[inWidthIndex] / 2) * static_cast<Byte>(inputStrides[inWidthDim]);

            // Output Offsets for tiling
            const Byte outSliceOffset0 = 0 * static_cast<Byte>(outputStrides[outHeightDim]) +
                                         0 * static_cast<Byte>(outputStrides[outWidthDim]);

            const Byte outSliceOffset1 = 0 * static_cast<Byte>(outputStrides[outHeightDim]) +
                                         (outShape[inWidthIndex] / 2) * static_cast<Byte>(outputStrides[outWidthDim]);

            const Byte outSliceOffset2 =
                    (outShape[inHeightIndex] / 2) * static_cast<Byte>(outputStrides[outHeightDim]) +
                    0 * static_cast<Byte>(outputStrides[outWidthDim]);

            const Byte outSliceOffset3 =
                    (outShape[inHeightIndex] / 2) * static_cast<Byte>(outputStrides[outHeightDim]) +
                    (outShape[inWidthIndex] / 2) * static_cast<Byte>(outputStrides[outWidthDim]);

            inSliceOffsets = {inSliceOffset0, inSliceOffset1, inSliceOffset2, inSliceOffset3};
            outSliceOffsets = {outSliceOffset0, outSliceOffset1, outSliceOffset2, outSliceOffset3};

        } else {
            VPUX_THROW("Tiling not supported for NEAREST interpolation type");
        }
    } else if (inputTotalSize > maxM2iHwBlockInputSize) {
        VPUX_THROW("Input tensor size is bigger than m2i max supported size: {0}",
                   static_cast<uint32_t>(maxM2iHwBlockInputSize));
    }
    int barrierNumber = 0;

    // Input Buffers
    std::vector<vpux::VPURT::DeclareBufferOp> networkInputBuffers;
    std::vector<vpux::VPURT::DeclareBufferOp> inCMXTiles;

    if (params.doTiling) {
        for (auto tile = 0; tile < nrOfInputTiles; tile++) {
            auto CMXTileIdx = tile % availableCMXTiles;
            auto inputBuffer = createDeclareTensorOp(funcBuilder, VPURT::BufferSection::NetworkInput,
                                                     inShapeTile.at(tile), origInputTypeIf.getElementType(),
                                                     origInputTypeIf.getDimsOrder(), inputStrides,
                                                     /*locale=*/0, inSliceOffsets.at(tile).count());
            networkInputBuffers.push_back(inputBuffer);

            auto inCMXtype = getMemRefType(VPURT::BufferSection::CMX_NN, CMXTileIdx, inShapeTile.at(tile), inputType,
                                           DimsOrder::NCHW);
            auto inCMXTile = createDeclareTensorOp(
                    funcBuilder, inCMXtype, VPURT::BufferSection::CMX_NN, CMXTileIdx,
                    getCMXTileOffset(CMXTileOffsets, CMXTileIdx, totalTensorSize(inShapeTile.at(tile), inputType)));
            inCMXTiles.push_back(inCMXTile);
        }
    } else {
        auto inCMXtype = getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShapeTile.at(0), inputType, DimsOrder::NCHW);
        auto inCMXTile = createDeclareTensorOp(
                funcBuilder, inCMXtype, VPURT::BufferSection::CMX_NN, 0,
                getCMXTileOffset(CMXTileOffsets, 0, totalTensorSize(inShapeTile.at(0), inputType)));
        inCMXTiles.push_back(inCMXTile);
    }

    std::vector<vpux::VPURT::DeclareBufferOp> networkOutputBuffers;
    std::vector<vpux::VPURT::DeclareBufferOp> outCMXTiles;

    // Output Buffers
    if (params.doTiling) {
        for (auto tile = 0; tile < nrOfInputTiles; tile++) {
            auto outputBuffer = createDeclareTensorOp(funcBuilder, VPURT::BufferSection::NetworkOutput,
                                                      outShapeTile.at(tile), origOutputTypeIf.getElementType(),
                                                      origOutputTypeIf.getDimsOrder(), outputStrides,
                                                      /*locale=*/0, outSliceOffsets.at(tile).count());
            networkOutputBuffers.push_back(outputBuffer);
        }
    }
    for (auto tile = 0; tile < nrOfInputTiles; tile++) {
        auto CMXTileIdx = tile % availableCMXTiles;
        auto outCMXtype = getMemRefType(VPURT::BufferSection::CMX_NN, CMXTileIdx, outShapeTile.at(tile), outputType,
                                        DimsOrder::NCHW);
        auto outCMXTile = createDeclareTensorOp(
                funcBuilder, outCMXtype, VPURT::BufferSection::CMX_NN, CMXTileIdx,
                getCMXTileOffset(CMXTileOffsets, CMXTileIdx, totalTensorSize(outShapeTile.at(tile), outputType)));
        outCMXTiles.push_back(outCMXTile);
    }

    VPURT::DeclareBufferOp m2iProfOutputCMX;
    VPURT::DeclareBufferOp m2iProfOutputDDR;

    if (profilingParams.profilingEnabled()) {
        if (profilingParams.m2iProfilingEnabled) {
            auto m2iProfOutputCMXType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, profilingOutputShapeUI8,
                                                      getUInt8Type(ctx), DimsOrder::C);
            m2iProfOutputCMX = createDeclareTensorOp(funcBuilder, m2iProfOutputCMXType, VPURT::BufferSection::CMX_NN, 0,
                                                     CMXTileOffsets[0]);
            m2iProfOutputDDR =
                    createDeclareTensorOp(funcBuilder,
                                          getMemRefType(VPURT::BufferSection::ProfilingOutput, profilingOutputShapeUI8,
                                                        getUInt8Type(ctx), DimsOrder::C),
                                          VPURT::BufferSection::ProfilingOutput, 0, 0);
        }
    }

    auto updateBarrier = funcBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

    mlir::IntegerType uint32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);

    // DMA input
    if (params.doTiling) {
        for (auto tile = 0; tile < nrOfInputTiles; tile++) {
            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), builder.getUnknownLoc(),
                                                  networkInputBuffers.at(tile).getOperation()->getResult(0),
                                                  inCMXTiles.at(tile).getOperation()->getResult(0), 0);
        }
    } else {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder,
                                              mlir::ValueRange(),                            // waits
                                              mlir::ValueRange(updateBarrier.getBarrier()),  // updates
                                              builder.getUnknownLoc(),
                                              funcInput0,                                     // src (DDR)
                                              inCMXTiles.at(0).getOperation()->getResult(0),  // dst (CMX)
                                              0);
    }
    auto waitBarrier = updateBarrier;
    updateBarrier = funcBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

    for (auto tile = 0; tile < nrOfInputTiles; tile++) {
        mlir::IntegerAttr widthTilingOffset = nullptr;
        mlir::IntegerAttr heightTilingOffset = nullptr;
        if (params.doTiling) {
            widthTilingOffset =
                    funcBuilder.getIntegerAttr(uint32Type, static_cast<uint32_t>(offsetFixedPointsWidth.at(tile)));
            heightTilingOffset =
                    funcBuilder.getIntegerAttr(uint32Type, static_cast<uint32_t>(offsetFixedPointsHeight.at(tile)));
        }

        auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, printToString("m2i?t_M2I/cluster_{0}", tile)));
        auto m2iTask = VPURT::wrapIntoTaskOp<VPUIP::M2ITaskOp>(
                funcBuilder,                                   // 0) builder
                mlir::ValueRange(waitBarrier.getBarrier()),    // 1) wait-barrier
                mlir::ValueRange(updateBarrier.getBarrier()),  // 2) update-barrier
                loc,                                           // 3) loc
                // next: actual M2ITaskOp args (see 'builder')
                outCMXTiles.at(tile).getType(),
                profilingParams.m2iProfilingEnabled ? m2iProfOutputCMX.getType() : nullptr,
                inCMXTiles.at(tile).getOperation()->getResult(0), outCMXTiles.at(tile).getOperation()->getResult(0),
                profilingParams.m2iProfilingEnabled ? m2iProfOutputCMX.getOperation()->getResult(0) : nullptr,
                params.doCsc, params.doNorm,
                getM2iFmt(params.iFmt),                 // in fmt
                getM2iFmt(params.oFmt),                 // out fmt
                isM2iChromaOrderInverted(params.iFmt),  // chroma order in
                isM2iChromaOrderInverted(params.oFmt),  // chroma order out
                isM2iLumaOrderInverted(params.iFmt),    // luma order in
                isM2iLumaOrderInverted(params.oFmt),    // luma order out
                scaleFactorWidthOrig,                   // scale factor in
                scaleFactorHeightOrig,                  // scale factor out
                normCoefs,                              // norm coefs
                widthTilingOffset,                      // tiling x
                heightTilingOffset,                     // tiling y
                nullptr,                                // profiling metadata
                getM2iInterp(params.interp));

        if (profilingParams.m2iProfilingEnabled) {
            auto profMeta = VPUIP::M2IProfilingMetadataAttr::get(ctx, /*bufferId*/ getIntAttr(ctx, 0),
                                                                 /*bufferOffset*/ getIntAttr(ctx, 0));
            m2iTask.setProfilingMetadataAttr(profMeta);
        }

        waitBarrier = updateBarrier;
        if (params.doTiling) {
            updateBarrier =
                    funcBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
        }
    }

    // DMA output
    auto outputBufferDMA = outCMXTiles.at(0).getOperation()->getResult(0);
    if (params.doTiling) {
        auto networkOutputBuffer = createDeclareTensorOp(funcBuilder, VPURT::BufferSection::NetworkOutput, outShape,
                                                         origOutputTypeIf.getElementType(),
                                                         origOutputTypeIf.getDimsOrder(),  // outputStrides,
                                                         /*locale=*/0, outSliceOffsets.at(0).count());
        for (auto tile = 0; tile < nrOfInputTiles; tile++) {
            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder,
                                                  mlir::ValueRange(waitBarrier.getBarrier()),    // waits
                                                  mlir::ValueRange(updateBarrier.getBarrier()),  // updates
                                                  builder.getUnknownLoc(),
                                                  outCMXTiles.at(tile).getOperation()->getResult(0),  // src (CMX)
                                                  networkOutputBuffers.at(tile).getOperation()->getResult(0), 0);
        }
        waitBarrier = updateBarrier;
        outputBufferDMA = networkOutputBuffer.getOperation()->getResult(0);
    }

    mlir::SmallVector<mlir::Value> funcOutputs;

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    // copy output from CMX to DDR
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder,
                                          mlir::ValueRange(waitBarrier.getBarrier()),   // waits
                                          mlir::ValueRange(finalBarrier.getBarrier()),  // updates
                                          builder.getUnknownLoc(),
                                          outputBufferDMA,  // src (CMX)
                                          funcOutput, 0);
    funcOutputs.push_back(funcOutput);

    // copy profiling data from CMX to DDR
    if (profilingParams.profilingEnabled()) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder,
                                              mlir::ValueRange(waitBarrier.getBarrier()),   // waits
                                              mlir::ValueRange(finalBarrier.getBarrier()),  // updates
                                              builder.getUnknownLoc(),
                                              m2iProfOutputCMX,  // src (CMX)
                                              m2iProfOutputDDR, 0);
        funcOutputs.push_back(funcProfOutput);
    }

    funcBuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcOutputs);

    // set runtime resources
    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto dpuTiles = params.doTiling ? VPU::getMaxArchDPUClusterNum(testDesc.getArchitecture()) : 1;
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = dpuTiles;
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    mlir::SmallVector<ProfilingDataSection> profilingDataSections;
    size_t offset = 0;
    if (profilingParams.m2iProfilingEnabled) {
        profilingDataSections.push_back({HWP_M2I_SECTION_EXEC_TYPE, offset, m2iProfilingBufferSizeBytes});
        offset += m2iProfilingBufferSizeBytes;
    }
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NCHW, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NCHW, nullptr)}, profilingDataSections);
}

}  // namespace hwtest
}  // namespace vpux

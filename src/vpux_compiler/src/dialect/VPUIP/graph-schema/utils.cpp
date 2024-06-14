//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"

using namespace vpux;

// Serialization utils

const EnumMap<ov::element::Type_t, MVCNN::OVNodeType> VPUIP::mapElementType = {
        {ov::element::Type_t::undefined, MVCNN::OVNodeType::OVNodeType_UNDEFINED},
        {ov::element::Type_t::dynamic, MVCNN::OVNodeType::OVNodeType_DYNAMIC},
        {ov::element::Type_t::boolean, MVCNN::OVNodeType::OVNodeType_BOOLEAN},
        {ov::element::Type_t::bf16, MVCNN::OVNodeType::OVNodeType_BF16},
        {ov::element::Type_t::f16, MVCNN::OVNodeType::OVNodeType_F16},
        {ov::element::Type_t::f32, MVCNN::OVNodeType::OVNodeType_F32},
        {ov::element::Type_t::f64, MVCNN::OVNodeType::OVNodeType_F64},
        {ov::element::Type_t::i4, MVCNN::OVNodeType::OVNodeType_I4},
        {ov::element::Type_t::i8, MVCNN::OVNodeType::OVNodeType_I8},
        {ov::element::Type_t::i16, MVCNN::OVNodeType::OVNodeType_I16},
        {ov::element::Type_t::i32, MVCNN::OVNodeType::OVNodeType_I32},
        {ov::element::Type_t::i64, MVCNN::OVNodeType::OVNodeType_I64},
        {ov::element::Type_t::u1, MVCNN::OVNodeType::OVNodeType_U1},
        {ov::element::Type_t::u4, MVCNN::OVNodeType::OVNodeType_U4},
        {ov::element::Type_t::u8, MVCNN::OVNodeType::OVNodeType_U8},
        {ov::element::Type_t::u16, MVCNN::OVNodeType::OVNodeType_U16},
        {ov::element::Type_t::u32, MVCNN::OVNodeType::OVNodeType_U32},
        {ov::element::Type_t::u64, MVCNN::OVNodeType::OVNodeType_U64},
};

MVCNN::TargetDevice VPUIP::mapTargetDevice(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::NPU30XX:
        return MVCNN::TargetDevice::TargetDevice_VPUX30XX;
    case VPU::ArchKind::NPU37XX:
        return MVCNN::TargetDevice::TargetDevice_VPUX37XX;
    case VPU::ArchKind::NPU40XX:
        return MVCNN::TargetDevice::TargetDevice_VPUX40XX;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

MVCNN::TargetDeviceRevision VPUIP::mapTargetDeviceRevision(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::NPU30XX:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0;
    default:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_NONE;
    }
}

MVCNN::PerfDataMode VPUIP::mapProfilingMode(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::NPU30XX:
    case VPU::ArchKind::NPU37XX:
        return MVCNN::PerfDataMode_MODE0;
    case VPU::ArchKind::NPU40XX:
        return MVCNN::PerfDataMode_MODE3;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

MVCNN::DType VPUIP::createDType(mlir::Type type) {
    if (type.isF64()) {
        return MVCNN::DType_FP64;
    } else if (type.isF32()) {
        return MVCNN::DType_FP32;
    } else if (type.isF16()) {
        return MVCNN::DType_FP16;
    } else if (type.isBF16()) {
        return MVCNN::DType_BFP16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return MVCNN::DType_I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return MVCNN::DType_I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return MVCNN::DType_I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return MVCNN::DType_I8;
    } else if (type.isSignedInteger(4)) {
        return MVCNN::DType_I4;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint64_t))) {
        return MVCNN::DType_U64;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint32_t))) {
        return MVCNN::DType_U32;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint16_t))) {
        return MVCNN::DType_U16;
    } else if (type.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return MVCNN::DType_U8;
    } else if (type.isInteger(4)) {
        return MVCNN::DType_U4;
    } else if (type.isInteger(2)) {
        return MVCNN::DType_I2;
    } else if (type.isInteger(1)) {
        return MVCNN::DType_BIN;
    } else if (type.isa<mlir::quant::QuantizedType>()) {
        return createDType(type.cast<mlir::quant::QuantizedType>().getStorageType());
    } else {
        VPUX_THROW("Unsupported element type {0}", type);
    }
}

MVCNN::MemoryLocation VPUIP::createMemoryLocation(VPURT::BufferSection section) {
    switch (section) {
    case VPURT::BufferSection::NetworkInput:
        return MVCNN::MemoryLocation_ProgrammableInput;
    case VPURT::BufferSection::NetworkOutput:
        return MVCNN::MemoryLocation_ProgrammableOutput;
    case VPURT::BufferSection::ProfilingOutput:
        return MVCNN::MemoryLocation_ProfilingOutput;
    case VPURT::BufferSection::Constant:
        return MVCNN::MemoryLocation_GraphFile;
    case VPURT::BufferSection::SW_KernelText:
        return MVCNN::MemoryLocation_GFEmbeddedKernel;
    case VPURT::BufferSection::DDR:
        return MVCNN::MemoryLocation_VPU_DDR_Heap;
    case VPURT::BufferSection::CSRAM:
        return MVCNN::MemoryLocation_VPU_CSRAM;
    case VPURT::BufferSection::CMX_UPA:
        return MVCNN::MemoryLocation_VPU_CMX_UPA;
    case VPURT::BufferSection::CMX_NN:
        return MVCNN::MemoryLocation_VPU_CMX_NN;
    case VPURT::BufferSection::Register:
        return MVCNN::MemoryLocation_AbsoluteAddr;
    case VPURT::BufferSection::MAC_Accumulators:
        return MVCNN::MemoryLocation_MAC_Accumulators;
    default:
        VPUX_THROW("Unsupported BufferSection {0}", section);
    }
}

MVCNN::order3 VPUIP::createOrder3(mlir::ArrayAttr attr) {
    auto vec = parseIntArrayAttr<int64_t>(attr);
    std::reverse(vec.begin(), vec.end());

    VPUX_THROW_UNLESS(vec.size() <= 3, "Got wrong order array : {0}", vec);

    uint8_t x = 0, y = 0, z = 0;
    if (vec.size() >= 1) {
        x = checked_cast<uint8_t>(vec[0]);
    }
    if (vec.size() >= 2) {
        y = checked_cast<uint8_t>(vec[1]);
    }
    if (vec.size() >= 3) {
        z = checked_cast<uint8_t>(vec[2]);
    }

    return MVCNN::order3(x, y, z);
}

MVCNN::DepthToSpaceMode VPUIP::convertVPUXDepthToSpaceMode2MVCNN(IE::DepthToSpaceMode mode) {
    MVCNN::DepthToSpaceMode out_code = MVCNN::DepthToSpaceMode_BLOCKS_FIRST;
    switch (mode) {
    case IE::DepthToSpaceMode::BLOCKS_FIRST:
        out_code = MVCNN::DepthToSpaceMode_BLOCKS_FIRST;
        break;
    case IE::DepthToSpaceMode::DEPTH_FIRST:
        out_code = MVCNN::DepthToSpaceMode_DEPTH_FIRST;
        break;
    default:
        VPUX_THROW("Unknown DepthToSpaceMode. Blocks_FIRST and DEPTH_FIRST methods are supported only");
    }
    return out_code;
}

MVCNN::ROIAlignMethod VPUIP::convertVPUXROIAlignMethod2MVCNN(IE::ROIAlignMethod method) {
    MVCNN::ROIAlignMethod mvcnn_method;
    switch (method) {
    case IE::ROIAlignMethod::AVG:
        mvcnn_method = MVCNN::ROIAlignMethod_roi_align_avg;
        break;
    case IE::ROIAlignMethod::MAX:
        mvcnn_method = MVCNN::ROIAlignMethod_roi_align_max;
        break;
    default:
        VPUX_THROW("Unknown ROIAlignMethod. avg and max methods are supported only");
    }
    return mvcnn_method;
}

MVCNN::SpaceToDepthMode VPUIP::convertVPUXSpaceToDepthMode2MVCNN(IE::SpaceToDepthMode vpux_mode) {
    MVCNN::SpaceToDepthMode mvcnn_mode;
    switch (vpux_mode) {
    case IE::SpaceToDepthMode::BLOCKS_FIRST:
        mvcnn_mode = MVCNN::SpaceToDepthMode::SpaceToDepthMode_BLOCKS_FIRST;
        break;
    case IE::SpaceToDepthMode::DEPTH_FIRST:
        mvcnn_mode = MVCNN::SpaceToDepthMode::SpaceToDepthMode_DEPTH_FIRST;
        break;
    default:
        VPUX_THROW("Unsupported SpaceToDepthMode {0}", vpux_mode);
    }
    return mvcnn_mode;
}

MVCNN::PadMode VPUIP::convertVPUXPadMode2MVCNN(IE::PadMode vpux_mode) {
    MVCNN::PadMode mvcnn_mode;
    switch (vpux_mode) {
    case IE::PadMode::EDGE:
        mvcnn_mode = MVCNN::PadMode::PadMode_Edge;
        break;
    case IE::PadMode::REFLECT:
        mvcnn_mode = MVCNN::PadMode::PadMode_Reflect;
        break;
    case IE::PadMode::CONSTANT:
        mvcnn_mode = MVCNN::PadMode::PadMode_Constant;
        break;
    case IE::PadMode::SYMMETRIC:
        mvcnn_mode = MVCNN::PadMode::PadMode_Symmetric;
        break;
    default:
        VPUX_THROW("Unsupported PadMode {0}", vpux_mode);
    }
    return mvcnn_mode;
}

MVCNN::RoundMode VPUIP::convertVPUXRoundMode2MVCNN(IE::RoundMode vpux_mode) {
    MVCNN::RoundMode mvcnn_mode;
    switch (vpux_mode) {
    case IE::RoundMode::HALF_TO_EVEN:
        mvcnn_mode = MVCNN::RoundMode::RoundMode_HALF_TO_EVEN;
        break;
    case IE::RoundMode::HALF_AWAY_FROM_ZERO:
        mvcnn_mode = MVCNN::RoundMode::RoundMode_HALF_AWAY_FROM_ZERO;
        break;
    default:
        VPUX_THROW("Unsupported RoundMode {0}", vpux_mode);
    }
    return mvcnn_mode;
}

MVCNN::PSROIPoolingMode VPUIP::convertVPUXPSROIPoolingModeToMVNCNN(IE::PSROIPoolingMode mode) {
    switch (mode) {
    case IE::PSROIPoolingMode::AVERAGE:
        return MVCNN::PSROIPoolingMode::PSROIPoolingMode_AVERAGE;
    case IE::PSROIPoolingMode::BILINEAR:
        return MVCNN::PSROIPoolingMode::PSROIPoolingMode_BILINEAR;
    default:
        VPUX_THROW("Unknown PSROIPoolingMode. Got {0} mode", mode);
    }
}

MVCNN::M2IFormat VPUIP::convertM2iColor2MVCNN(VPU::M2iColorFmt fmt) {
    switch (fmt) {
    case VPU::M2iColorFmt::PL_YUV444_8:
        return MVCNN::M2IFormat::M2IFormat_PL_YUV444_8;
    case VPU::M2iColorFmt::PL_YUV420_8:
        return MVCNN::M2IFormat::M2IFormat_PL_YUV420_8;
    case VPU::M2iColorFmt::PL_RGB24:
        return MVCNN::M2IFormat::M2IFormat_PL_RGB24;
    case VPU::M2iColorFmt::PL_RGB30:
        return MVCNN::M2IFormat::M2IFormat_PL_RGB30;
    case VPU::M2iColorFmt::PL_GRAY8:
        return MVCNN::M2IFormat::M2IFormat_PL_GRAY8;
    case VPU::M2iColorFmt::PL_FP16_RGB:
        return MVCNN::M2IFormat::M2IFormat_PL_FP16_RGB;
    case VPU::M2iColorFmt::PL_FP16_YUV:
        return MVCNN::M2IFormat::M2IFormat_PL_FP16_YUV;
    case VPU::M2iColorFmt::PL_YUV422_8:
        return MVCNN::M2IFormat::M2IFormat_PL_YUV422_8;
    case VPU::M2iColorFmt::SP_NV12_8:
        return MVCNN::M2IFormat::M2IFormat_SP_NV12_8;
    case VPU::M2iColorFmt::SP_NV12_10:
        return MVCNN::M2IFormat::M2IFormat_SP_NV12_10;
    case VPU::M2iColorFmt::SP_P010:
        return MVCNN::M2IFormat::M2IFormat_SP_P010;
    case VPU::M2iColorFmt::IL_YUV422_8:
        return MVCNN::M2IFormat::M2IFormat_IL_YUV422_8;
    case VPU::M2iColorFmt::IL_RGB8888:
        return MVCNN::M2IFormat::M2IFormat_IL_RGB8888;
    case VPU::M2iColorFmt::IL_RGB888:
        return MVCNN::M2IFormat::M2IFormat_IL_RGB888;
    case VPU::M2iColorFmt::IL_RGB30:
        return MVCNN::M2IFormat::M2IFormat_IL_RGB30;
    default:
        VPUX_THROW("Unsupported M2I {0} format", fmt);
    }
}

MVCNN::M2IInterp VPUIP::convertM2iInterp2MVCNN(VPU::M2iInterp interp) {
    switch (interp) {
    case VPU::M2iInterp::NEAREST:
        return MVCNN::M2IInterp::M2IInterp_NEAREST;
    case VPU::M2iInterp::BILINEAR:
        return MVCNN::M2IInterp::M2IInterp_BILINEAR;
    default:
        VPUX_THROW("Unsupported M2I {0} interpolation type", interp);
    }
}

MVCNN::DeformablePSROIPoolingMode VPUIP::convertVPUXDeformablePSROIPoolingModeToMVNCNN(
        IE::DeformablePSROIPoolingMode mode) {
    switch (mode) {
    case IE::DeformablePSROIPoolingMode::AVERAGE:
        return MVCNN::DeformablePSROIPoolingMode::DeformablePSROIPoolingMode_AVERAGE;
    case IE::DeformablePSROIPoolingMode::BILINEAR_DEFORMABLE:
        return MVCNN::DeformablePSROIPoolingMode::DeformablePSROIPoolingMode_BILINEAR_DEFORMABLE;
    default:
        VPUX_THROW("Unknown DeformablePSROIPoolingMode. Got {0} mode", mode);
    }
}

// This method converts value from ROIPoolingMethod view to corresponds t_ROIPooling_method view from runtime
uint32_t VPUIP::convertVPUXROIPoolingMethod2Int32(IE::ROIPoolingMethod method) {
    uint32_t out_code = 0;
    switch (method) {
    case IE::ROIPoolingMethod::MAX:
        out_code = 0;
        break;
    case IE::ROIPoolingMethod::BILINEAR:
        out_code = 1;
        break;
    default:
        VPUX_THROW("Unknown ROIPoolingMethod. max and bilinear methods are supported only");
    }
    return out_code;
}

const EnumMap<IE::InterpolateMode, MVCNN::InterpolationMethod> VPUIP::supportedInterpModeMap = {
        {IE::InterpolateMode::NEAREST, MVCNN::InterpolationMethod_NEAREST},         //
        {IE::InterpolateMode::LINEAR, MVCNN::InterpolationMethod_BILINEAR},         //
        {IE::InterpolateMode::LINEAR_ONNX, MVCNN::InterpolationMethod_LINEARONNX},  //
};

const EnumMap<IE::InterpolateNearestMode, MVCNN::InterpolationNearestMode> VPUIP::nearestModeMap = {
        {IE::InterpolateNearestMode::ROUND_PREFER_FLOOR, MVCNN::InterpolationNearestMode_ROUND_PREFER_FLOOR},  //
        {IE::InterpolateNearestMode::ROUND_PREFER_CEIL, MVCNN::InterpolationNearestMode_ROUND_PREFER_CEIL},    //
        {IE::InterpolateNearestMode::FLOOR, MVCNN::InterpolationNearestMode_FLOOR},                            //
        {IE::InterpolateNearestMode::CEIL, MVCNN::InterpolationNearestMode_CEIL},                              //
        {IE::InterpolateNearestMode::SIMPLE, MVCNN::InterpolationNearestMode_SIMPLE},                          //
};

const EnumMap<IE::InterpolateCoordMode, MVCNN::InterpolationCoordTransMode> VPUIP::coordTransformModeMap = {
        {IE::InterpolateCoordMode::HALF_PIXEL, MVCNN::InterpolationCoordTransMode_HALF_PIXEL},                      //
        {IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL, MVCNN::InterpolationCoordTransMode_PYTORCH_HALF_PIXEL},      //
        {IE::InterpolateCoordMode::ASYMMETRIC, MVCNN::InterpolationCoordTransMode_ASYMMETRIC},                      //
        {IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN, MVCNN::InterpolationCoordTransMode_TF_HALF_PIXEL_FOR_NN},  //
        {IE::InterpolateCoordMode::ALIGN_CORNERS, MVCNN::InterpolationCoordTransMode_ALIGN_CORNERS},                //
};

MVCNN::PhysicalProcessor VPUIP::createPhysicalProcessor(VPU::ExecutorKind execKind) {
    switch (execKind) {
    case VPU::ExecutorKind::SHAVE_UPA:
        return MVCNN::PhysicalProcessor_UPA_SHV;
    case VPU::ExecutorKind::SHAVE_NN:
        return MVCNN::PhysicalProcessor_NN_SHV;
    case VPU::ExecutorKind::NCE:
        return MVCNN::PhysicalProcessor_NCE_Cluster;
    case VPU::ExecutorKind::DPU:
        return MVCNN::PhysicalProcessor_NCE_PerClusterDPU;
    default:
        VPUX_THROW("Unsupported ExecutorKind '{0}'", execKind);
    }
}

flatbuffers::Offset<MVCNN::ProcessorMapping> VPUIP::createProcessorMapping(flatbuffers::FlatBufferBuilder& fbb,
                                                                           IERT::ComputeResourceOpInterface res,
                                                                           mlir::ModuleOp module) {
    const auto execKind =
            VPU::getKindValue<VPU::ExecutorKind>(mlir::cast<IERT::ResourceOpInterface>(res.getOperation()));
    MVCNN::ProcessorMappingBuilder builder(fbb);
    builder.add_item(createPhysicalProcessor(execKind));
    builder.add_number(checked_cast<double>(res.getCountAttr().getValue().getSExtValue()));
    builder.add_is_bitmask(false);
    builder.add_activity_factor(VPU::getActivityFactor(execKind, module, res));
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::ProcessorMapping> VPUIP::createProcessorFreqMapping(flatbuffers::FlatBufferBuilder& fbb,
                                                                               IE::TileResourceOp res) {
    const auto execKind = VPU::getKindValue<VPU::ExecutorKind>(res);

    MVCNN::ProcessorMappingBuilder builder(fbb);
    builder.add_item(createPhysicalProcessor(execKind));
    builder.add_number(res.getProcessorFrequency().getValueAsDouble());
    builder.add_is_bitmask(false);
    return builder.Finish();
}

MVCNN::PhysicalMem VPUIP::createPhysicalMem(VPU::MemoryKind mem) {
    switch (mem) {
    case VPU::MemoryKind::DDR:
        return MVCNN::PhysicalMem_DDR;
    case VPU::MemoryKind::CSRAM:
        return MVCNN::PhysicalMem_CSRAM;
    case VPU::MemoryKind::CMX_UPA:
        return MVCNN::PhysicalMem_UPA_CMX;
    case VPU::MemoryKind::CMX_NN:
        return MVCNN::PhysicalMem_NN_CMX;
    default:
        VPUX_THROW("Unsupported MemoryKind '{0}'", mem);
    }
}

flatbuffers::Offset<MVCNN::MemoryRelationshipMapping> VPUIP::createBandwidthMapping(flatbuffers::FlatBufferBuilder& fbb,
                                                                                    IE::MemoryResourceOp src,
                                                                                    IE::MemoryResourceOp dst,
                                                                                    double bandwidth) {
    MVCNN::MemoryRelationshipMappingBuilder builder(fbb);
    const auto srcKind = VPU::getKindValue<VPU::MemoryKind>(src);
    const auto dstKind = VPU::getKindValue<VPU::MemoryKind>(dst);

    builder.add_from_item(createPhysicalMem(srcKind));
    builder.add_to_item(createPhysicalMem(dstKind));
    builder.add_number(bandwidth);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::MemoryMapping> VPUIP::createMemoryMapping(flatbuffers::FlatBufferBuilder& fbb,
                                                                     IE::MemoryResourceOp res) {
    const auto memKind = VPU::getKindValue<VPU::MemoryKind>(res);

    MVCNN::MemoryMappingBuilder builder(fbb);
    builder.add_item(createPhysicalMem(memKind));
    builder.add_number(checked_cast<double>(res.getByteSize()));
    return builder.Finish();
}

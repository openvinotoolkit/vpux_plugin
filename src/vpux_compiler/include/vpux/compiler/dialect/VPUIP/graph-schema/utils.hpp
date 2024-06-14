//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/enums.hpp"

#include <openvino/core/type/element_type.hpp>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

//
// Serialize utils
//

MVCNN::TargetDevice mapTargetDevice(VPU::ArchKind kind);
MVCNN::TargetDeviceRevision mapTargetDeviceRevision(VPU::ArchKind kind);
MVCNN::PerfDataMode mapProfilingMode(VPU::ArchKind kind);

MVCNN::DType createDType(mlir::Type type);
MVCNN::MemoryLocation createMemoryLocation(VPURT::BufferSection section);
MVCNN::order3 createOrder3(mlir::ArrayAttr attr);

extern const EnumMap<ov::element::Type_t, MVCNN::OVNodeType> mapElementType;

MVCNN::DepthToSpaceMode convertVPUXDepthToSpaceMode2MVCNN(IE::DepthToSpaceMode mode);
MVCNN::ROIAlignMethod convertVPUXROIAlignMethod2MVCNN(IE::ROIAlignMethod method);
MVCNN::SpaceToDepthMode convertVPUXSpaceToDepthMode2MVCNN(IE::SpaceToDepthMode vpux_mode);
MVCNN::PadMode convertVPUXPadMode2MVCNN(IE::PadMode vpux_mode);
MVCNN::RoundMode convertVPUXRoundMode2MVCNN(IE::RoundMode vpux_mode);
MVCNN::PSROIPoolingMode convertVPUXPSROIPoolingModeToMVNCNN(IE::PSROIPoolingMode mode);
MVCNN::M2IFormat convertM2iColor2MVCNN(VPU::M2iColorFmt fmt);
MVCNN::M2IInterp convertM2iInterp2MVCNN(VPU::M2iInterp interp);
MVCNN::DeformablePSROIPoolingMode convertVPUXDeformablePSROIPoolingModeToMVNCNN(IE::DeformablePSROIPoolingMode mode);
uint32_t convertVPUXROIPoolingMethod2Int32(IE::ROIPoolingMethod method);

extern const EnumMap<IE::InterpolateMode, MVCNN::InterpolationMethod> supportedInterpModeMap;
extern const EnumMap<IE::InterpolateNearestMode, MVCNN::InterpolationNearestMode> nearestModeMap;
extern const EnumMap<IE::InterpolateCoordMode, MVCNN::InterpolationCoordTransMode> coordTransformModeMap;

MVCNN::PhysicalProcessor createPhysicalProcessor(VPU::ExecutorKind execKind);
MVCNN::PhysicalMem createPhysicalMem(VPU::MemoryKind mem);
flatbuffers::Offset<MVCNN::ProcessorMapping> createProcessorMapping(flatbuffers::FlatBufferBuilder& fbb,
                                                                    IERT::ComputeResourceOpInterface res,
                                                                    mlir::ModuleOp module);
flatbuffers::Offset<MVCNN::ProcessorMapping> createProcessorFreqMapping(flatbuffers::FlatBufferBuilder& fbb,
                                                                        IE::TileResourceOp res);
flatbuffers::Offset<MVCNN::MemoryRelationshipMapping> createBandwidthMapping(flatbuffers::FlatBufferBuilder& fbb,
                                                                             IE::MemoryResourceOp src,
                                                                             IE::MemoryResourceOp dst,
                                                                             double bandwidth);
flatbuffers::Offset<MVCNN::MemoryMapping> createMemoryMapping(flatbuffers::FlatBufferBuilder& fbb,
                                                              IE::MemoryResourceOp res);

}  // namespace VPUIP
}  // namespace vpux

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strided_shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attr_interfaces.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/Support/FormatVariadic.h>

namespace vpux {
namespace VPU {

class PaddingAttr;

}  // namespace VPU
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/IE/attributes.hpp.inc>
#include <vpux/compiler/dialect/VPU/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPU/attributes.hpp.inc>

namespace vpux {
namespace VPU {

// This one represents a CMX_NN memory space with fragmentation consideration
constexpr StringLiteral CMX_NN_FragmentationAware = "CMX_NN_FragmentationAware";

//
// Run-time resources
//

StringLiteral getMemoryDerateAttrName();
StringLiteral getMemoryBandwidthAttrName();

/**
 * @brief Get DPU frequency
 *
 * @param arch - architecture
 * @return DPU clock frequency [MHz]
 *
 * @note Provides processor frequency values (ie. dpu_clk) that get exported to values under
 * header>resources>processor_frequencies>number in the blob.
 *
 * @note Note the difference between the vpu and dpu clock frequencies.
 *
 * @note Values returned by this function are tight to definitions provided by
 * vpucostmodel.
 */
unsigned int getDpuFrequency(vpux::VPU::ArchKind arch);

/**
 * @brief Get maximal DMA bandwidth for a given architecture
 *
 * @param module
 * @return bandwidth in GB/s
 *
 * The BW value depends on platform, number of DMA channels and DPU clock frequency.
 * The function uses vpuperformance specifications and typically
 * corresponds to the maximal DPU frequencies (and dual DMA if available) for given architecture.
 *
 * The value is serialized into blob header fields (header>resources>memory_bandwidth>number).
 */
double getDmaBandwidthGBps(mlir::ModuleOp module);

/**
 * @brief Get maximal DMA bandwidth for a given architecture
 *
 * @param arch - architectire
 *
 * See getDmaBandwidthGBps(mlir::ModuleOp module)
 */
double getDmaBandwidthGBps(ArchKind arch);

uint32_t getMaxArchDPUClusterNum(ArchKind arch);
uint32_t getMaxArchDPUClusterNum(mlir::Operation* op);
uint32_t getMaxDMAPorts(ArchKind arch);

/**
 * @brief return DMA bandwidth
 *
 * @param arch
 * @return DMA bandwidth in bytes per DPU clock cycle
 */
double getDMABandwidth(ArchKind arch);

/**
 * @brief NCE troughput
 *
 * @param arch
 * @return return NCE troughtput in MOPS (millions of operations per second)
 */
double getNCEThroughput(ArchKind arch);

Byte getTotalCMXSize(mlir::Operation* op);
Byte getTotalCMXSize(mlir::ModuleOp module);
Byte getTotalCMXFragmentationAwareSize(mlir::Operation* op);
Byte getTotalCMXFragmentationAwareSize(mlir::ModuleOp module);
Byte getTotalCMXVFPipelineFragmentationAwareSize(mlir::Operation* op);

//
// ArchKind
//

void setArch(mlir::ModuleOp module, ArchKind kind, std::optional<int> numOfDPUGroups = std::nullopt,
             std::optional<int> numOfDMAPorts = std::nullopt,
             std::optional<vpux::Byte> availableCMXMemory = std::nullopt, bool allowCustomValues = false);

ArchKind getArch(mlir::Operation* op);
bool isArchVPUX3XXX(VPU::ArchKind arch);

//
// CompilationMode
//

void setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode);
bool hasCompilationMode(mlir::ModuleOp module);
CompilationMode getCompilationMode(mlir::Operation* op);

//
// RevisionID
//

void setRevisionID(mlir::ModuleOp module, RevisionID revisionID);
bool hasRevisionID(mlir::ModuleOp module);
std::optional<RevisionID> getRevisionID(mlir::Operation* op);

//
// PaddingAttr
//

PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, int64_t left, int64_t right, int64_t top, int64_t bottom);
PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd);
PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, const PadInfo& pad);

PadInfo toPadInfo(PaddingAttr attr);

//
// PPETaskAttr
//

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode);

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow, int64_t clampHigh,
                           int64_t lreluMult, int64_t lreluShift);

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow, int64_t clampHigh,
                           int64_t lreluMult, int64_t lreluShift, ArrayRef<int64_t> quantMult,
                           ArrayRef<int64_t> quantShift, int64_t quantPostShift);

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow, int64_t clampHigh,
                           int64_t lreluMult, int64_t lreluShift, ArrayRef<double> quantScale);

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow, int64_t clampHigh,
                           int64_t lreluMult, int64_t lreluShift, ArrayRef<int64_t> quantMult,
                           ArrayRef<int64_t> quantShift, int64_t quantPostShift, ArrayRef<int64_t> in1QuantMult,
                           ArrayRef<int64_t> in2QuantMult, double fpPReluAlpha = 1.0);

VPU::PPEMode getPPEMode(VPU::EltwiseType type);

//
// DistributedTensorAttr
//

struct OverlapDistributionParams {
    OverlapDistributionParams() = default;
    OverlapDistributionParams(const OverlapDistributionParams&) = default;

    OverlapDistributionParams& operator=(const OverlapDistributionParams&) = default;

    ~OverlapDistributionParams() = default;

    OverlapDistributionParams(mlir::ArrayAttr kernel, VPU::PaddingAttr pads, mlir::ArrayAttr stride,
                              mlir::UnitAttr equalComputeAndMemoryView = nullptr)
            : kernel(kernel), pads(pads), stride(stride), equalComputeAndMemoryView(equalComputeAndMemoryView){};

    OverlapDistributionParams(mlir::ArrayAttr memoryShapes, mlir::ArrayAttr memoryOffsets,
                              mlir::ArrayAttr computeShapes, mlir::ArrayAttr computeOffsets)
            : memoryShapes(memoryShapes),
              memoryOffsets(memoryOffsets),
              computeShapes(computeShapes),
              computeOffsets(computeOffsets){};

    bool hasNonnullComputeAndMemoryShapesOffsets() const {
        return (memoryShapes != nullptr) && (memoryOffsets != nullptr) && (computeShapes != nullptr) &&
               (computeOffsets != nullptr);
    }

    mlir::ArrayAttr kernel = nullptr;
    VPU::PaddingAttr pads = nullptr;
    mlir::ArrayAttr stride = nullptr;
    mlir::UnitAttr equalComputeAndMemoryView = nullptr;
    mlir::ArrayAttr memoryShapes = nullptr;
    mlir::ArrayAttr memoryOffsets = nullptr;
    mlir::ArrayAttr computeShapes = nullptr;
    mlir::ArrayAttr computeOffsets = nullptr;
};

mlir::LogicalResult verify(FuncRef<mlir::InFlightDiagnostic()> emitError, DistributedTensorAttr distributedAttr,
                           ArrayRef<int64_t> shape);
mlir::LogicalResult canTheDistributionModesBeCompatible(DistributionMode sourceMode, DistributionMode targetMode);
mlir::LogicalResult areDistributionNumClustersCompatible(mlir::IntegerAttr sourceNumClusters,
                                                         mlir::IntegerAttr targetNumClusters);
mlir::LogicalResult areDistributionElementTypesCompatible(mlir::Type inType, mlir::Type outType);
SmallVector<Shape> getPerClusterComputeShapes(ShapeRef shapeRef, DistributedTensorAttr distributionAttr);
SmallVector<Shape> getPerClusterComputeShapeOffsets(ShapeRef shapeRef, DistributedTensorAttr distributionAttr);
std::optional<SmallVector<Shape>> getPerClusterMemoryShapes(ShapeRef shapeRef, DistributedTensorAttr distributionAttr);
SmallVector<Shape> getPerClusterMemoryShapeOffsets(ShapeRef shapeRef, DistributedTensorAttr distributionAttr);
SmallVector<PadInfo> getPerClusterPadding(DistributedTensorAttr distributionAttr, PadInfo kernelPadding);
SmallVector<StridedShape> getPerClusterMemoryStridedShapes(ShapeRef shape, StridesRef strides, DimsOrder dimsOrder,
                                                           DistributionModeAttr mode, ArrayRef<Shape> memoryShapes);
SmallVector<Shape> getOverlappedPerClusterNewMemoryShapes(ShapeRef newShape, ShapeRef origShape,
                                                          DistributedTensorAttr distributionAttr);
SmallVector<Shape> getOverlappedPerClusterNewMemoryShapeOffsets(ShapeRef shapeRef,
                                                                DistributedTensorAttr distributionAttr);
int64_t getDistributedTilingAxis(ArrayRef<int64_t> tilingScheme);
bool isDistributedAttrWithExplicitShapesAndOffsets(DistributedTensorAttr distributionAttr);
bool isUniformDistributedSegmentsSupported(mlir::Operation* op);
SmallVector<Shape> arrayAttrToVecOfShapes(mlir::ArrayAttr arr);

bool isSegmentedOverH(VPU::DistributedTensorAttr distAttr);
bool isSegmentedOverC(VPU::DistributedTensorAttr distAttr);
bool isSegmentedDuplicatedOverC(VPU::DistributedTensorAttr distAttr);
bool isSegmentedOverN(VPU::DistributedTensorAttr distAttr);
bool isOverlappedOverH(VPU::DistributedTensorAttr distAttr);

//
// SparsityCompressionAttr
//

VPU::SparsityCompressionAttr getSparsityCompressionAttr(mlir::Type type);
mlir::Type setSparsityCompressionAttr(mlir::Type type, VPU::SparsityCompressionAttr sparsityCompressionAttr);

VPU::SparsityCompressionAttr tileSparsityCompression(VPU::SparsityCompressionAttr sparsityCompression,
                                                     ShapeRef tileOffsets, ShapeRef tileShape);

//
// Resource kind value getter
//

template <typename ConcreteKind, typename ResourceOp>
ConcreteKind getKindValue(ResourceOp op) {
    VPUX_THROW_WHEN(!op.getKind(), "Can't find attributes for Operation");
    const auto maybeKind = vpux::VPU::symbolizeEnum<ConcreteKind>(op.getKind());
    VPUX_THROW_WHEN(!maybeKind.has_value(), "Unsupported attribute kind");
    return maybeKind.value();
}

//
// Common utilities
//

template <VPU::MemoryKind KIND>
std::optional<VPU::MemoryKind> getMemKind(StringRef) {
    return KIND;
}

std::optional<SmallVector<Shape>> splitSegmentedShape(ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme,
                                                      const int64_t numClusters, const int64_t axis,
                                                      std::optional<ArrayRef<int64_t>> alignment,
                                                      bool uniformDistributedSegments = false);

// This is a temporary enum containing deprecated values of VPU::ArchKind, which are used only as a bridge for old
// parameters of lit-tests.
enum class DeprecatedArchKind : uint64_t {
    UNKNOWN = 0,
    VPUX30XX = 1,
    VPUX37XX = 3,
    VPUX40XX = 4,
};

std::optional<DeprecatedArchKind> symbolizeDeprecatedArchKind(StringRef);
ArchKind mapDeprecatedArchKind(DeprecatedArchKind);

}  // namespace VPU
}  // namespace vpux

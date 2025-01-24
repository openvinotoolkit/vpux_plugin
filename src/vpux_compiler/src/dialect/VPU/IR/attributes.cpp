//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include "vpu/performance.h"

using namespace vpux;

//
// Dialect hooks
//

void VPU::VPUDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPU/attributes.cpp.inc>
            >();
}

//
// Run-time resources
//

namespace {

constexpr StringLiteral derateFactorAttrName = "VPU.derateFactor";
constexpr StringLiteral bandwidthAttrName = "VPU.bandwidth"; /*!< This attribute corresponds to a single JSON field
                      nested at header>resources>memory_bandwidth>number in the deserialized version of the blob.
                      */

}  // namespace

StringLiteral vpux::VPU::getMemoryDerateAttrName() {
    return derateFactorAttrName;
}

StringLiteral vpux::VPU::getMemoryBandwidthAttrName() {
    return bandwidthAttrName;
}

uint32_t vpux::VPU::getMaxArchDPUClusterNum(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return VPUX37XX_MAX_DPU_GROUPS;
    case VPU::ArchKind::NPU40XX:
        return VPUX40XX_MAX_DPU_GROUPS;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

uint32_t vpux::VPU::getMaxArchDPUClusterNum(mlir::Operation* op) {
    return VPU::getMaxArchDPUClusterNum(VPU::getArch(op));
}

uint32_t vpux::VPU::getMaxDMAPorts(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return VPUX37XX_MAX_DMA_PORTS;
    case VPU::ArchKind::NPU40XX:
        return VPUX40XX_MAX_DMA_PORTS;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getDMABandwidth(ArchKind arch, VPU::RevisionID rev) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_2_7) / VPU::getDpuFrequency(arch, rev);
    case VPU::ArchKind::NPU40XX:
        return VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_4_0) / VPU::getDpuFrequency(arch, rev);
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getNCEThroughput(ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX:
        return 8000000.0;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

unsigned int vpux::VPU::getDpuFrequency(vpux::VPU::ArchKind arch, vpux::VPU::RevisionID rev) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return VPUNN::get_dpu_fclk(VPUNN::VPUDevice::VPU_2_7); /*!< The value 1300 corresponds to Highvcc of dpuclk.
                (See NPU37XX HAS #voltage-and-frequency-targets section).
                 */
    case VPU::ArchKind::NPU40XX:
        if (rev >= VPU::RevisionID::REVISION_B) {
            return 1850;  // MHz; TODO: switch to the value from vpunn, once this frequency is implemented. E#127567
        }
        return VPUNN::get_dpu_fclk(VPUNN::VPUDevice::VPU_4_0);
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    }
}

double vpux::VPU::getDmaBandwidthGBps(mlir::ModuleOp module) {
    const ArchKind arch = getArch(module);
    return getDmaBandwidthGBps(arch);
}

double vpux::VPU::getDmaBandwidthGBps(vpux::VPU::ArchKind arch) {
    double BW = 0;
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        BW = VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_2_7);  // 27000 MB/s
        break;
    case VPU::ArchKind::NPU40XX:
        BW = VPUNN::get_dram_bandwidth_MBps(VPUNN::VPUDevice::VPU_4_0);  // 45000 MB/s
        break;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", arch);
    };

    BW /= 1000;  // convert to GB/s
    return BW;
}

Byte vpux::VPU::getTotalCMXSize(mlir::ModuleOp module) {
    auto cmxRes = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);

    // This function is used to determine the best tile size. It tries to put maximum data in CMX.
    // Available CMX memory is decreased by two profilingBufferSize even if profiling is disabled
    // because we want to get exactly same compiled networks with profiling enabled and disabled.
    // Two buffer sizes are required in case when profiling allocates new buffer and old buffer
    // is still not disposed. Second buffer can be treated as an optimisation that prevents spilling.
    const int64_t profilingBufferSize = vpux::VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE +
                                        vpux::VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE +
                                        vpux::VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE;

    return cmxRes.size() - Byte(2 * profilingBufferSize);
}

Byte vpux::VPU::getTotalCMXSize(mlir::Operation* op) {
    return getTotalCMXSize(getModuleOp(op));
}

Byte vpux::VPU::getTotalCMXFragmentationAwareSize(mlir::ModuleOp module) {
    auto cmxRes = IE::getAvailableMemory(module,
                                         mlir::SymbolRefAttr::get(module.getContext(), VPU::CMX_NN_FragmentationAware));
    VPUX_THROW_UNLESS(cmxRes != nullptr, "Can't get information about {0} memory", VPU::CMX_NN_FragmentationAware);

    const ArchKind arch = getArch(module);

    // This function is used to determine the best tile size. It tries to put maximum data in CMX.
    // Available CMX memory is decreased by two profilingBufferSize even if profiling is disabled
    // because we want to get exactly same compiled networks with profiling enabled and disabled.
    // Two buffer sizes are required in case when profiling allocates new buffer and old buffer
    // is still not disposed. Second buffer can be treated as an optimisation that prevents spilling.
    const int64_t profilingBufferSize =
            vpux::VPUIP::HW_DMA_PROFILING_MAX_BUFFER_SIZE + vpux::VPUIP::HW_DPU_PROFILING_MAX_BUFFER_SIZE +
            ((arch == VPU::ArchKind::NPU37XX) ? vpux::VPUIP::HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE : 0);

    return cmxRes.size() - Byte(2 * profilingBufferSize);
}

Byte vpux::VPU::getTotalCMXFragmentationAwareSize(mlir::Operation* op) {
    return getTotalCMXFragmentationAwareSize(getModuleOp(op));
}

Byte vpux::VPU::getTotalCMXVFPipelineFragmentationAwareSize(mlir::Operation* op) {
    return Byte(static_cast<double>(getTotalCMXSize(op).count()) * vpux::FRAGMENTATION_AVOID_RATIO_VF_PIPELINING);
}

//
// ArchKind
//

namespace {

constexpr StringLiteral archAttrName = "VPU.arch";

constexpr Byte DDR_HEAP_SIZE = 4000_MB;

struct Resources {
    int numOfDPUGroups = 1;
    std::optional<int> numOfDMAPorts = std::nullopt;
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    Resources(int numOfDPUGroups, std::optional<int> numOfDMAPorts, std::optional<vpux::Byte> availableCMXMemory)
            : numOfDPUGroups(numOfDPUGroups), numOfDMAPorts(numOfDMAPorts), availableCMXMemory(availableCMXMemory) {
    }
};

struct SetResoursesFuncs {
    using AddExecutorFuncType = FuncRef<IE::ExecutorResourceOp(VPU::ExecutorKind, size_t)>;
    using AddTileExecutorFuncType = FuncRef<IE::TileResourceOp(size_t)>;
    using AddSubExecutorFuncType = FuncRef<IE::ExecutorResourceOp(IE::TileResourceOp, VPU::ExecutorKind, size_t)>;
    using AddMemoryFuncType = FuncRef<IE::MemoryResourceOp(mlir::SymbolRefAttr, Byte)>;
    using AddMemoryWithAttrsFuncType = FuncRef<void(mlir::SymbolRefAttr, Byte, double, size_t)>;
    using AddInnerMemoryFuncType = FuncRef<IE::MemoryResourceOp(IE::TileResourceOp, mlir::SymbolRefAttr, Byte)>;
    using AddInnerMemoryWithAttrsFuncType =
            FuncRef<void(IE::TileResourceOp, mlir::SymbolRefAttr, Byte, double, size_t)>;

    AddExecutorFuncType addExecutor;
    AddTileExecutorFuncType addTileExecutor;
    AddSubExecutorFuncType addSubExecutor;
    AddMemoryFuncType addMemory;
    AddMemoryWithAttrsFuncType addMemoryWithAttrs;
    AddInnerMemoryFuncType addInnerMemory;
    AddInnerMemoryWithAttrsFuncType addInnerMemoryWithAttrs;

    SetResoursesFuncs(AddExecutorFuncType addExecutor, AddTileExecutorFuncType addTileExecutor,
                      AddSubExecutorFuncType addSubExecutor, AddMemoryFuncType addMemory,
                      AddMemoryWithAttrsFuncType addMemoryWithAttrs, AddInnerMemoryFuncType addInnerMemory,
                      AddInnerMemoryWithAttrsFuncType addInnerMemoryWithAttrs)
            : addExecutor(addExecutor),
              addTileExecutor(addTileExecutor),
              addSubExecutor(addSubExecutor),
              addMemory(addMemory),
              addMemoryWithAttrs(addMemoryWithAttrs),
              addInnerMemory(addInnerMemory),
              addInnerMemoryWithAttrs(addInnerMemoryWithAttrs) {
    }
};

void setArch(mlir::ModuleOp module, VPU::ArchKind kind, const Resources& res, const SetResoursesFuncs& funcs,
             bool allowCustom) {
    VPUX_THROW_WHEN(!allowCustom && module->hasAttr(archAttrName),
                    "Architecture is already defined, probably you run '--init-compiler' twice");

    if (!module->hasAttr(archAttrName)) {
        module->setAttr(archAttrName, VPU::ArchKindAttr::get(module.getContext(), kind));
    }

    auto numOfDPUGroups = res.numOfDPUGroups;
    auto numOfDMAPorts = res.numOfDMAPorts;
    auto availableCMXMemory = res.availableCMXMemory;

    const auto getNumOfDMAPortsVal = [&](int maxDmaPorts) {
        int numOfDMAPortsVal = numOfDMAPorts.has_value() ? numOfDMAPorts.value() : maxDmaPorts;
        return numOfDMAPortsVal;
    };

    IE::TileResourceOp nceCluster;

    const auto ddrSymbolAttr = mlir::SymbolRefAttr::get(module.getContext(), stringifyEnum(VPU::MemoryKind::DDR));
    const auto cmxSymbolAttr = mlir::SymbolRefAttr::get(module.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto cmxFragAwareSymbolAttr = mlir::SymbolRefAttr::get(module.getContext(), VPU::CMX_NN_FragmentationAware);

    switch (kind) {
    case VPU::ArchKind::NPU37XX: {
        const auto workspaceCMXSize =
                availableCMXMemory.has_value() ? availableCMXMemory.value() : VPUX37XX_CMX_WORKSPACE_SIZE;
        const auto workspaceFragmentationAwareSize =
                availableCMXMemory.has_value()
                        ? Byte(static_cast<double>(availableCMXMemory.value().count()) * FRAGMENTATION_AVOID_RATIO)
                        : VPUX37XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE;

        funcs.addMemoryWithAttrs(ddrSymbolAttr, DDR_HEAP_SIZE, 0.6, 8);

        // Have NN_DMA as shared resource across clusters
        funcs.addExecutor(VPU::ExecutorKind::DMA_NN, getNumOfDMAPortsVal(VPUX37XX_MAX_DMA_PORTS));
        nceCluster = funcs.addTileExecutor(numOfDPUGroups);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::DPU, 1);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_NN, 1);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_ACT, 2);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxSymbolAttr, workspaceCMXSize, 1.0, 32);
        funcs.addInnerMemory(nceCluster, cmxFragAwareSymbolAttr, workspaceFragmentationAwareSize);

        break;
    }
    case VPU::ArchKind::NPU40XX: {
        const auto workspaceCMXSize =
                availableCMXMemory.has_value() ? availableCMXMemory.value() : VPUX40XX_CMX_WORKSPACE_SIZE;
        const auto workspaceFragmentationAwareSize =
                availableCMXMemory.has_value()
                        ? Byte(static_cast<double>(availableCMXMemory.value().count()) * FRAGMENTATION_AVOID_RATIO)
                        : VPUX40XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE;

        funcs.addMemoryWithAttrs(ddrSymbolAttr, DDR_HEAP_SIZE, 0.6, 64);

        // Have NN_DMA as shared resource across clusters
        auto numClusters = numOfDPUGroups;
        funcs.addExecutor(VPU::ExecutorKind::DMA_NN,
                          getNumOfDMAPortsVal(std::min(numClusters, VPUX40XX_MAX_DMA_PORTS)));
        funcs.addExecutor(VPU::ExecutorKind::M2I, 1);
        nceCluster = funcs.addTileExecutor(numClusters);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::DPU, 1);
        funcs.addSubExecutor(nceCluster, VPU::ExecutorKind::SHAVE_ACT, 2);
        funcs.addInnerMemoryWithAttrs(nceCluster, cmxSymbolAttr, workspaceCMXSize, 1.0, 64);
        funcs.addInnerMemory(nceCluster, cmxFragAwareSymbolAttr, workspaceFragmentationAwareSize);

        break;
    }
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }

    VPUX_THROW_WHEN(!allowCustom && nceCluster.hasProcessorFrequency(),
                    "Processor frequencyis already defined, probably you run '--init-compiler' twice");
}

}  // namespace

void vpux::VPU::setArch(mlir::ModuleOp module, ArchKind kind, int numOfDPUGroups, std::optional<int> numOfDMAPorts,
                        std::optional<vpux::Byte> availableCMXMemory, bool allowCustomValues) {
    const auto addExecutor = [&](VPU::ExecutorKind kind, size_t count) {
        VPUX_THROW_WHEN(!allowCustomValues && IE::hasExecutor(module, kind),
                        "Available executor kind '{0}' was already added", kind);
        if (IE::hasExecutor(module, kind)) {
            return IE::getAvailableExecutor(module, kind);
        }

        return IE::addAvailableExecutor(module, kind, count);
    };

    const auto addTileExecutor = [&](size_t count) {
        VPUX_THROW_WHEN(!allowCustomValues && IE::hasTileExecutor(module), "Available tile executor was already added");
        if (IE::hasTileExecutor(module)) {
            return IE::getTileExecutor(module);
        }

        return IE::addTileExecutor(module, count);
    };

    const auto addSubExecutor = [&](IE::TileResourceOp tileResOp, VPU::ExecutorKind kind, size_t count) {
        VPUX_THROW_WHEN(!allowCustomValues && tileResOp.hasSubExecutor(kind),
                        "Available executor kind '{0}' was already added", kind);
        if (tileResOp.hasSubExecutor(kind)) {
            return tileResOp.getSubExecutor(kind);
        }

        return tileResOp.addSubExecutor(kind, count);
    };

    const auto addAvailableMemory = [&](mlir::SymbolRefAttr memSpace, Byte size) {
        VPUX_THROW_WHEN(!allowCustomValues && IE::hasAvailableMemory(module, memSpace),
                        "Available memory kind '{0}' was already added", memSpace);
        if (IE::hasAvailableMemory(module, memSpace)) {
            return IE::getAvailableMemory(module, memSpace);
        }

        return IE::addAvailableMemory(module, memSpace, size);
    };

    const auto addMemWithAttrs = [&](mlir::SymbolRefAttr memSpace, Byte size, double derateFactor, size_t bandwidth) {
        auto mem = addAvailableMemory(memSpace, size);
        if (!mem->hasAttr(derateFactorAttrName)) {
            mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        }

        if (!mem->hasAttr(bandwidthAttrName)) {
            mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
        }
    };

    const auto addInnerAvailableMemory = [&](IE::TileResourceOp tileResOp, mlir::SymbolRefAttr memSpace, Byte size) {
        VPUX_THROW_WHEN(!allowCustomValues && tileResOp.hasAvailableMemory(memSpace),
                        "Available memory kind '{0}' was already added", memSpace);
        if (tileResOp.hasAvailableMemory(memSpace)) {
            return tileResOp.getAvailableMemory(memSpace);
        }

        return tileResOp.addAvailableMemory(memSpace, size);
    };

    const auto addInnerAvailableMemoryWithAttrs = [&](IE::TileResourceOp tileResOp, mlir::SymbolRefAttr memSpace,
                                                      Byte size, double derateFactor, size_t bandwidth) {
        auto mem = addInnerAvailableMemory(tileResOp, memSpace, size);
        if (!mem->hasAttr(derateFactorAttrName)) {
            mem->setAttr(derateFactorAttrName, getFPAttr(module.getContext(), derateFactor));
        }

        if (!mem->hasAttr(bandwidthAttrName)) {
            mem->setAttr(bandwidthAttrName, getIntAttr(module.getContext(), bandwidth));
        }
    };

    ::Resources res(numOfDPUGroups, numOfDMAPorts, availableCMXMemory);
    ::SetResoursesFuncs funcs(addExecutor, addTileExecutor, addSubExecutor, addAvailableMemory, addMemWithAttrs,
                              addInnerAvailableMemory, addInnerAvailableMemoryWithAttrs);

    return ::setArch(module, kind, res, funcs, allowCustomValues);
}

VPU::ArchKind vpux::VPU::getArch(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (auto attr = module->getAttr(archAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<VPU::ArchKindAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          archAttrName, attr);
        return attr.cast<VPU::ArchKindAttr>().getValue();
    }

    return VPU::ArchKind::UNKNOWN;
}

// To discern between VPUX3XXX and later on architectures
bool vpux::VPU::isArchVPUX3XXX(VPU::ArchKind arch) {
    return (arch == VPU::ArchKind::NPU37XX);
}

//
// CompilationMode
//

namespace {

constexpr StringLiteral compilationModeAttrName = "VPU.compilationMode";
constexpr StringLiteral descriptorHandleAttrName = "DescriptorHandle";

}  // namespace

void vpux::VPU::setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode) {
    module->setAttr(compilationModeAttrName, VPU::CompilationModeAttr::get(module.getContext(), compilationMode));
}

bool vpux::VPU::hasCompilationMode(mlir::ModuleOp module) {
    return module->hasAttr(compilationModeAttrName);
}

VPU::CompilationMode vpux::VPU::getCompilationMode(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (auto attr = module->getAttr(compilationModeAttrName)) {
        VPUX_THROW_UNLESS(attr.isa<VPU::CompilationModeAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                          compilationModeAttrName, attr);

        return attr.cast<VPU::CompilationModeAttr>().getValue();
    }

    // Use DefaultHW as a default mode
    return VPU::CompilationMode::DefaultHW;
}

//
// RevisionID
//

namespace {

constexpr StringLiteral revisionIDAttrName = "VPU.revisionID";

}  // namespace

void vpux::VPU::setRevisionID(mlir::ModuleOp module, RevisionID revisionID) {
    module->setAttr(revisionIDAttrName, VPU::RevisionIDAttr::get(module.getContext(), revisionID));
}

bool vpux::VPU::hasRevisionID(mlir::ModuleOp module) {
    return module->hasAttr(revisionIDAttrName);
}

VPU::RevisionID vpux::VPU::getRevisionID(mlir::Operation* op) {
    auto module = getModuleOp(op);

    if (module->hasAttr(revisionIDAttrName)) {
        if (auto attr = module->getAttr(revisionIDAttrName)) {
            VPUX_THROW_UNLESS(attr.isa<VPU::RevisionIDAttr>(), "Module attribute '{0}' has unsupported value '{1}'",
                              revisionIDAttrName, attr);

            return attr.cast<VPU::RevisionIDAttr>().getValue();
        }
    }

    return VPU::RevisionID::REVISION_NONE;
}

//
// PaddingAttr
//

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, int64_t left, int64_t right, int64_t top,
                                           int64_t bottom) {
    return PaddingAttr::get(ctx, getIntAttr(ctx, left), getIntAttr(ctx, right), getIntAttr(ctx, top),
                            getIntAttr(ctx, bottom));
}

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, ArrayRef<int64_t> padsBegin,
                                           ArrayRef<int64_t> padsEnd) {
    VPUX_THROW_UNLESS(padsBegin.size() == 2, "Paddings array has unsuppoted size '{0}'", padsBegin.size());
    VPUX_THROW_UNLESS(padsEnd.size() == 2, "Paddings array has unsuppoted size '{0}'", padsEnd.size());
    return getPaddingAttr(ctx, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]);
}

VPU::PaddingAttr vpux::VPU::getPaddingAttr(mlir::MLIRContext* ctx, const PadInfo& pad) {
    return getPaddingAttr(ctx, pad.left, pad.right, pad.top, pad.bottom);
}

PadInfo vpux::VPU::toPadInfo(PaddingAttr attr) {
    const auto left = attr.getLeft().getValue().getSExtValue();
    const auto right = attr.getRight().getValue().getSExtValue();
    const auto top = attr.getTop().getValue().getSExtValue();
    const auto bottom = attr.getBottom().getValue().getSExtValue();
    return PadInfo(left, right, top, bottom);
}

//
// PPEAttr
//

VPU::PPEMode vpux::VPU::getPPEMode(VPU::EltwiseType type) {
    switch (type) {
    case VPU::EltwiseType::ADD:
        return vpux::VPU::PPEMode::ADD;
    case VPU::EltwiseType::AND:
        return vpux::VPU::PPEMode::AND;
    case VPU::EltwiseType::MULTIPLY:
        return vpux::VPU::PPEMode::MULT;
    case VPU::EltwiseType::SUBTRACT:
        return vpux::VPU::PPEMode::SUB;
    case VPU::EltwiseType::MIN:
        return vpux::VPU::PPEMode::MINIMUM;
    case VPU::EltwiseType::MAX:
        return vpux::VPU::PPEMode::MAXIMUM;
    default:
        VPUX_THROW("Unsupported EltwiseType '{0}' for PPEMode", type);
    }
}

//
// DistributionInfoAttr
//

mlir::LogicalResult vpux::VPU::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                      DistributionInfoAttr distributedAttr, ArrayRef<int64_t> shape) {
    if (distributedAttr.getComputeShapes() != nullptr && distributedAttr.getComputeOffsets() == nullptr) {
        return printTo(emitError(), "Missing compute_offsets.");
    }

    if (distributedAttr.getComputeShapes() == nullptr && distributedAttr.getComputeOffsets() != nullptr) {
        return printTo(emitError(), "Missing compute_shapes.");
    }

    if (distributedAttr.getMemoryShapes() != nullptr && distributedAttr.getMemoryOffsets() == nullptr) {
        return printTo(emitError(), "Missing memory_offsets.");
    }

    if (distributedAttr.getMemoryShapes() == nullptr && distributedAttr.getMemoryOffsets() != nullptr) {
        return printTo(emitError(), "Missing memory_shapes.");
    }

    const bool hasComputeShapesOffsets =
            distributedAttr.getComputeShapes() != nullptr && distributedAttr.getComputeOffsets() != nullptr;
    const bool hasMemoryShapesOffsets =
            distributedAttr.getMemoryShapes() != nullptr && distributedAttr.getMemoryOffsets() != nullptr;

    if (hasComputeShapesOffsets && !hasMemoryShapesOffsets) {
        return printTo(emitError(), "Missing memory shapes and offsets.");
    }

    if (!hasComputeShapesOffsets && hasMemoryShapesOffsets) {
        return printTo(emitError(), "Missing compute shapes and offsets.");
    }

    const auto distributionMode = distributedAttr.getMode().getValue();

    if (distributionMode == VPU::DistributionMode::NONE) {
        return mlir::success();
    }

    if (distributedAttr.getNumClusters() == nullptr) {
        return printTo(emitError(), "Missing number of clusters.");
    }

    const auto numClusters = distributedAttr.getNumClusters().getInt();
    if (numClusters <= 0) {
        return printTo(emitError(), "The number of clusters must be greater than 0. Got: {0}", numClusters);
    }

    const auto neutralTilingScheme = SmallVector<int64_t>(shape.size(), 1);
    const auto tilingScheme = distributedAttr.getNumTiles() == nullptr
                                      ? neutralTilingScheme
                                      : vpux::parseIntArrayAttr<int64_t>(distributedAttr.getNumTiles());

    auto areShapesOffsetsValidForShape = [&](mlir::ArrayAttr perClusterShapesAttr,
                                             mlir::ArrayAttr perClusterOffsetsAttr) -> bool {
        if (perClusterShapesAttr.size() != perClusterOffsetsAttr.size() ||
            perClusterShapesAttr.size() != static_cast<size_t>(numClusters)) {
            return false;
        }

        auto perClusterShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(perClusterShapesAttr);
        auto perClusterOffsets = vpux::parseIntArrayOfArrayAttr<int64_t>(perClusterOffsetsAttr);
        for (int64_t cluster = 0; cluster < numClusters; cluster++) {
            if (shape.size() != perClusterShapes[cluster].size() || shape.size() != perClusterOffsets[cluster].size()) {
                return false;
            }

            for (size_t dim = 0; dim < shape.size(); dim++) {
                if (tilingScheme[dim] != 1) {
                    // If dim is split (SEG/OVERLAPPED) over clusters,
                    // ensure the start and end offsets are in range 0 -> dim_size - 1
                    if (perClusterOffsets[cluster][dim] < 0 ||
                        perClusterOffsets[cluster][dim] + perClusterShapes[cluster][dim] > shape[dim]) {
                        return false;
                    }

                    if (perClusterShapes[cluster][dim] <= 0 || perClusterShapes[cluster][dim] > shape[dim]) {
                        return false;
                    }
                } else {
                    // If dim is not split among clusters,
                    // ensure the start offset is 0, while the per cluster shape is equal to the full shape
                    if (perClusterOffsets[cluster][dim] != 0) {
                        return false;
                    }

                    if (perClusterShapes[cluster][dim] != shape[dim]) {
                        return false;
                    }
                }
            }
        }

        return true;
    };

    if (hasComputeShapesOffsets &&
        !areShapesOffsetsValidForShape(distributedAttr.getComputeShapes(), distributedAttr.getComputeOffsets())) {
        return printTo(emitError(), "Invalid compute shapes/offsets for tensor shape = {0}. Distribution = {1}", shape,
                       distributedAttr);
    }

    if (hasMemoryShapesOffsets &&
        !areShapesOffsetsValidForShape(distributedAttr.getMemoryShapes(), distributedAttr.getMemoryOffsets())) {
        return printTo(emitError(), "Invalid memory shapes/offsets for tensor shape = {0}. Distribution = {1}", shape,
                       distributedAttr);
    }

    const auto isTiledMode = [](VPU::DistributionMode mode) {
        return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::SEGMENTED) ||
               VPU::bitEnumContainsAny(mode, VPU::DistributionMode::OVERLAPPED);
    };

    if (!isTiledMode(distributionMode)) {
        return mlir::success();
    }

    if (distributedAttr.getNumTiles() == nullptr) {
        return printTo(emitError(), "Missing number of tiles.");
    }

    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    if (llvm::count_if(tilingScheme, isValidTile) != 1) {
        return printTo(emitError(), "Currently supporting single axis cluster tiling.");
    }

    const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));

    if (tilingScheme[axis] != numClusters) {
        return printTo(emitError(), "Incompatibility between tiling scheme '{0}' and number of clusters '{1}'",
                       tilingScheme[axis], numClusters);
    }

    // Limitations on tiling axes
    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        if (axis != Dims4D::Act::H.ind() && axis != Dims4D::Act::W.ind() && axis != Dims4D::Act::N.ind()) {
            return printTo(emitError(), "Overlapped cluster tiling is only supported for dimensions N, H and W");
        }

        if (distributedAttr.getAlignment() != nullptr) {
            const auto alignment = parseIntArrayAttr<int64_t>(distributedAttr.getAlignment());
            if (alignment[axis] != 1) {
                return printTo(
                        emitError(),
                        "Overlapped cluster tiling does not support alignment on the same axis used for tiling.");
            }
        }

        const bool overlappedWithKernelStridesPads = distributedAttr.getKernel() != nullptr &&
                                                     distributedAttr.getPads() != nullptr &&
                                                     distributedAttr.getStrides() != nullptr;

        if (!overlappedWithKernelStridesPads && !hasComputeShapesOffsets) {
            return printTo(emitError(), "Overlapped cluster tiling requires kernel, pads and strides or compute "
                                        "shapes and offsets to be set");
        }

        if (overlappedWithKernelStridesPads && hasComputeShapesOffsets) {
            return printTo(emitError(), "Overlapped cluster tiling must be defined by either kernel/strides/pads "
                                        "or compute shape/offsets, not both");
        }

        if (overlappedWithKernelStridesPads && axis == Dims4D::Act::N.ind()) {
            return printTo(emitError(), "Cannot have OVERLAPPED on dim N with kernel, pads, strides configuration ");
        }
    }

    if (distributedAttr.getAlignment() != nullptr) {
        const auto alignment = parseIntArrayAttr<int64_t>(distributedAttr.getAlignment());
        if (shape.size() != alignment.size()) {
            return printTo(emitError(), "Incompatibility in sizes between tensor shape '{0}' and alignment '{1}'",
                           shape.size(), alignment.size());
        }
    }

    if (distributedAttr.getNumTiles() != nullptr) {
        const auto numTiles = parseIntArrayAttr<int64_t>(distributedAttr.getNumTiles());
        if (shape.size() != numTiles.size()) {
            return printTo(emitError(), "Incompatibility in sizes between tensor shape '{0}' and tiling scheme '{1}'",
                           shape.size(), numTiles.size());
        }
    }

    if (distributedAttr.getKernel() != nullptr) {
        const auto kernel = parseIntArrayAttr<int64_t>(distributedAttr.getKernel());
        if (kernel.size() != 2) {
            return printTo(emitError(), "Expected kernel size to be 2. Got '{0}'", kernel.size());
        }
        const auto KY = kernel[Dims4D::Kernel::Y.ind()];
        const auto KX = kernel[Dims4D::Kernel::X.ind()];
        if (KY <= 0 || KX <= 0) {
            return printTo(emitError(), "Invalid kernel size: height '{0}', width '{1}'", KY, KX);
        }
    }

    if (distributedAttr.getPads() != nullptr) {
        const auto padTop = distributedAttr.getPads().getTop().getInt();
        const auto padBottom = distributedAttr.getPads().getBottom().getInt();
        const auto padLeft = distributedAttr.getPads().getLeft().getInt();
        const auto padRight = distributedAttr.getPads().getRight().getInt();
        if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
            return printTo(emitError(), "Invalid pads: top '{0}', bottom '{1}', left '{2}', right '{3}'", padTop,
                           padBottom, padLeft, padRight);
        }
    }

    if (distributedAttr.getStrides() != nullptr) {
        const auto strides = parseIntArrayAttr<int64_t>(distributedAttr.getStrides());
        if (strides.size() != 2) {
            return printTo(emitError(), "Expected strides size to be 2. Got '{0}'", strides.size());
        }
        const auto SY = strides[Dims4D::Strides::Y.ind()];
        const auto SX = strides[Dims4D::Strides::X.ind()];
        if (SY <= 0 || SX <= 0) {
            return printTo(emitError(), "Invalid strides: height '{0}', width '{1}'", SY, SX);
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::canTheDistributionModesBeCompatible(DistributionMode sourceMode,
                                                                   DistributionMode targetMode) {
    // Consecutive distribution modes for a SOK chain or from HKSwitch to SOK
    if ((sourceMode == (DistributionMode::DUPLICATED | DistributionMode::SEGMENTED) ||
         sourceMode == (DistributionMode::MULTICASTED | DistributionMode::SEGMENTED)) &&
        targetMode == DistributionMode::DUPLICATED) {
        return mlir::success();
    }

    // DUPLICATED -> SEG | DUPLICATED: None const weights for Matmul
    // DUPLICATED -> SEG | MULTICASTED: Subview to NCEClusterTiling output
    if (sourceMode == DistributionMode::DUPLICATED &&
        (targetMode == (DistributionMode::DUPLICATED | DistributionMode::SEGMENTED) ||
         targetMode == (DistributionMode::MULTICASTED | DistributionMode::SEGMENTED))) {
        return mlir::success();
    }

    // SEGMENTED & OVERLAPPED can be compatible if their memory view is equal
    if ((sourceMode == DistributionMode::SEGMENTED && targetMode == DistributionMode::OVERLAPPED) ||
        (sourceMode == DistributionMode::OVERLAPPED && targetMode == DistributionMode::SEGMENTED)) {
        return mlir::success();
    }

    return mlir::failure();
}

mlir::LogicalResult vpux::VPU::areDistributionNumClustersCompatible(int64_t sourceNumClusters,
                                                                    int64_t targetNumClusters) {
    return mlir::success(sourceNumClusters >= targetNumClusters);
}

mlir::LogicalResult vpux::VPU::areDistributionNumClustersCompatible(mlir::IntegerAttr sourceNumClusters,
                                                                    mlir::IntegerAttr targetNumClusters) {
    return areDistributionNumClustersCompatible(sourceNumClusters.getInt(), targetNumClusters.getInt());
}

mlir::LogicalResult vpux::VPU::areDistributionElementTypesCompatible(mlir::Type inType, mlir::Type outType) {
    if (inType != outType) {
        // allow different quantization parameters
        if (!inType.isa<mlir::quant::QuantizedType>() || !outType.isa<mlir::quant::QuantizedType>()) {
            return mlir::failure();
        }
        if (vpux::getElemTypeSize(inType) != vpux::getElemTypeSize(outType)) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

int64_t vpux::VPU::getDistributedTilingAxis(ArrayRef<int64_t> tilingScheme) {
    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    return std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
}

bool vpux::VPU::isDistributedAttrWithExplicitShapesAndOffsets(VPU::DistributionInfoAttr distributionAttr) {
    const bool hasComputeShapesOffsets =
            distributionAttr.getComputeShapes() != nullptr && distributionAttr.getComputeOffsets() != nullptr;
    const bool hasMemoryShapesOffsets =
            distributionAttr.getMemoryShapes() != nullptr && distributionAttr.getMemoryOffsets() != nullptr;

    return hasComputeShapesOffsets && hasMemoryShapesOffsets;
}

bool vpux::VPU::isDistributionWithExplicitShapesAndOffsets(const VPU::DistributionInfo& distribution) {
    const bool hasComputeShapesOffsets =
            !distribution.getComputeShapes().empty() && !distribution.getComputeOffsets().empty();
    const bool hasMemoryShapesOffsets =
            !distribution.getMemoryShapes().empty() && !distribution.getMemoryOffsets().empty();

    return hasComputeShapesOffsets && hasMemoryShapesOffsets;
}

bool vpux::VPU::isUniformDistributedSegmentsSupported(mlir::Operation* op) {
    return !VPU::isArchVPUX3XXX(VPU::getArch(op));
}

//
// Tiling utils
//

// Segmentation logic operates on schema and runtime assumption that a segmented tensor should be split equally
// across the axis, with the remainder cluster possibly having a smaller tile.
std::optional<SmallVector<Shape>> VPU::splitSegmentedShape(ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme,
                                                           const int64_t numClusters, const int64_t axis,
                                                           std::optional<ArrayRef<int64_t>> alignment,
                                                           bool uniformDistributedSegments) {
    VPUX_THROW_UNLESS(axis < int64_t(shape.size()),
                      "An invalid split axis {0} specified, the shape tensor is {1} dimensional", axis, shape.size());
    VPUX_THROW_UNLESS(tilingScheme[axis] == numClusters,
                      "The number of tiles on axis {0} must be equal to the number of clusters specified for "
                      "compilation {1} but got {2}",
                      axis, tilingScheme[axis], numClusters);

    SmallVector<Shape> segmentedTiles;
    auto tiledShape = to_small_vector(shape);
    auto remainderTileShape = to_small_vector(shape);
    if (!uniformDistributedSegments) {
        // Split in an equal manner such that first N-1 tiles are equal
        // and the last tile can be less or equal.
        tiledShape[axis] = divUp(tiledShape[axis], tilingScheme[axis]);
        tiledShape = alignShape(tiledShape, alignment, alignValUp<int64_t>);

        // Last tile will have the remainder and it doesn't have to be aligned
        remainderTileShape[axis] = shape[axis] - tiledShape[axis] * (tilingScheme[axis] - 1);
        if (remainderTileShape[axis] <= 0) {
            return std::nullopt;
        }
        segmentedTiles.insert(segmentedTiles.end(), numClusters - 1, Shape(tiledShape));
        segmentedTiles.push_back(Shape(remainderTileShape));
    } else {
        // Split into a more balanced approach such that there's
        // a minimum different between the segments sizes.
        // For example a height of 6 is split across 4 tile as [2, 2, 1, 1].

        // Compute baseline tile, specifically also align it
        tiledShape[axis] = tiledShape[axis] / tilingScheme[axis];
        tiledShape = alignShape(tiledShape, alignment, alignValDown<int64_t>);
        if (tiledShape[axis] <= 0) {
            return std::nullopt;
        }
        // Remainder of data is distributed across first few tiles
        remainderTileShape = tiledShape;
        auto remainderCount = shape[axis] - tiledShape[axis] * tilingScheme[axis];
        auto axisAlignment = 1;
        if (alignment.has_value()) {
            axisAlignment = alignment.value()[axis];
        }
        if (remainderCount % axisAlignment) {
            return std::nullopt;
        }
        auto remainderElements = remainderCount / axisAlignment;
        remainderTileShape[axis] = tiledShape[axis] + axisAlignment;

        segmentedTiles.insert(segmentedTiles.end(), remainderElements, Shape(remainderTileShape));
        segmentedTiles.insert(segmentedTiles.end(), numClusters - remainderElements, Shape(tiledShape));
    }
    return segmentedTiles;
}

std::optional<SmallVector<DimRange>> getOverlappedInputTileDimRanges(
        ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme, ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
        const std::optional<VPU::Padding>& pad, const int64_t axis, const int64_t numClusters,
        const bool uniformDistributedSegments) {
    const auto axisDim = Dim(axis);
    VPUX_THROW_UNLESS(axisDim == Dims4D::Act::W || axisDim == Dims4D::Act::H,
                      "Input overlapping supported only for W or H axes");

    const auto N = shape[Dims4D::Act::N.ind()];
    const auto C = shape[Dims4D::Act::C.ind()];
    const auto Y = shape[Dims4D::Act::H.ind()];
    const auto X = shape[Dims4D::Act::W.ind()];

    VPUX_THROW_UNLESS(pad.has_value(), "Pads value is required");
    auto padInfo = vpux::PadInfo(pad.value().getLeftPad(), pad.value().getRightPad(), pad.value().getTopPad(),
                                 pad.value().getBottomPad());

    const auto getOutputHW = vpux::spatialOutputForInputWindowSize({Y, X}, kernel, strides, padInfo);
    if (!getOutputHW.has_value()) {
        return std::nullopt;
    }
    const auto outputHW = getOutputHW.value();

    const SmallVector<int64_t> outputShape{N, C, outputHW.first, outputHW.second};

    // Alignment should only be considered for final input shape,
    // not the intermediary output shape

    const auto segmentedShape = VPU::splitSegmentedShape(outputShape, tilingScheme, numClusters, axis, std::nullopt,
                                                         uniformDistributedSegments);

    if (!segmentedShape.has_value()) {
        return std::nullopt;
    }

    const auto outputTiles = segmentedShape.value();

    int64_t offset = 0;
    VPUX_THROW_WHEN(kernel.empty(), "Kernel shouldn't be empty");
    const auto KY = kernel[Dims4D::Kernel::Y.ind()];
    const auto KX = kernel[Dims4D::Kernel::X.ind()];

    VPUX_THROW_WHEN(strides.empty(), "Strides shouldn't be empty");
    const auto SY = strides[Dims4D::Strides::Y.ind()];
    const auto SX = strides[Dims4D::Strides::X.ind()];

    const auto padTop = pad.value().getTopPad();
    const auto padBottom = pad.value().getBottomPad();
    const auto padLeft = pad.value().getLeftPad();
    const auto padRight = pad.value().getRightPad();
    SmallVector<DimRange> inputTileDimRanges;
    for (const auto& outputTile : outputTiles) {
        const auto dimSize = outputTile[Dim(axis)];
        const DimRange tileSize(offset, offset + dimSize);
        offset += dimSize;

        DimRange inputTile(0, 0);
        if (axis == Dims4D::Act::H.ind()) {
            std::tie(inputTile, std::ignore, std::ignore) =
                    vpux::inputForOutputDim(tileSize, KY, SY, {0, Y}, padTop, padBottom);
        } else if (axis == Dims4D::Act::W.ind()) {
            std::tie(inputTile, std::ignore, std::ignore) =
                    vpux::inputForOutputDim(tileSize, KX, SX, {0, X}, padLeft, padRight);
        } else {
            VPUX_THROW("Unsupported axis '{0}'", axis);
        }
        inputTileDimRanges.push_back(inputTile);
    }
    return inputTileDimRanges;
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapes(ShapeRef shapeRef, DistributionInfoAttr distributionAttr) {
    return getPerClusterComputeShapes(shapeRef, VPU::DistributionInfo::getClassFromAttr(distributionAttr));
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapes(ShapeRef shapeRef, const VPU::DistributionInfo& distribution) {
    auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distribution.getDistributionMode();

    const auto numClusters = distribution.getNumClusters();
    auto tiledComputeShapes = SmallVector<Shape>(numClusters);

    std::optional<ArrayRef<int64_t>> optionalAlignment = std::nullopt;
    auto alignment = SmallVector<int64_t>(distribution.getAlignment());
    if (!alignment.empty()) {
        optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
    }

    auto getComputeSplitIntoSegments = [&]() -> SmallVector<Shape> {
        const auto tilingScheme = distribution.getNumTiles();
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
        VPUX_THROW_UNLESS(axis < int64_t(tilingScheme.size()), "Segmented tiling scheme requires at least 1 dimension "
                                                               "to be segmented but the tiling schema is [1, 1, 1, 1]");
        const auto segmentedShape = VPU::splitSegmentedShape(shape, tilingScheme, numClusters, axis, optionalAlignment,
                                                             distribution.hasUniformDistributedSegments());
        VPUX_THROW_UNLESS(segmentedShape.has_value(), "Improper split, '{0}' over '{1}' tiles", shape[axis],
                          tilingScheme[axis]);
        return segmentedShape.value();
    };

    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        return getComputeSplitIntoSegments();
    }

    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        if (distribution.hasEqualMemoryAndComputeView()) {
            const auto optionalPerClusterMemoryShapes = getPerClusterMemoryShapes(shapeRef, distribution);

            VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                              "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distribution);
            return optionalPerClusterMemoryShapes.value();
        }

        return getComputeSplitIntoSegments();
    }

    if (distributionMode == VPU::DistributionMode::DUPLICATED ||
        distributionMode == VPU::DistributionMode::MULTICASTED) {
        std::fill_n(tiledComputeShapes.begin(), tiledComputeShapes.size(),
                    Shape(alignShape(shape, optionalAlignment, alignValUp<int64_t>)));
        return tiledComputeShapes;
    }

    VPUX_THROW("Cannot get per cluster memory shapes. Unsupported distribution: {0}", distribution);
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapeOffsets(ShapeRef shapeRef,
                                                               DistributionInfoAttr distributionAttr) {
    return getPerClusterComputeShapeOffsets(shapeRef, VPU::DistributionInfo::getClassFromAttr(distributionAttr));
}

SmallVector<Shape> vpux::VPU::getPerClusterComputeShapeOffsets(ShapeRef shapeRef,
                                                               const VPU::DistributionInfo& distribution) {
    const auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distribution.getDistributionMode();

    const auto numClusters = distribution.getNumClusters();
    auto tiledComputeShapeOffsets = SmallVector<Shape>(numClusters, Shape(shapeRef.size(), 0));

    auto getOffsetsForSegments = [&](SmallVector<Shape>& perClusterOffsets) -> SmallVector<Shape> {
        const auto tiledComputeShapes = getPerClusterComputeShapes(shapeRef, distribution);
        const auto tilingScheme = distribution.getNumTiles();
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);

        int64_t offset = 0;
        for (int64_t idx = 0; idx < numClusters; idx++) {
            perClusterOffsets[idx][Dim(axis)] = offset;
            offset += tiledComputeShapes[idx][Dim(axis)];
        }

        return perClusterOffsets;
    };

    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        return getOffsetsForSegments(tiledComputeShapeOffsets);
    }

    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        if (distribution.hasEqualMemoryAndComputeView()) {
            return getPerClusterMemoryShapeOffsets(shapeRef, distribution);
        }

        return getOffsetsForSegments(tiledComputeShapeOffsets);
    }

    if (distributionMode == VPU::DistributionMode::DUPLICATED ||
        distributionMode == VPU::DistributionMode::MULTICASTED) {
        return tiledComputeShapeOffsets;
    }

    VPUX_THROW("Cannot get per cluster memory shapes. Unsupported distribution: {0}", distribution);
}

std::optional<SmallVector<Shape>> vpux::VPU::getPerClusterMemoryShapes(ShapeRef shapeRef,
                                                                       DistributionInfoAttr distributionAttr)

{
    return getPerClusterMemoryShapes(shapeRef, VPU::DistributionInfo::getClassFromAttr(distributionAttr));
}

std::optional<SmallVector<Shape>> vpux::VPU::getPerClusterMemoryShapes(ShapeRef shapeRef,
                                                                       const VPU::DistributionInfo& distribution) {
    auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distribution.getDistributionMode();

    const auto numClusters = distribution.getNumClusters();
    auto tiledMemoryShapes = SmallVector<Shape>(numClusters);

    std::optional<ArrayRef<int64_t>> optionalAlignment = std::nullopt;
    auto alignment = SmallVector<int64_t>(distribution.getAlignment());
    if (!alignment.empty()) {
        optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
    }

    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::MULTICASTED)) {
        std::fill_n(tiledMemoryShapes.begin(), tiledMemoryShapes.size(),
                    Shape(alignShape(shape, optionalAlignment, alignValUp<int64_t>)));

        return tiledMemoryShapes;
    }

    if (distributionMode == VPU::DistributionMode::SEGMENTED) {
        const auto tilingScheme = distribution.getNumTiles();
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
        VPUX_THROW_UNLESS(axis < int64_t(tilingScheme.size()), "Segmented tiling scheme requires at least 1 dimension "
                                                               "to be segmented but the tiling schema is [1, 1, 1, 1]");
        return vpux::VPU::splitSegmentedShape(shape, tilingScheme, numClusters, axis, optionalAlignment,
                                              distribution.hasUniformDistributedSegments());
    }

    if (distributionMode == VPU::DistributionMode::OVERLAPPED) {
        const auto tilingScheme = distribution.getNumTiles();
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);

        const auto optionalInputTileDimRanges = getOverlappedInputTileDimRanges(
                shape, tilingScheme, distribution.getKernel(), distribution.getStrides(), distribution.getPadding(),
                axis, numClusters, distribution.hasUniformDistributedSegments());

        if (!optionalInputTileDimRanges.has_value()) {
            return std::nullopt;
        }

        const auto inputTileDimRanges = optionalInputTileDimRanges.value();

        for (auto p : inputTileDimRanges | indexed) {
            const auto inputTile = p.value();
            const auto cluster = p.index();
            shape[axis] = inputTile.end - inputTile.begin;
            tiledMemoryShapes[cluster] = Shape(alignShape(shape, optionalAlignment, alignValUp<int64_t>));
        }

        return tiledMemoryShapes;
    }

    VPUX_THROW("Cannot get per cluster memory shapes. Unsupported distribution: {0}", distribution);
}

SmallVector<Shape> vpux::VPU::getPerClusterMemoryShapeOffsets(ShapeRef shapeRef,
                                                              DistributionInfoAttr distributionAttr) {
    return getPerClusterMemoryShapeOffsets(shapeRef, VPU::DistributionInfo::getClassFromAttr(distributionAttr));
}

SmallVector<Shape> vpux::VPU::getPerClusterMemoryShapeOffsets(ShapeRef shapeRef,
                                                              const VPU::DistributionInfo& distribution) {
    const auto shape = to_small_vector(shapeRef.raw());
    const auto distributionMode = distribution.getDistributionMode();

    const auto numClusters = distribution.getNumClusters();

    auto tiledMemoryOffsets = SmallVector<Shape>(numClusters, Shape(shapeRef.size(), 0));

    // For distribution mode containing either DUPLICATED or MULTICASTED, the starting offset
    // will be 0 across all dimensions since the entire output tensor can be found in each cluster
    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::MULTICASTED)) {
        return tiledMemoryOffsets;
    }

    if (distributionMode == VPU::DistributionMode::SEGMENTED) {
        const auto optionalPerClusterMemoryShapes = getPerClusterMemoryShapes(shapeRef, distribution);

        VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                          "Cannot get per cluster memory shape offsets. Unsupported distribution: {0}", distribution);

        const auto tiledComputeShapes = optionalPerClusterMemoryShapes.value();
        const auto tilingScheme = distribution.getNumTiles();
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);

        int64_t offset = 0;
        for (int64_t idx = 0; idx < numClusters; idx++) {
            tiledMemoryOffsets[idx][Dim(axis)] = offset;
            offset += tiledComputeShapes[idx][Dim(axis)];
        }

        return tiledMemoryOffsets;
    }

    if (distributionMode == VPU::DistributionMode::OVERLAPPED) {
        const auto tilingScheme = distribution.getNumTiles();
        const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);

        const auto optionalInputTileDimRanges = getOverlappedInputTileDimRanges(
                shape, tilingScheme, distribution.getKernel(), distribution.getStrides(), distribution.getPadding(),
                axis, numClusters, distribution.hasUniformDistributedSegments());

        VPUX_THROW_UNLESS(optionalInputTileDimRanges.has_value(),
                          "Cannot get per cluster memory shape offsets. Unsupported distribution: {0}", distribution);

        const auto inputTileDimRanges = optionalInputTileDimRanges.value();
        for (auto p : inputTileDimRanges | indexed) {
            const auto inputTile = p.value();
            const auto cluster = p.index();
            tiledMemoryOffsets[cluster][Dim(axis)] = inputTile.begin;
        }

        return tiledMemoryOffsets;
    }

    VPUX_THROW("Cannot get per cluster memory shapes. Unsupported distribution: {0}", distribution);
}

SmallVector<Shape> vpux::VPU::getOverlappedPerClusterNewMemoryShapes(ShapeRef newShape, ShapeRef origShape,
                                                                     DistributionInfoAttr distributionAttr) {
    auto shape = to_small_vector(newShape.raw());
    auto originalShape = to_small_vector(origShape.raw());
    const auto distributionMode = distributionAttr.getMode().getValue();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    auto tiledMemoryShapes = SmallVector<Shape>(numClusters);
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());

    VPUX_THROW_UNLESS(distributionMode == VPU::DistributionMode::OVERLAPPED,
                      "Only support OVERLAPPED mode, current mode - {0}",
                      VPU::stringifyDistributionMode(distributionMode));

    VPUX_THROW_UNLESS(distributionAttr.getMemoryShapes() != nullptr,
                      "Only support distributedAttr with explicit shapes and offsets");

    for (auto dim : irange(originalShape.size())) {
        VPUX_THROW_WHEN(tilingScheme[dim] > 1 && originalShape[dim] != shape[dim],
                        "Shape change dim should not be on the same dim as tiling");
    }

    const auto origPerClusterShapes = parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getMemoryShapes());
    for (size_t cluster = 0; cluster < static_cast<size_t>(numClusters); cluster++) {
        for (size_t dim = 0; dim < shape.size(); dim++) {
            if (tilingScheme[dim] != 1) {
                shape[dim] = origPerClusterShapes[cluster][dim];
            }
        }
        tiledMemoryShapes[cluster] = Shape(shape);
    }

    return tiledMemoryShapes;
}

SmallVector<Shape> vpux::VPU::getOverlappedPerClusterNewMemoryShapeOffsets(ShapeRef shapeRef,
                                                                           DistributionInfoAttr distributionAttr) {
    const auto distributionMode = distributionAttr.getMode().getValue();
    const auto numClusters = distributionAttr.getNumClusters().getInt();
    auto tiledMemoryOffsets = SmallVector<Shape>(numClusters, Shape(shapeRef.size(), 0));

    VPUX_THROW_UNLESS(distributionMode == VPU::DistributionMode::OVERLAPPED,
                      "Only support OVERLAPPED mode, current mode - {0}",
                      VPU::stringifyDistributionMode(distributionMode));

    VPUX_THROW_UNLESS(distributionAttr.getMemoryOffsets() != nullptr,
                      "Only support distributedAttr with explicit shapes and offsets");

    auto offsets = parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getMemoryOffsets());
    for (auto cluster : irange(offsets.size())) {
        tiledMemoryOffsets[cluster] = Shape(offsets[cluster]);
    }

    return tiledMemoryOffsets;
}

SmallVector<PadInfo> vpux::VPU::getPerClusterPadding(DistributionInfoAttr distributionAttr, PadInfo kernelPadding) {
    const auto mode = distributionAttr.getMode().getValue();
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::OVERLAPPED,
                      "Currently getting per cluster padding is supported only for OVERLAPPED, mode - {0}",
                      VPU::stringifyDistributionMode(mode));

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto axisDim = Dim(vpux::VPU::getDistributedTilingAxis(tilingScheme));

    VPUX_THROW_UNLESS(axisDim == Dims4D::Act::H || axisDim == Dims4D::Act::W,
                      "Currently getting per cluster padding is supported only for tiling axis H or W, axis - {0}",
                      axisDim);

    SmallVector<PadInfo> perClusterPadInfo;
    const auto top = kernelPadding.top;
    const auto bottom = kernelPadding.bottom;
    const auto left = kernelPadding.left;
    const auto right = kernelPadding.right;

    const auto firstClusterPadInfo =
            (axisDim == Dims4D::Act::H) ? PadInfo(left, right, top, 0) : PadInfo(left, 0, top, bottom);
    const auto lastClusterPadInfo =
            (axisDim == Dims4D::Act::H) ? PadInfo(left, right, 0, bottom) : PadInfo(0, right, top, bottom);

    perClusterPadInfo.push_back(firstClusterPadInfo);
    for (auto cluster = 1; cluster < distributionAttr.getNumClusters().getInt() - 1; cluster++) {
        const auto padInfo = (axisDim == Dims4D::Act::H) ? PadInfo(left, right, 0, 0) : PadInfo(0, 0, top, bottom);
        perClusterPadInfo.push_back(padInfo);
    }
    perClusterPadInfo.push_back(lastClusterPadInfo);

    return perClusterPadInfo;
}

SmallVector<StridedShape> vpux::VPU::getPerClusterMemoryStridedShapes(ShapeRef shape, StridesRef strides,
                                                                      DimsOrder dimsOrder, DistributionModeAttr mode,
                                                                      ArrayRef<Shape> memoryShapes) {
    const auto distributionMode = mode.getValue();

    SmallVector<StridedShape> stridedShapes;
    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::DUPLICATED)) {
        for (const auto& memoryShape : memoryShapes) {
            stridedShapes.emplace_back(memoryShape, strides);
        }
        return stridedShapes;
    }

    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED) ||
        VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        const auto adaptedStrides = adaptStrides(shape, strides, memoryShapes, dimsOrder);
        for (const auto& p : zip(memoryShapes, adaptedStrides)) {
            stridedShapes.emplace_back(std::get<0>(p), std::get<1>(p));
        }
        return stridedShapes;
    }

    VPUX_THROW("Unsupported mode '{0}'", VPU::stringifyEnum(distributionMode));
}

SmallVector<Shape> vpux::VPU::arrayAttrToVecOfShapes(mlir::ArrayAttr arr) {
    SmallVector<Shape> shapesVec;
    const auto parsedVec = parseIntArrayOfArrayAttr<int64_t>(arr);
    for (auto ind : irange(parsedVec.size())) {
        shapesVec.push_back(Shape(parsedVec[ind]));
    }

    return shapesVec;
}

bool vpux::VPU::isSegmentedOverH(VPU::DistributionInfoAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::C.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isSegmentedOverC(VPU::DistributionInfoAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::H.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isSegmentedDuplicatedOverC(VPU::DistributionInfoAttr distAttr) {
    if (distAttr.getMode().getValue() != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::H.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isSegmentedOverN(VPU::DistributionInfoAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::C.ind()] > 1 || numTiles[Dims4D::Act::H.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isOverlappedOverH(VPU::DistributionInfoAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::C.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isOverlappedOverH(VPU::DistributionInfo& distribution) {
    if (distribution.getDistributionMode() != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    const auto numTiles = distribution.getNumTiles();
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::C.ind()] > 1 ||
        numTiles[Dims4D::Act::W.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isOverlappedOverW(VPU::DistributionInfoAttr distAttr) {
    if (distAttr.getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    const auto numTiles = parseIntArrayAttr<int64_t>(distAttr.getNumTiles());
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::C.ind()] > 1 ||
        numTiles[Dims4D::Act::H.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isOverlappedOverW(VPU::DistributionInfo& distribution) {
    if (distribution.getDistributionMode() != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }
    const auto numTiles = distribution.getNumTiles();
    if (numTiles.size() != 4 || numTiles[Dims4D::Act::N.ind()] > 1 || numTiles[Dims4D::Act::C.ind()] > 1 ||
        numTiles[Dims4D::Act::H.ind()] > 1) {
        return false;
    }
    return true;
}

bool vpux::VPU::isDuplicated(VPU::DistributionInfoAttr distAttr) {
    const auto mode = distAttr.getMode().getValue();

    return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
           VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);
}

//
// SparsityCompressionAttr
//

int64_t VPU::SparsityCompressionAttr::getTotalNumElems() const {
    if (getNumElems().empty()) {
        return 0;
    }
    auto numElems = getNumElems().getValues<int64_t>();
    return std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0));
}

int64_t VPU::SparsityCompressionAttr::getNumElemsInRange(int64_t startIdx, int64_t size) const {
    const auto numElems = getNumElems().getValues<int64_t>();
    const auto startIt = numElems.begin() + startIdx;
    const auto endIt = startIt + size;
    return std::accumulate(startIt, endIt, static_cast<int64_t>(0));
}

Byte VPU::SparsityCompressionAttr::getAllocSize(mlir::Type elemType) const {
    const auto elemByteSize = getElemTypeSize(elemType).to<Byte>().count();
    const int64_t alignment = (getAlignment() != nullptr) ? getAlignment().getInt() : 1;
    const auto numElems = getNumElems().getValues<int64_t>();
    int64_t totalAllocSize = 0;
    for (auto num : numElems) {
        totalAllocSize += alignValUp<int64_t>(num * elemByteSize, alignment);
    }
    return Byte(totalAllocSize);
}

VPU::SparsityCompressionAttr VPU::getSparsityCompressionAttr(mlir::Type type) {
    if (auto sparseType = type.dyn_cast_or_null<VPU::SparseTensorType>()) {
        return sparseType.getSparsityCompression();
    }
    return nullptr;
}

mlir::Type VPU::setSparsityCompressionAttr(mlir::Type type, VPU::SparsityCompressionAttr sparsityCompressionAttr) {
    if (auto sparseType = type.dyn_cast_or_null<VPU::SparseTensorType>()) {
        return VPU::SparseTensorType::get(sparseType.getData(), sparseType.getSparsityMap(),
                                          sparseType.getStorageElementTable(), sparseType.getIsWeights(),
                                          sparsityCompressionAttr);
    }
    return type;
}

VPU::SparsityCompressionAttr VPU::tileSparsityCompression(VPU::SparsityCompressionAttr sparsityCompression,
                                                          ShapeRef tileOffsets, ShapeRef tileShape) {
    if (sparsityCompression == nullptr) {
        return nullptr;
    }
    VPUX_THROW_UNLESS(sparsityCompression.getAxis() != nullptr,
                      "Cannot tile compression scheme that is not over an axis");
    const size_t axis = sparsityCompression.getAxis().getInt();
    VPUX_THROW_UNLESS(axis < tileOffsets.size() && axis < tileShape.size(),
                      "Axis {0} outside the range of tile dimensions: offsets size {1}, shape size {2}", axis,
                      tileOffsets.size(), tileShape.size());

    const auto numElems = sparsityCompression.getNumElems().getValues<int64_t>();
    const auto dimOffset = tileOffsets[Dim(axis)];
    const auto dimShape = tileShape[Dim(axis)];

    const auto startIt = numElems.begin() + dimOffset;
    const auto endIt = startIt + dimShape;
    const auto tileNumElems = SmallVector<int64_t>(startIt, endIt);

    auto ctx = sparsityCompression.getContext();
    const auto tileNumElemsType =
            mlir::RankedTensorType::get({static_cast<int64_t>(tileNumElems.size())}, getInt64Type(ctx));
    const auto tileNumElemsAttr = mlir::DenseElementsAttr::get(tileNumElemsType, ArrayRef(tileNumElems));
    return VPU::SparsityCompressionAttr::get(ctx, sparsityCompression.getAxis(), tileNumElemsAttr,
                                             sparsityCompression.getAlignment());
}

SmallVector<SmallVector<int64_t>> VPU::arrayOfArrayFromShape(ArrayRef<Shape> shape) {
    SmallVector<SmallVector<int64_t>> ret;
    for (const auto& a : shape) {
        ret.push_back(a.raw());
    }
    return ret;
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/enums.cpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPU/attributes.cpp.inc>

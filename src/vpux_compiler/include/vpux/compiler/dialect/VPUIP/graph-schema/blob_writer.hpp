//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/act_kernels/compilation.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <flatbuffers/flatbuffers.h>

#include <llvm/ADT/MapVector.h>
#include <unordered_map>

#include "vpux/utils/core/quant_params.hpp"

namespace vpux {
namespace VPUIP {

using DMADescriptorReference = MVCNN::DMADescriptorReference;

class BlobWriter final {
public:
    using Task = flatbuffers::Offset<MVCNN::Task>;
    using TaskList = flatbuffers::Offset<MVCNN::TaskList>;

    struct SpecificTask {
        flatbuffers::Offset<void> obj;
        MVCNN::SpecificTask type;
    };

    struct SoftwareLayerParams {
        flatbuffers::Offset<void> obj;
        MVCNN::SoftwareLayerParams type;
    };

    using TensorReference = flatbuffers::Offset<MVCNN::TensorReference>;
    using IndirectDataReference = flatbuffers::Offset<MVCNN::IndirectDataReference>;

    using Barrier = flatbuffers::Offset<MVCNN::Barrier>;
    using BarrierReference = flatbuffers::Offset<MVCNN::BarrierReference>;

    using BinaryData = flatbuffers::Offset<MVCNN::BinaryData>;

    using KernelData = flatbuffers::Offset<MVCNN::KernelData>;
    using ActKernel = flatbuffers::Offset<MVCNN::ActKernel>;
    using KernelDataRef = flatbuffers::Offset<MVCNN::KernelDataReference>;
    using ActShavesKernelDataMap =
            llvm::MapVector<std::string, SerializedKernelDataDesc, std::unordered_map<std::string, size_t>>;

    using OVParameters = flatbuffers::Offset<MVCNN::OVNode>;
    using OVResults = flatbuffers::Offset<MVCNN::OVNode>;
    using OVNodes = flatbuffers::Offset<MVCNN::OVNode>;

    using HaloRegion = flatbuffers::Offset<MVCNN::HaloRegion>;

    using String = flatbuffers::Offset<flatbuffers::String>;

    template <typename T>
    using Vector = flatbuffers::Offset<flatbuffers::Vector<T>>;

public:
    BlobWriter(Logger log, VPU::ArchKind architecture): _log(log), _architecture(architecture) {
    }

public:
    Task createTask(mlir::Operation* op);
    void setAliasForSerializedTensors(mlir::Operation* op);

public:
    SpecificTask createUPALayerTask(mlir::Operation* op, const SoftwareLayerParams& params);

    SpecificTask createComputeSWKernelTask(mlir::Operation* op);
    SpecificTask createCacheOpSWKernelTask(mlir::Operation* op);
    SpecificTask createSW_KernelTask(mlir::Operation* op);
    ActKernel createRuntimeKernelTask(mlir::ModuleOp module, mlir::Operation* op);

    //  compiles kernel code and returns it's data and text sections
    ActKernelDesc compileKernelData(const CompilationUnitDesc& unitDesc, const vpux::VPU::ArchKind& archKind,
                                    std::optional<VPU::RevisionID> revisionID, bool hasInputsInDDR);
    ActKernelDesc compileManagementKernelData(const vpux::VPU::ArchKind& archKind,
                                              std::optional<VPU::RevisionID> revisionID);

    KernelDataRef createKernelDataRef(StringRef name, uint64_t dataOffset, uint64_t dataSize,
                                      ArrayRef<uint8_t> content = std::nullopt);
    KernelDataRef createActKernelPerfDataRef(StringRef name, mlir::ShapedType type, VPURT::BufferSection section,
                                             int64_t sectionIndex, int64_t byteOffset);
    KernelDataRef createKernelDataRef(const KernelDataDesc& desc);

    const ActShavesKernelDataMap& getKernelData() const;

public:
    TensorReference createTensorRef(StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section,
                                    ArrayRef<int64_t> sectionIndex, int64_t byteOffset, ArrayRef<int64_t> mult,
                                    ArrayRef<int64_t> shift, int64_t postShift, ArrayRef<uint8_t> zeroPoints,
                                    std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt);
    TensorReference createTensorRef(StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section,
                                    ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                    std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt);
    TensorReference createTensorRef(StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section,
                                    int64_t sectionIndex, int64_t byteOffset,
                                    std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt);
    TensorReference createTensorRef(mlir::Value val, StringRef name, VPURT::BufferSection section,
                                    ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                    std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt);
    TensorReference createTensorRef(mlir::Value val, StringRef name, VPURT::BufferSection section, int64_t sectionIndex,
                                    int64_t byteOffset, std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt);
    TensorReference getTensorRef(mlir::Value val) const;
    const DMADescriptorReference getDepthToSpaceNNDMADescriptorReference(mlir::Operation* op) const;
    const DMADescriptorReference getSpaceToDepthNNDMADescriptorReference(mlir::Operation* op) const;
    const DMADescriptorReference getExpandNNDMADescriptorReference(mlir::Operation* op) const;
    const DMADescriptorReference getPermuteNNDMADescriptorReference(mlir::Operation* op) const;
    const DMADescriptorReference getPerAxisTileNNDMADescriptorReference(mlir::Operation* op) const;
    const DMADescriptorReference getUpsamplingNNDMADescriptorReference(mlir::Operation* op) const;
    const DMADescriptorReference getSyncNNDMADescriptorReference(mlir::Operation* op) const;

public:
    BinaryData createBinaryData(ArrayRef<uint64_t> content, vpux::NDTypeInterface type, bool csram_cacheable = false);
    BinaryData createBinaryData(ArrayRef<uint64_t> content, size_t totalBytes, bool csram_cacheable = false);

public:
    Barrier createBarrier(mlir::Value val, std::optional<int64_t> physicalID = std::nullopt);

    uint32_t getBarrierVirtualID(mlir::Value val) const;
    std::optional<uint32_t> getBarrierPhysicalID(mlir::Value val) const;

    BarrierReference createBarrierReference(mlir::Operation* op);

public:
    Vector<uint32_t> createDims(ShapeRef shape);
    Vector<uint32_t> createDims(vpux::NDTypeInterface type);
    template <typename T>
    Vector<T> createStrides(StridesRef strides, Bit elemSize);
    template <typename T>
    Vector<T> createStrides(vpux::NDTypeInterface type);
    IndirectDataReference createIndirectDataReference(int64_t dataIndex,
                                                      std::optional<int64_t> sparsityIndex = std::nullopt,
                                                      std::optional<int64_t> storageElementIndex = std::nullopt,
                                                      std::optional<int64_t> storageElementSize = std::nullopt);

public:
    auto createString(StringRef str) {
        return _impl.CreateString(str.data(), str.size());
    }

    template <typename T>
    auto createVector(ArrayRef<T> arr) {
        return _impl.CreateVector(arr.data(), arr.size());
    }

    template <class Range>
    auto createVector(const Range& range) {
        const auto vec = to_small_vector(range);
        return _impl.CreateVector(vec.data(), vec.size());
    }

    template <typename T>
    auto createVectorOfStructs(ArrayRef<T> arr) {
        return _impl.CreateVectorOfStructs(arr.data(), arr.size());
    }

public:
    auto& impl() {
        return _impl;
    }

    operator flatbuffers::FlatBufferBuilder&() {
        return impl();
    }

private:
    using TaskMap = std::unordered_map<mlir::Operation*, Task>;
    using TensorReferenceMap = DenseMap<mlir::Value, TensorReference>;
    using BarrierMap = DenseMap<mlir::Value, uint32_t>;

    template <class UnderlyingType>
    auto arrayCast(ArrayRef<int64_t> source) {
        SmallVector<UnderlyingType> casted(source.size());
        std::transform(source.begin(), source.end(), casted.begin(), [](auto value) {
            return checked_cast<UnderlyingType>(value);
        });
        return createVector(casted);
    }

private:
    Logger _log;
    VPU::ArchKind _architecture;
    flatbuffers::FlatBufferBuilder _impl;
    TaskMap _tasks;
    ActShavesKernelDataMap _actKernelsData;
    TensorReferenceMap _tensors;
    BarrierMap _barriersVirtIds;
    BarrierMap _barriersPhysIds;
};

}  // namespace VPUIP
}  // namespace vpux

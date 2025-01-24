//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/utils/schema.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <flatbuffers/flatbuffers.h>

#include <llvm/ADT/MapVector.h>
#include <unordered_map>

namespace vpux {
namespace VPUMI37XX {

using DMADescriptorReference = MVCNN::DMADescriptorReference;

class BlobWriter final {
public:
    using Task = flatbuffers::Offset<MVCNN::Task>;
    using TaskList = flatbuffers::Offset<MVCNN::TaskList>;

    using TensorReference = flatbuffers::Offset<MVCNN::TensorReference>;
    using IndirectDataReference = flatbuffers::Offset<MVCNN::IndirectDataReference>;

    using String = flatbuffers::Offset<flatbuffers::String>;

    template <typename T>
    using Vector = flatbuffers::Offset<flatbuffers::Vector<T>>;

public:
    BlobWriter(Logger log): _log(log) {
    }

public:
    TensorReference createTensorRef(StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section,
                                    ArrayRef<int64_t> sectionIndex, int64_t byteOffset, ArrayRef<int64_t> mult,
                                    ArrayRef<int64_t> shift, int64_t postShift, ArrayRef<uint8_t> zeroPoints,
                                    std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt,
                                    std::optional<uint64_t> descriptor = std::nullopt);
    TensorReference createTensorRef(StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section,
                                    ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                    std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt,
                                    std::optional<uint64_t> descriptor = std::nullopt);
    TensorReference createTensorRef(StringRef name, vpux::NDTypeInterface type, VPURT::BufferSection section,
                                    int64_t sectionIndex, int64_t byteOffset,
                                    std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt,
                                    std::optional<uint64_t> descriptor = std::nullopt);
    TensorReference createTensorRef(mlir::Value val, StringRef name, VPURT::BufferSection section,
                                    ArrayRef<int64_t> sectionIndex, int64_t byteOffset,
                                    std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt,
                                    std::optional<uint64_t> descriptor = std::nullopt);
    TensorReference createTensorRef(mlir::Value val, StringRef name, VPURT::BufferSection section, int64_t sectionIndex,
                                    int64_t byteOffset, std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                    std::optional<int64_t> storageElementOffset = std::nullopt,
                                    std::optional<int64_t> storageElementSize = std::nullopt,
                                    std::optional<int64_t> swizzlingKey = std::nullopt,
                                    std::optional<uint64_t> descriptor = std::nullopt);

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
    flatbuffers::FlatBufferBuilder _impl;
    TaskMap _tasks;
    TensorReferenceMap _tensors;
};

}  // namespace VPUMI37XX
}  // namespace vpux

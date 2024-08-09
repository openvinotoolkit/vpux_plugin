//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/padding_native.hpp"

namespace vpux {
namespace VPU {
class DistributedTensorNative {
private:
    DistributionMode _distributionMode = DistributionMode::NONE;
    SmallVector<int64_t> _numTiles = {};
    SmallVector<int64_t> _kernel = {};
    std::optional<Padding> _pad = std::nullopt;
    SmallVector<int64_t> _strides = {};
    int64_t _numClusters = 0;
    SmallVector<int64_t> _alignment = {};
    bool _uniformDistributedSegments = false;
    SmallVector<SmallVector<int64_t>> _computeShapes = {};
    SmallVector<SmallVector<int64_t>> _computeOffsets = {};
    SmallVector<SmallVector<int64_t>> _memoryShapes = {};
    SmallVector<SmallVector<int64_t>> _memoryOffsets = {};
    bool _equalMemoryAndComputeView = false;

public:
    DistributedTensorNative() = default;
    DistributedTensorNative(const DistributionMode mode, ArrayRef<int64_t> numTiles, ArrayRef<int64_t> kernel,
                            ArrayRef<int64_t> strides, const std::optional<Padding>& padding, int64_t clusters,
                            ArrayRef<int64_t> alignment, const bool hasUniformDistributedSegments,
                            ArrayRef<SmallVector<int64_t>> computeShapes, ArrayRef<SmallVector<int64_t>> computeOffsets,
                            ArrayRef<SmallVector<int64_t>> memoryShapes, ArrayRef<SmallVector<int64_t>> memoryOffsets,
                            const bool hasEqualMemoryAndComputeView) {
        _distributionMode = mode;
        _numTiles = SmallVector<int64_t>(numTiles);
        _kernel = SmallVector<int64_t>(kernel);
        _strides = SmallVector<int64_t>(strides);
        _pad = padding;
        _numClusters = clusters;
        _alignment = SmallVector<int64_t>(alignment);
        _uniformDistributedSegments = hasUniformDistributedSegments;
        for (const auto& v : computeShapes) {
            _computeShapes.push_back(v);
        }

        for (const auto& v : computeOffsets) {
            _computeOffsets.push_back(v);
        }

        for (const auto& v : memoryShapes) {
            _memoryShapes.push_back(v);
        }

        for (const auto& v : memoryOffsets) {
            _memoryOffsets.push_back(v);
        }

        _equalMemoryAndComputeView = hasEqualMemoryAndComputeView;
    }

    ~DistributedTensorNative() = default;
    static DistributedTensorNative getClassFromAttr(DistributedTensorAttr distributionAttr);
    static DistributedTensorAttr getAttrFromClass(mlir::MLIRContext* ctx, const DistributedTensorNative& distribution);

    friend bool operator==(const DistributedTensorNative& lhs, const DistributedTensorNative& rhs) {
        return lhs._distributionMode == rhs._distributionMode && lhs._numTiles == rhs._numTiles &&
               lhs._kernel == rhs._kernel && lhs._strides == rhs._strides && lhs._pad == rhs._pad &&
               lhs._numClusters == rhs._numClusters && lhs._alignment == rhs._alignment &&
               lhs._uniformDistributedSegments == rhs._uniformDistributedSegments &&
               lhs._computeShapes == rhs._computeShapes && lhs._computeOffsets == rhs._computeOffsets &&
               lhs._memoryShapes == rhs._memoryShapes && lhs._memoryOffsets == rhs._memoryOffsets &&
               lhs._equalMemoryAndComputeView == rhs._equalMemoryAndComputeView;
    }

    DistributionMode getDistributionMode() const {
        return _distributionMode;
    }
    void setDistributionMode(const DistributionMode& mode) {
        _distributionMode = mode;
    }

    int64_t getNumClusters() const {
        return _numClusters;
    }
    void setNumClusters(int64_t num) {
        _numClusters = num;
    }

    ArrayRef<int64_t> getNumTiles() const {
        return _numTiles;
    }
    void setNumTiles(ArrayRef<int64_t> numTiles) {
        _numTiles = SmallVector<int64_t>(numTiles);
    }

    ArrayRef<int64_t> getKernel() const {
        return _kernel;
    }
    void setKernel(ArrayRef<int64_t> kernel) {
        _kernel = SmallVector<int64_t>(kernel);
    }

    ArrayRef<int64_t> getStrides() const {
        return _strides;
    }
    void setStrides(ArrayRef<int64_t> strides) {
        _strides = SmallVector<int64_t>(strides);
    }

    ArrayRef<int64_t> getAlignment() const {
        return _alignment;
    }
    void setAlignment(ArrayRef<int64_t> alignment) {
        _alignment = SmallVector<int64_t>(alignment);
    }

    bool hasUniformDistributedSegments() const {
        return _uniformDistributedSegments;
    }
    void setUniformDistributedSegments(const bool uds) {
        _uniformDistributedSegments = uds;
    }

    ArrayRef<SmallVector<int64_t>> getComputeShapes() const {
        return _computeShapes;
    }
    void setComputeShapes(ArrayRef<SmallVector<int64_t>> computeShapes) {
        _computeShapes.clear();
        for (const auto& v : computeShapes) {
            _computeShapes.push_back(SmallVector<int64_t>(v));
        }
    }

    ArrayRef<SmallVector<int64_t>> getComputeOffsets() const {
        return _computeOffsets;
    }
    void setComputeOffsets(ArrayRef<SmallVector<int64_t>> computeOffsets) {
        _computeOffsets.clear();
        for (const auto& v : computeOffsets) {
            _computeOffsets.push_back(SmallVector<int64_t>(v));
        }
    }

    ArrayRef<SmallVector<int64_t>> getMemoryShapes() const {
        return _memoryShapes;
    }
    void setMemoryShapes(ArrayRef<SmallVector<int64_t>> memoryShapes) {
        _memoryShapes.clear();
        for (const auto& v : memoryShapes) {
            _memoryShapes.push_back(SmallVector<int64_t>(v));
        }
    }

    ArrayRef<SmallVector<int64_t>> getMemoryOffsets() const {
        return _memoryOffsets;
    }
    void setMemoryOffsets(ArrayRef<SmallVector<int64_t>> memoryOffsets) {
        _memoryOffsets.clear();
        for (const auto& v : memoryOffsets) {
            _memoryOffsets.push_back(SmallVector<int64_t>(v));
        }
    }

    bool hasEqualMemoryAndComputeView() const {
        return _equalMemoryAndComputeView;
    }
    void setEqualMemoryAndComputeView(const bool emcv) {
        _equalMemoryAndComputeView = emcv;
    }

    std::optional<Padding> getPadding() const {
        return _pad;
    }
    void setPadding(const Padding& padding) {
        _pad = padding;
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "\n#VPU.DistributedTensor<mode = {0}", VPU::stringifyDistributionMode(_distributionMode));
        printTo(stream, ", num_tiles = ");
        ListFormatProvider::format(_numTiles, stream, {});
        printTo(stream, ", kernel = ");
        ListFormatProvider::format(_kernel, stream, {});
        printTo(stream, ", {0}", _pad.has_value() ? _pad : Padding{});
        printTo(stream, ", strides = ");
        ListFormatProvider::format(_strides, stream, {});
        printTo(stream, ", num_clusters = {0}", _numClusters);
        printTo(stream, ", alignment = ");
        ListFormatProvider::format(_alignment, stream, {});
        printTo(stream, ", _uniformDistributedSegments = {0}", _uniformDistributedSegments);
        printTo(stream, ", compute_shapes = [");
        for (const auto& it : _computeShapes) {
            ListFormatProvider::format(it, stream, {});
        }
        printTo(stream, "]");
        printTo(stream, ", compute_offsets = [");
        for (const auto& it : _computeOffsets) {
            ListFormatProvider::format(it, stream, {});
        }
        printTo(stream, "]");
        printTo(stream, ", memory_shapes = [");
        for (const auto& it : _memoryShapes) {
            ListFormatProvider::format(it, stream, {});
        }
        printTo(stream, "]");
        printTo(stream, ", memory_offsets = [");
        for (const auto& it : _memoryOffsets) {
            ListFormatProvider::format(it, stream, {});
        }
        printTo(stream, "]");
        printTo(stream, ", _equalMemoryAndComputeView = {0}>", _equalMemoryAndComputeView);
    }
};

}  // namespace VPU
}  // namespace vpux

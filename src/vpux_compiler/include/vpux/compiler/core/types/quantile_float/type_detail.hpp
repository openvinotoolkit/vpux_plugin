// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/TypeSupport.h>
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace mlir;

namespace vpux {
namespace detail {

/// Quantile float Type Storage and Uniquing.
struct QuantileFloatTypeStorage : public mlir::TypeStorage {
    unsigned width;
    const double* quantilesElements;
    unsigned quantilesParamsSize;

    struct KeyTy {
        KeyTy(unsigned width, ArrayRef<double> quantiles): width(width), quantiles(quantiles) {
        }

        unsigned width;
        ArrayRef<double> quantiles;
        unsigned getWidth() const {
            return width;
        }
        ArrayRef<double> getQuantiles() const {
            return quantiles;
        }

        template <typename T, typename U>
        static bool genericIsEqual(const T& lhs, const U& rhs) {
            return lhs.getWidth() == rhs.getWidth() && lhs.getQuantiles() == rhs.getQuantiles();
        }

        bool operator==(const KeyTy& other) const {
            return genericIsEqual(*this, other);
        }

        unsigned getHashValue() const {
            int64_t* quantilesCast = llvm::bit_cast<int64_t*>(quantiles.data());
            ArrayRef<int64_t> quantilesBits(quantilesCast, quantiles.size());
            return llvm::hash_combine(llvm::hash_combine_range(quantilesBits.begin(), quantilesBits.end()),
                                      static_cast<int64_t>(width));
        }
    };

    bool operator==(const KeyTy& key) const {
        return KeyTy::genericIsEqual(*this, key);
    }

    QuantileFloatTypeStorage(const KeyTy& key, ArrayRef<double> quantiles)
            : width(key.width), quantilesElements(quantiles.data()), quantilesParamsSize(quantiles.size()) {
    }

    static QuantileFloatTypeStorage* construct(TypeStorageAllocator& allocator, KeyTy key) {
        ArrayRef<double> quantiles = allocator.copyInto(key.quantiles);
        return new (allocator.allocate<QuantileFloatTypeStorage>()) QuantileFloatTypeStorage(key, quantiles);
    }

    static unsigned hashKey(const KeyTy& key) {
        return key.getHashValue();
    }

    ArrayRef<double> getQuantiles() const {
        return ArrayRef<double>(quantilesElements, quantilesParamsSize);
    }

    unsigned getWidth() const {
        return width;
    }
};

}  // namespace detail
}  // namespace vpux

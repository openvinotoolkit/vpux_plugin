//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include <memory>
#include "vpux/utils/core/helper_macros.hpp"

namespace vpux {
namespace VPU {

//
// MaxKernelSize
//

struct MaxKernelSizeConstant {
    template <typename T>
    MaxKernelSizeConstant(T t) noexcept: self{std::make_unique<Model<T>>(std::move(t))} {
    }

    int64_t getMaxKernelSize() const;

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual int64_t getMaxKernelSize() const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(T s) noexcept: self{std::move(s)} {
        }
        virtual int64_t getMaxKernelSize() const override {
            return self.getMaxKernelSize();
        }
        T self;
    };

    std::unique_ptr<Concept> self;
};

}  // namespace VPU
}  // namespace vpux

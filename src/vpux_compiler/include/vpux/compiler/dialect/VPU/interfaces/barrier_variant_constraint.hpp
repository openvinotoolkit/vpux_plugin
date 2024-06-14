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
// PerBarrierVariantConstraint
//

struct PerBarrierVariantConstraint {
    template <typename T>
    PerBarrierVariantConstraint(T t) noexcept: self{std::make_unique<Model<T>>(std::move(t))} {
    }

    size_t getPerBarrierMaxVariantSum() const;
    size_t getPerBarrierMaxVariantCount() const;

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual size_t getPerBarrierMaxVariantSum() const = 0;
        virtual size_t getPerBarrierMaxVariantCount() const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(T s) noexcept: self{std::move(s)} {
        }
        virtual size_t getPerBarrierMaxVariantSum() const override {
            return self.getPerBarrierMaxVariantSum();
        }
        virtual size_t getPerBarrierMaxVariantCount() const override {
            return self.getPerBarrierMaxVariantCount();
        }
        T self;
    };

    std::unique_ptr<Concept> self;
};

}  // namespace VPU
}  // namespace vpux

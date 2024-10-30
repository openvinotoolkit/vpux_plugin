//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"

namespace vpux::VPU {
/* @brief
 * Static class for encapsulating PPE-related objects.
 */
class PpeVersionConfig {
private:
    static std::unique_ptr<IPpeFactory>& _getFactory();

public:
    template <typename ConcreteFactoryT>
    static void setFactory() {
        Logger::global().info("Changed PpeFectory instance");
        _getFactory() = std::make_unique<ConcreteFactoryT>();
    }

    static const IPpeFactory& getFactory();

    template <typename DstT, std::enable_if_t<std::is_pointer_v<DstT>, bool> = true>
    static auto getFactoryAs() {
        using ConstDstPtrT = std::add_pointer_t<std::add_const_t<std::remove_pointer_t<DstT>>>;
        return dynamic_cast<const ConstDstPtrT>(&getFactory());
    }

    template <typename DstT, std::enable_if_t<!std::is_pointer_v<DstT>, bool> = true>
    static const DstT& getFactoryAs() {
        const auto* casted = dynamic_cast<const DstT*>(&getFactory());
        VPUX_THROW_WHEN(casted == nullptr, "Failed to cast the default PpeFactory instance to the required type");
        return *casted;
    }

    static PPEAttr retrievePPEAttribute(mlir::Operation* operation) {
        return getFactory().retrievePPEAttribute(operation);
    }
};

}  // namespace vpux::VPU

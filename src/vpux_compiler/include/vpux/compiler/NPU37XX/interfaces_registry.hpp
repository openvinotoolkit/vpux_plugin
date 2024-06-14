//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/interfaces_registry.hpp"

namespace vpux {

//
// IntefacesRegistry37XX
//

class InterfacesRegistry37XX final : public IInterfaceRegistry {
public:
    void registerInterfaces(mlir::DialectRegistry& registry) override;
};

}  // namespace vpux

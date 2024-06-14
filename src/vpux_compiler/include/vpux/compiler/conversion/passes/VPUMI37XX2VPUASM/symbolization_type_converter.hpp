//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpumi37xx2vpuasm {

class SymbolizationTypeConverter : public mlir::TypeConverter {
public:
    SymbolizationTypeConverter();
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux

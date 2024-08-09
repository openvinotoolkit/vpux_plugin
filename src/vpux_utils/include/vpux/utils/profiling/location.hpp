//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

namespace vpux {

// This char is a separator between original layer name provided in xml
// and metadata added by the compiler.
// It is crucial to provide layer names matching the original model in xml.
constexpr char LOCATION_ORIGIN_SEPARATOR = '?';

// Separates location segments
constexpr char LOCATION_SEPARATOR = '/';

}  // namespace vpux

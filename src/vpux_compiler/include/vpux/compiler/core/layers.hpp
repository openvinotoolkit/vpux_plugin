//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"

#include "vpux/utils/core/error.hpp"

namespace vpux {

//
// Dims3D
//

struct Dims3D final {
    // Matmul3d activations

    struct Act final {
        static const Dim B;
        static const Dim H;
        static const Dim IC;

        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 1, "Dims3D::Act: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 1);
        }
    };

    // Matmul3d filter

    struct Filter final {
        static const Dim B;
        static const Dim IC;
        static const Dim OC;

        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 1, "Dims3D::Filter: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 1);
        }
    };

    // Matmul3d output

    struct Output final {
        static const Dim B;
        static const Dim H;
        static const Dim OC;
    };
};

//
// Dims4D
//

struct Dims4D final {
    // Convolution2D/Pooling2D activations

    struct Act final {
        static const Dim N;
        static const Dim C;
        static const Dim H;
        static const Dim W;

        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 2, "Dims4D::Act: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 2);
        }
    };

    // Convolution2D filter

    struct Filter final {
        static const Dim OC;
        static const Dim IC;
        static const Dim KY;
        static const Dim KX;

        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 2, "Dims4D::Filter: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 2);
        }
    };

    // Pooling2D kernel

    struct Kernel final {
        static const Dim Y;
        static const Dim X;
    };

    // Convolution2D/Pooling2D strides

    struct Strides final {
        static const Dim Y;
        static const Dim X;
    };

    // Convolution2D dilations

    struct Dilation final {
        static const Dim Y;
        static const Dim X;
    };

    // Convolution2D/Pooling2D paddings

    struct PadsBegin final {
        static const Dim Top;
        static const Dim Left;
    };
    struct PadsEnd final {
        static const Dim Bottom;
        static const Dim Right;
    };

    // TransposedConvolution2D output paddings

    struct PadsOutput final {
        static const Dim Y;
        static const Dim X;
    };
};

//
// Dims5D
//

struct Dims5D final {
    // Convolution3D/Pooling3D activations

    struct Act final {
        static const Dim N;
        static const Dim C;
        static const Dim D;
        static const Dim H;
        static const Dim W;

        static constexpr size_t numSpatialDims = 3;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 3, "Dims5D::Act: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 2);
        }
    };

    // Convolution3D filter

    struct Filter final {
        static const Dim OC;
        static const Dim IC;
        static const Dim KZ;
        static const Dim KY;
        static const Dim KX;

        static constexpr size_t numSpatialDims = 3;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 3, "Dims5D::Filter: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 2);
        }
    };

    // Pooling3D kernel

    struct Kernel final {
        static const Dim Z;
        static const Dim Y;
        static const Dim X;
    };

    // Convolution3D/Pooling3D strides

    struct Strides final {
        static const Dim Z;
        static const Dim Y;
        static const Dim X;
    };

    // Convolution3D dilations

    struct Dilation final {
        static const Dim Z;
        static const Dim Y;
        static const Dim X;
    };

    // Convolution3D/Pooling3D paddings
    // refer to openvino/src/frontends/paddle/src/op/pad3d.cpp
    struct PadsBegin final {
        static const Dim Front;
        static const Dim Top;
        static const Dim Left;
    };
    struct PadsEnd final {
        static const Dim Back;
        static const Dim Bottom;
        static const Dim Right;
    };

    // TransposedConvolution3D output paddings

    struct PadsOutput final {
        static const Dim Z;
        static const Dim Y;
        static const Dim X;
    };
};

// Layer itself is 2D, but several layers are grouped without sharing weights
struct DimsGroups5D final {
    // Grouped layer activations
    struct Act final {
        static const Dim G;
        static const Dim N;
        static const Dim C;
        static const Dim H;
        static const Dim W;

        static constexpr size_t numDims = 5;
        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 2, "DimsGroups5D::Act: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 3);
        }
    };

    // Grouped layer filter
    struct Filter final {
        static const Dim G;
        static const Dim OC;
        static const Dim IC;
        static const Dim KY;
        static const Dim KX;

        static constexpr size_t numDims = 5;
        static constexpr size_t numSpatialDims = 2;

        static Dim getSpatialDim(size_t index) {
            VPUX_THROW_UNLESS(index < 2, "DimsGroups5D::Filter: Wrong spatial dimension index '{0}'", index);
            return Dim(index + 3);
        }
    };

    // Grouped layer kernel
    struct Kernel final {
        static const Dim Y;
        static const Dim X;
    };

    // Grouped layer strides
    struct Strides final {
        static const Dim Y;
        static const Dim X;
    };

    // Grouped layer dilations
    struct Dilation final {
        static const Dim Y;
        static const Dim X;
    };

    // Grouped layer paddings
    struct PadsBegin final {
        static const Dim Top;
        static const Dim Left;
    };
    struct PadsEnd final {
        static const Dim Bottom;
        static const Dim Right;
    };

    // Grouped layer output paddings
    struct PadsOutput final {
        static const Dim Y;
        static const Dim X;
    };
};

}  // namespace vpux

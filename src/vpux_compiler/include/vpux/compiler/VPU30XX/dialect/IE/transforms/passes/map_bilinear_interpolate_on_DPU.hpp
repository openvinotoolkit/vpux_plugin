//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes/map_bilinear_interpolate_on_DPU.hpp"

namespace vpux {
namespace IE {
namespace arch30xx {

//
// MapBilinearInterpolateOnDPURewriter
//

class MapBilinearInterpolateOnDPURewriter final : public MapBilinearInterpolateOnDPUBaseRewriter {
public:
    MapBilinearInterpolateOnDPURewriter(mlir::MLIRContext* ctx, Logger log)
            : MapBilinearInterpolateOnDPUBaseRewriter(ctx, log) {
        setDebugName("MapBilinearInterpolateOnDPURewriterVPUX30XX");
    }

private:
    mlir::Value createIdentityPooling(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                      mlir::Value input) const override;
};

}  // namespace arch30xx
}  // namespace IE
}  // namespace vpux

//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace IE {

//
// ExpandWithLayer
//

class ExpandWithLayer final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    ExpandWithLayer(mlir::MLIRContext* ctx,
                    const std::function<bool(IE::ExpandOp, mlir::Operation*)>& isBeneficalToSwap, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _isBeneficalToSwap(isBeneficalToSwap), _log(log) {
        setDebugName("ExpandWithLayer");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origExpandOp, mlir::PatternRewriter& rewriter) const final;

private:
    std::function<bool(IE::ExpandOp, mlir::Operation* layerOp)> _isBeneficalToSwap;
    Logger _log;
};

}  // namespace IE
}  // namespace vpux

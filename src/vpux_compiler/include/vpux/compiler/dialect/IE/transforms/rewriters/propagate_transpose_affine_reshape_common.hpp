//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
namespace vpux {
namespace IE {

bool doesAffineReshapeChangeRank(IE::AffineReshapeOp reshape);

SmallVector<int64_t> invertDimMappingWithAxesNotSplitOrMerged(ArrayRef<SmallVector<int64_t>> dimMapping,
                                                              ShapeRef affineInShape, ShapeRef affineOutShape);

bool areModifiedAxesSplitOrMerged(ArrayRef<SmallVector<int64_t>> dimMapping, ShapeRef affineInShape,
                                  ShapeRef affineOutShape, const mlir::DenseSet<int64_t>& modifiedAxes, bool swapOrder,
                                  Logger log);

std::optional<int64_t> getNewSoftmaxAxisAfterSwappingWithAffineReshape(IE::SoftMaxOp softmaxOp,
                                                                       IE::AffineReshapeOp affineReshapeOp,
                                                                       const Logger& log);

//
// MoveTransposeAffineReshapeThroughAdd
//

/* Rewrite the pattern from:

        Transpose
            |
      AffineReshape
            |
           Add
            |
      (QuantizeCast)

    to:
           Add
            |
      (QuantizeCast)
            |
        Transpose
            |
      AffineReshape
*/

class MoveTransposeAffineReshapeThroughAdd final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    MoveTransposeAffineReshapeThroughAdd(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx, benefit), _log(log) {
        this->setDebugName("MoveTransposeAffineReshapeThroughAdd");
    }
    enum InputsMode {
        Asymmetry = 0,
        Symmetry = 1,
        Unsupported = 2,
    };

private:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isBeneficialConversion(IE::AddOp origOp, InputsMode mode) const;
    std::tuple<InputsMode, IE::TransposeOp, IE::AffineReshapeOp> checkAddInputsMode(IE::AddOp origOp) const;

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux

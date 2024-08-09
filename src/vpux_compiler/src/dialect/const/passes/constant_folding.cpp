//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"

using namespace vpux;

namespace {

//
// ConstantFoldingPass
//

class ConstantFoldingPass final : public Const::ConstantFoldingBase<ConstantFoldingPass> {
public:
    explicit ConstantFoldingPass(Logger log): _log(log) {
        _log.setName(getArgumentName());
    }

private:
    void runOnOperation() final;
    Logger _log;
};

void ConstantFoldingPass::runOnOperation() {
    auto origOp = getOperation();

    _log.trace("Folding constant at location '{0}'", origOp.getLoc());

    mlir::OpBuilder builder(origOp);

    const auto content = origOp.getContent();
    const auto contentType = content.getType();
    const auto contentElemType = contentType.getElementType();

    const auto bufSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
    std::vector<char> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(tempBuf.data(), bufSize));

    auto rankedTensorType = contentType.cast<mlir::RankedTensorType>();

    const auto elemTypeBitSize = contentType.getElemTypeSize().count();
    // As of now sub byte types are not supported as DenseElementsAttr storage, I1 is an exception
    const auto isUnsupportedSubByteStorageType = elemTypeBitSize < CHAR_BIT && elemTypeBitSize > 1;
    if (isUnsupportedSubByteStorageType) {
        rankedTensorType = contentType
                                   .changeShapeElemType(Shape({1, 1, 1, checked_cast<int32_t>(bufSize)}),
                                                        getUInt8Type(contentType.getContext()))
                                   .cast<mlir::RankedTensorType>();
    } else if (auto qtype = contentElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        rankedTensorType = contentType.changeElemType(normalizeQuantStorageType(qtype)).cast<mlir::RankedTensorType>();
    }

    const auto denseAttr = mlir::DenseElementsAttr::getFromRawBuffer(rankedTensorType, tempBuf);
    auto origType = origOp.getType().cast<NDTypeInterface>();
    if (isUnsupportedSubByteStorageType) {
        // Temporary fix to enable compilation.
        // Final design to also include a mechanism to FREEZE constants
        // from accepting future transformations due to the fact of packed
        // sub byte values stored, which would require an unpacking and a repacking
        origOp.setContentAttr(Const::ContentAttr::get(denseAttr).changeShapeAndElemType(origType.getShape(),
                                                                                        origType.getElementType()));
    } else {
        origOp.setContentAttr(Const::ContentAttr::get(denseAttr));
    }
}

}  // namespace

//
// createConstantFoldingPass
//

std::unique_ptr<mlir::Pass> vpux::Const::createConstantFoldingPass(Logger log) {
    return std::make_unique<ConstantFoldingPass>(log);
}

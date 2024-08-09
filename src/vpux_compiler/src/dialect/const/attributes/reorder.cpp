//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"

#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// ReorderAttr::verify
//

mlir::LogicalResult vpux::Const::ReorderAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::AffineMapAttr order) {
    if (order == nullptr) {
        return printTo(emitError(), "Got NULL 'order' in 'ReorderAttr'");
    }

    if (!order.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'order' in 'ReorderAttr'");
    }

    return mlir::success();
}

//
// ReorderAttr::print
//

void vpux::Const::ReorderAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getOrder());
    printer << ">";
}

//
// ReorderAttr::parse
//

mlir::Attribute vpux::Const::ReorderAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::AffineMapAttr order;
    if (mlir::failed(parser.parseAttribute(order))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::ReorderAttr>(order);
}

//
// ReorderAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ReorderAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto order = DimsOrder::fromAffineMap(getOrder().getValue());
    VPUX_THROW_UNLESS(order.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", order, input);

    return input.changeDimsOrder(order);
}

bool vpux::Const::ReorderAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

static SmallVector<uint32_t> computeOrder(const DimsOrder inOrder, const DimsOrder outOrder) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    SmallVector<uint32_t> memPerm(inPerm.size());
    for (auto p : outPerm | indexed) {
        memPerm[p.index()] = static_cast<uint32_t>(inOrder.dimPos(p.value()));
    }
    return memPerm;
}

//
// ReorderAttr::transform
//

Const::Content vpux::Const::ReorderAttr::transform(vpux::Const::Content& input) const {
    const auto outType = inferOutputType(input.getType());
    const auto inOrder = input.getType().getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    const auto memPerm = mlir::AffineMap::getPermutationMap(ArrayRef(computeOrder(inOrder, outOrder)), getContext());
    return Const::details::memPermuteTransformation(input, outType, memPerm);
}

//
// ContentAttr::reorder
//

Const::ContentAttr vpux::Const::ContentAttr::reorder(DimsOrder newOrder) const {
    return ContentAttr::addTransformation(
            *this, Const::ReorderAttr::get(mlir::AffineMapAttr::get(newOrder.toAffineMap(getContext())))
                           .cast<Const::TransformAttrInterface>());
}

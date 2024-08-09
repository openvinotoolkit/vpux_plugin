//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/core/attributes/strides.hpp>
#include <vpux/compiler/dialect/VPUMI40XX/attributes.hpp>
#include <vpux/compiler/dialect/VPUMI40XX/dialect.hpp>
#include <vpux/compiler/utils/types.hpp>

using namespace vpux;

DMATransaction VPUMI40XX::NNDMATransactionAttr::getDMATransaction() const {
    DMATransaction dmaTransaction;

    dmaTransaction.inputs.push_back(reduceDimsForDma(getInputType().cast<NDTypeInterface>()));
    dmaTransaction.outputs.push_back(reduceDimsForDma(getOutputType().cast<NDTypeInterface>()));

    return dmaTransaction;
}

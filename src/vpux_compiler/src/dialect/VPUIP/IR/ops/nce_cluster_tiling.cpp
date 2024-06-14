//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/asm.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// Inner info
//

mlir::Operation* vpux::VPUIP::NCEClusterTilingOp::getInnerTaskOp() {
    for (auto& op : getBody().front().getOperations()) {
        if (VPUIP::isPureViewOp(&op)) {
            continue;
        }
        return &op;
    }
    return nullptr;
}

mlir::MutableArrayRef<mlir::BlockArgument> vpux::VPUIP::NCEClusterTilingOp::getInnerInputs() {
    return getBody().getArguments().take_front(getInputs().size());
}

mlir::MutableArrayRef<mlir::BlockArgument> vpux::VPUIP::NCEClusterTilingOp::getInnerOutputs() {
    return getBody().getArguments().slice(getInputs().size(), getOutputs().size());
}

//
// Executor info
//

IndexedSymbolAttr vpux::VPUIP::NCEClusterTilingOp::getExecutor() {
    // For NCEClusterTiling retrieve executer of inner operation
    auto op = mlir::dyn_cast<VPUIP::AsyncLayerOpInterface>(getInnerTaskOp());
    VPUX_THROW_UNLESS(op != nullptr, "Inner operation does not support interface for executors");
    return op.getExecutor();
}

//
// print/parse
//

void vpux::VPUIP::NCEClusterTilingOp::print(mlir::OpAsmPrinter& p) {
    // (%operand as %blockArg: <type>, ...)

    VPUX_THROW_UNLESS(!getBody().empty(), "Cannot serialize operation with empty body.");

    auto* entry = &getBody().front();
    VPUX_THROW_UNLESS(getNumOperands() == entry->getNumArguments(),
                      "Mismatch between the number of operands({0}) and body arguments({1}).", getNumOperands(),
                      entry->getNumArguments());

    unsigned opIdx = 0;

    printGroupOfOperands(p, entry, "inputs", getInputs(), opIdx);
    printGroupOfOperands(p, entry, "outputs", getOutputBuffs(), opIdx);

    p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), {"operandSegmentSizes"});
    p.printOptionalArrowTypeList(getResultTypes());
    p << " ";
    p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

mlir::ParseResult vpux::VPUIP::NCEClusterTilingOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    // Parse operands (%operand as %blockArg : <type>).
    SmallVector<mlir::OpAsmParser::Argument> blockArgs;
    SmallVector<mlir::Type> blockTypes;

    int32_t inCount = 0;
    if (parseGroupOfOperands(parser, result, blockArgs, blockTypes, "inputs", inCount).failed()) {
        return mlir::failure();
    }

    int32_t outCount = 0;
    if (parseGroupOfOperands(parser, result, blockArgs, blockTypes, "outputs", outCount).failed()) {
        return mlir::failure();
    }

    // Add derived `operandSegmentSizes` attribute based on parsed operands.
    auto operandSegmentSizes = parser.getBuilder().getDenseI32ArrayAttr({inCount, outCount});
    result.addAttribute("operandSegmentSizes", operandSegmentSizes);

    // Parse operation attributes.
    mlir::NamedAttrList attrs;
    if (parser.parseOptionalAttrDictWithKeyword(attrs)) {
        return mlir::failure();
    }
    result.addAttributes(attrs);

    // Parse operation results.
    SmallVector<mlir::Type> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
        return mlir::failure();
    }
    result.addTypes(resultTypes);

    // Parse region.
    auto* body = result.addRegion();
    if (parser.parseRegion(*body, blockArgs)) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// build
//

void vpux::VPUIP::NCEClusterTilingOp::build(mlir::OpBuilder& builder, mlir::OperationState& result,
                                            mlir::TypeRange resultTypes, mlir::ValueRange operands,
                                            BodyBuilderFn bodyBuilder) {
    result.addOperands(operands);
    result.addTypes(resultTypes);

    int32_t inCount = static_cast<int32_t>(operands.size()) - static_cast<int32_t>(resultTypes.size());
    int32_t outCount = static_cast<int32_t>(resultTypes.size());

    auto operandSegmentSizes = builder.getDenseI32ArrayAttr({inCount, outCount});
    result.addAttribute("operandSegmentSizes", operandSegmentSizes);

    // Add a body region with block arguments
    auto* bodyRegion = result.addRegion();
    auto& bodyBlock = bodyRegion->emplaceBlock();
    for (auto operand : operands) {
        auto type = operand.getType();
        if (auto distributedType = type.dyn_cast<DistributedBufferType>()) {
            type = distributedType.getCompactType();
        } else if (auto sparseType = type.dyn_cast<SparseBufferType>()) {
            if (auto distDataType = sparseType.getData().dyn_cast<DistributedBufferType>()) {
                mlir::MemRefType dataType = distDataType.getCompactType();
                mlir::MemRefType smType = nullptr;
                if (sparseType.getSparsityMap() != nullptr &&
                    sparseType.getSparsityMap().isa<DistributedBufferType>()) {
                    smType = sparseType.getSparsityMap().cast<DistributedBufferType>().getCompactType();
                }
                mlir::MemRefType seType = nullptr;
                if (sparseType.getStorageElementTable() != nullptr &&
                    sparseType.getStorageElementTable().isa<DistributedBufferType>()) {
                    seType = sparseType.getStorageElementTable().cast<DistributedBufferType>().getCompactType();
                }
                type = SparseBufferType::get(dataType, smType, seType, sparseType.getIsWeights(),
                                             sparseType.getSparsityCompression(), sparseType.getSeAttr());
            }
        }

        bodyBlock.addArgument(type, operand.getLoc());
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);

    VPUX_THROW_UNLESS(bodyBuilder, "Got empty body builder.");
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
}

//
// verify
//

mlir::LogicalResult vpux::VPUIP::NCEClusterTilingOp::verify() {
    const auto op = getOperation();
    auto& opBody = getBody();
    if (!opBody.hasOneBlock()) {
        return errorAt(op->getLoc(), "Operation must have only one block.");
    }

    auto numOperands = op->getNumOperands();
    if (numOperands == 0) {
        return errorAt(op->getLoc(), "Operation must have at least one operand.");
    }

    auto bodyNumArgs = opBody.getNumArguments();
    if (numOperands != bodyNumArgs) {
        return errorAt(op->getLoc(), "Mismatch between the number of operands({0}) and body arguments({1}).",
                       numOperands, bodyNumArgs);
    }

    if (op->getNumResults() == 0) {
        return errorAt(op->getLoc(), "Operation must have at least one result.");
    }

    auto isNceClusterTaskType = mlir::isa<VPUIP::NCEClusterTaskOp>(getInnerTaskOp());
    const auto checkShape = [&](mlir::ValueRange operands) {
        for (auto operand : operands) {
            auto operandType = operand.getType();
            if (auto ndType = operand.getType().dyn_cast<vpux::NDTypeInterface>()) {
                auto rank = ndType.getRank();
                if (rank != 4 && rank != 1 && isNceClusterTaskType) {
                    return errorAt(op->getLoc(), "Only 4D/1D tensors are supported. Got {0}", rank);
                }
                continue;
            }

            VPUX_THROW("Unsupported type of operand: {0}", operandType);
        }

        return mlir::success();
    };

    auto isOperandsValid = checkShape(op->getOperands());
    if (isOperandsValid.failed()) {
        return mlir::failure();
    }

    auto isArgsValid = checkShape(opBody.getArguments());
    if (isArgsValid.failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

size_t vpux::VPUIP::NCEClusterTilingOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto clusterTilingOp = mlir::cast<VPUIP::NCEClusterTilingOp>(this->getOperation());
    auto cycleCostInterface = mlir::dyn_cast<VPUIP::CycleCostInterface>(clusterTilingOp.getInnerTaskOp());
    if (cycleCostInterface == nullptr) {
        return VPU::NO_COST;
    }
    return cycleCostInterface.getOperationCycleCost(costModel);
}

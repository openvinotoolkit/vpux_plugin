//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// RegionBranchOpInterface
//

mlir::OperandRange vpux::VPU::VerticalFusionOp::getEntrySuccessorOperands(mlir::RegionBranchPoint point) {
    mlir::Region* pRegion = point.getRegionOrNull();
    unsigned int index = (pRegion != nullptr) ? pRegion->getRegionNumber() : 0;
    VPUX_THROW_UNLESS(index == 0, "Invalid region index: {0}", index);
    return getOperands();
}

void vpux::VPU::VerticalFusionOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                                      SmallVectorImpl<mlir::RegionSuccessor>& regions) {
    mlir::Region* pRegion = point.getRegionOrNull();

    if (pRegion != nullptr) {
        unsigned int index = pRegion->getRegionNumber();
        VPUX_THROW_UNLESS(index == 0, "Invalid region index: {0}", index);
        regions.push_back(mlir::RegionSuccessor(getResults()));
        return;
    }

    regions.emplace_back(&getOps(), getOps().getArguments());
}

bool vpux::VPU::VerticalFusionOp::areTypesCompatible(mlir::Type, mlir::Type) {
    // TODO #-75680
    return true;
}

//
// Inner info
//

mlir::Operation* vpux::VPU::VerticalFusionOp::getFirstInnerTaskOp() {
    return &getOps().front().getOperations().front();
}

//
// print/parse
//

void vpux::VPU::VerticalFusionOp::print(mlir::OpAsmPrinter& p) {
    // (%operand as %blockArg: <type>, ...)

    VPUX_THROW_WHEN(getOps().empty(), "Cannot serialize operation with empty body.");

    auto* entry = &getOps().front();
    VPUX_THROW_WHEN(getNumOperands() != entry->getNumArguments(),
                    "Mismatch between the number of setOperands({0}) and body arguments({1}).", getNumOperands(),
                    entry->getNumArguments());

    p << " (";
    llvm::interleaveComma(getOperands(), p, [&, n = 0](mlir::Value operand) mutable {
        auto argument = entry->getArgument(n++);
        p << operand << " as " << argument << ": " << argument.getType();
    });
    p << ")";

    p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
    p.printOptionalArrowTypeList(getResultTypes());
    p << " ";
    p.printRegion(getOps(), /*printEntryBlockArgs=*/false);
}

mlir::ParseResult vpux::VPU::VerticalFusionOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    // Parse operands (%operand as %blockArg : <type>).
    SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    SmallVector<mlir::OpAsmParser::Argument> blockArgs;
    SmallVector<mlir::Type> operandRawTypes;
    SmallVector<mlir::Type> blockTypes;

    // Parse a single instance of `%operand as %blockArg : <type>`.
    auto parseOperands = [&]() -> mlir::ParseResult {
        if (parser.parseOperand(operands.emplace_back()) || parser.parseKeyword("as") ||
            parser.parseArgument(blockArgs.emplace_back()) || parser.parseColonType(blockTypes.emplace_back())) {
            return mlir::failure();
        }

        operandRawTypes.emplace_back();
        blockArgs.back().type = blockTypes.back();
        return mlir::success();
    };

    auto argsLoc = parser.getCurrentLocation();
    if (parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::OptionalParen, parseOperands) ||
        parser.resolveOperands(operands, operandRawTypes, argsLoc, result.operands)) {
        return mlir::failure();
    }

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

void vpux::VPU::VerticalFusionOp::build(mlir::OpBuilder& builder, mlir::OperationState& result,
                                        mlir::TypeRange resultTypes, mlir::ValueRange operands,
                                        BodyBuilderFn bodyBuilder, mlir::ArrayAttr tilingInfo) {
    result.addOperands(operands);
    result.addTypes(resultTypes);
    result.addAttribute("tilingStrategy", tilingInfo);

    // Add a body region with block arguments
    auto* bodyRegion = result.addRegion();
    auto& bodyBlock = bodyRegion->emplaceBlock();
    for (auto operand : operands) {
        auto type = operand.getType();
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

mlir::LogicalResult vpux::VPU::VerticalFusionOp::verify() {
    const auto op = getOperation();
    auto& opBody = getOps();
    if (!opBody.hasOneBlock()) {
        return errorAt(op->getLoc(), "Operation must have only one block.");
    }

    auto numOperands = op->getNumOperands();
    if (numOperands == 0) {
        return errorAt(op->getLoc(),
                       "Operation must have at least one operand to satisfy pure no-side-effects semantic.");
    }

    auto bodyNumArgs = opBody.getNumArguments();
    if (numOperands != bodyNumArgs) {
        return errorAt(op->getLoc(), "Mismatch between the number of setOperands({0}) and body arguments({1}).",
                       numOperands, bodyNumArgs);
    }

    if (op->getNumResults() == 0) {
        return errorAt(op->getLoc(), "Operation must have at least one result.");
    }

    auto yieldOps = getOps().getOps<vpux::VPU::YieldOp>();
    const auto numYieldOps = std::distance(yieldOps.begin(), yieldOps.end());
    if (numYieldOps != 1) {
        return errorAt(op->getLoc(), "Operation have to contain one YieldOp, but it has {0}", numYieldOps);
    }

    return mlir::success();
}

namespace {

class RemoveIfEmptyBody final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult RemoveIfEmptyBody::matchAndRewrite(VPU::VerticalFusionOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    if (origOp.getOperands().size() != 1 || origOp.getResults().size() != 1) {
        return mlir::failure();
    }

    auto input = origOp.getOperand(0);
    auto output = origOp.getResult(0);

    if (input.getType() != output.getType()) {
        return mlir::failure();
    }

    auto& bodyBlock = origOp.getOps().front();

    // Check if ops does not have any operation besides YieldOp which is
    // integral part of VerticalFusion
    bool allYields = llvm::all_of(bodyBlock.getOperations(), [](const auto& op) {
        return mlir::isa<VPU::YieldOp>(op);
    });

    if (allYields) {
        return mlir::failure();
    }

    rewriter.replaceOp(origOp, input);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::VerticalFusionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* ctx) {
    patterns.add<RemoveIfEmptyBody>(ctx);
}

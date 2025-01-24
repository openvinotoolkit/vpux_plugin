//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Operation.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;
using namespace VPU;

namespace {

unsigned getIndexOfInput(mlir::Operation* endOp, mlir::Operation* sourceOp) {
    // Get the input index of an endOp which is from the sourceOp
    for (auto i : irange(endOp->getNumOperands())) {
        if (endOp->getOperand(i).getDefiningOp() == sourceOp) {
            return i;
        }
    }
    VPUX_THROW("Operation {0} at {1} is not a child of Operation {0} at {1}", endOp->getName(), endOp->getLoc(),
               sourceOp->getName(), sourceOp->getLoc());
}

bool areDistributedTypesConcatenable(VPU::DistributedTensorType firstType, VPU::DistributedTensorType secondType) {
    if (firstType.getOrder() != secondType.getOrder() || firstType.getMemSpace() != secondType.getMemSpace()) {
        return false;
    }

    // checks modes are compatible, num_clusters is the same and num_tiles is the same, if applicable
    if (mlir::failed(areDistributionAttrsCompatible(firstType, secondType,
                                                    /*allowDifferentPerClusterMemoryView = */ true))) {
        return false;
    }

    const auto areInOutShapesOffsetsCompatible = [&](const SmallVector<Shape>& lhs,
                                                     const SmallVector<Shape>& rhs) -> bool {
        for (const auto& pair : zip(lhs, rhs)) {
            const auto shapesOffsetsLhs = std::get<0>(pair);
            const auto shapesOffsetsRhs = std::get<1>(pair);

            if (shapesOffsetsLhs.size() != shapesOffsetsRhs.size()) {
                return false;
            }

            for (size_t idx = 0; idx < shapesOffsetsLhs.size(); idx++) {
                // If dim is not a concatenation axis, check that per cluster shapes/offsets are the same for
                // the input & output.
                // Since pass only allows CMX Concat on single axis, it can be assumed that concatenation axis
                // is the axis where the dims do not match.
                // When checking consistency for output pattern parts, the two shapes should be equal, therefore
                // full memory view is verified.
                const auto dim = Dim(idx);
                if (firstType.getShape()[dim] == secondType.getShape()[dim]) {
                    if (shapesOffsetsLhs[dim] != shapesOffsetsRhs[dim]) {
                        return false;
                    }
                }
            }
        }

        return true;
    };

    const auto firstPerClusterOffsets = firstType.getPerClusterMemoryShapeOffsets();
    const auto secondPerClusterOffsets = secondType.getPerClusterMemoryShapeOffsets();
    if (!areInOutShapesOffsetsCompatible(firstPerClusterOffsets, secondPerClusterOffsets)) {
        return false;
    }

    const auto firstPerClusterShapes = firstType.getPerClusterMemoryShapes();
    const auto secondPerClusterShapes = secondType.getPerClusterMemoryShapes();
    if (!areInOutShapesOffsetsCompatible(firstPerClusterShapes, secondPerClusterShapes)) {
        return false;
    }

    return true;
}

NDTypeInterface getConcatDistributedType(VPU::DistributedTypeInterface origType, ShapeRef shape,
                                         mlir::Type elementType) {
    auto distributedDataType = origType.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
    const auto typeComponents = TypeComponents().setShape(shape).setElementType(elementType);

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedDataType.getDistribution())) {
        auto distribution = distributedDataType.getDistribution();
        if (auto sparseType = origType.dyn_cast<VPU::SparseTensorType>()) {
            distribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
        }

        auto newDistributedAttr =
                getConcatExplicitDistributedAttrForNewShape(distribution, shape, origType.getContext());
        return origType.changeTypeComponentsForExplicitDistribution(typeComponents, newDistributedAttr)
                .cast<NDTypeInterface>();
    }

    return origType.cast<NDTypeInterface>().changeTypeComponents(typeComponents);
}

size_t getSize(vpux::NDTypeInterface type) {
    return static_cast<size_t>(type.getTotalAllocSize().count());
}

//
// NceBasedPart
//
// Base class to store one of a Concat's inputs
// containing NCE->Copy chain
//

struct NceBasedPart {
    VPU::CopyOp copyOp;
    VPU::NCEOpInterface nceOp;

    NceBasedPart(VPU::CopyOp copy, VPU::NCEOpInterface nce): copyOp(copy), nceOp(nce) {
    }

    virtual ~NceBasedPart() = default;

    virtual bool isMultiCluster() const {
        auto isDistributedType = [](mlir::Value val) {
            auto distributedIf = mlir::dyn_cast<VPU::DistributedTypeInterface>(val.getType());
            return ((distributedIf != nullptr) && (distributedIf.containsDistributedTypes()));
        };

        if (copyOp != nullptr) {
            return (isDistributedType(copyOp->getOperand(0)) || isDistributedType(copyOp->getResult(0)));
        }
        if (nceOp != nullptr) {
            if (llvm::any_of(nceOp->getOperands(), isDistributedType)) {
                return true;
            }
            if (llvm::any_of(nceOp->getResults(), isDistributedType)) {
                return true;
            }
        }
        return false;
    }

    virtual mlir::Operation* getCopyOp() {
        return copyOp.getOperation();
    }

    virtual mlir::Operation* getNceOp() {
        return nceOp.getOperation();
    }
};

//
// InputConcatPart
//
// Class to store one of a Concat's inputs that could be:
// - Output from NCE
// - Block argument
//

struct InputConcatPart final : public NceBasedPart {
    mlir::Value concatOperand;
    bool isBlockArgOrConsant;

    bool insertNCEOps = false;

    InputConcatPart(mlir::Value operand)
            : NceBasedPart(nullptr, nullptr), concatOperand(operand), isBlockArgOrConsant(true) {
    }

    InputConcatPart(mlir::Value operand, VPU::CopyOp copy, VPU::NCEOpInterface nce)
            : NceBasedPart(copy, nce), concatOperand(operand), isBlockArgOrConsant(false) {
        if (copy == nullptr && nce == nullptr) {
            isBlockArgOrConsant = true;
        }
    }

    mlir::Operation* getCopyOp() override {
        if (isBlockArgOrConsant) {
            return nullptr;
        }

        return NceBasedPart::getCopyOp();
    }

    mlir::Operation* getNceOp() override {
        if (isBlockArgOrConsant) {
            return nullptr;
        }

        return NceBasedPart::getNceOp();
    }
};

class InputConcatPattern {
public:
    InputConcatPattern(VPU::ConcatOp concat, ArrayRef<InputConcatPart> inputParts, Logger log)
            : _concat(concat), _inputParts(inputParts.begin(), inputParts.end()), _log(log) {
        VPUX_THROW_WHEN(_inputParts.empty(), "Pattern have to have inputs");
    }

    ArrayRef<InputConcatPart> getInputParts() const;

    void rewrite();
    bool inputPatternCanBeCMXed(size_t cmxSize);
    bool inputConcatOnlyMeetRequirement();
    bool isInputConcatOnly = false;

private:
    VPU::ConcatOp _concat;
    SmallVector<InputConcatPart> _inputParts;
    Logger _log;

private:
    size_t getConcatSize();
    bool concatFitsInCMX(size_t cmxSize);
    bool inputsHaveNotOnlyCopiesUsers();
    bool insertNCEOperation();
    bool isMemConsistentPerCluster();
    bool areDistributionTypesConsistent();
    bool areAnyInputsInPlace();
    void insertCopyAfterConcat();
};

ArrayRef<InputConcatPart> InputConcatPattern::getInputParts() const {
    return _inputParts;
}

bool InputConcatPattern::inputConcatOnlyMeetRequirement() {
    const auto outShape = getShape(_concat.getOutput());
    for (const auto& dim : outShape) {
        if (dim > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
            return false;
        }
    }

    const auto concatType = _concat.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto dimsOrder = concatType.getDimsOrder();
    const auto outMemShape = dimsOrder.toMemoryOrder(outShape);
    // currently concat Dim limited to highest dim, we can extend it to other dim but some performance
    // regression need to solve, see E#130441
    for (auto& concatPart : _inputParts) {
        if (!concatPart.concatOperand.hasOneUse()) {
            return false;
        }

        const auto inShape = getShape(concatPart.concatOperand);
        const auto inMemShape = dimsOrder.toMemoryOrder(inShape);
        if (outMemShape[MemDim(0)] == inMemShape[MemDim(0)]) {
            return false;
        }
    }
    return true;
}

// Insert copy after concat, from CMX to DDR
void InputConcatPattern::insertCopyAfterConcat() {
    mlir::OpBuilder builder(_concat);

    builder.setInsertionPointAfter(_concat);
    const auto concatType = _concat.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto memSpace = concatType.getMemSpace();

    auto multiclusterIt = llvm::find_if(_inputParts, [](const InputConcatPart& inPart) {
        return inPart.isMultiCluster();
    });

    if (multiclusterIt == _inputParts.end()) {
        const auto memSpaceCMX = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(MemoryKind::CMX_NN), 0);
        auto newConcatType = concatType.changeMemSpace(memSpaceCMX);
        _concat.getOutput().setType(newConcatType);
        auto copyOp = builder.create<VPU::CopyOp>(_concat.getLoc(), _concat->getResult(0), memSpace);
        _concat->getResult(0).replaceAllUsesExcept(copyOp->getResult(0),
                                                   llvm::SmallPtrSet<mlir::Operation*, 1>{copyOp});
    } else {
        auto multiclusterPart = *multiclusterIt;
        const auto nceDistributedType =
                mlir::cast<VPU::DistributedTypeInterface>(multiclusterPart.nceOp->getResult(0).getType());
        auto newConcatOutputType =
                getConcatDistributedType(nceDistributedType, concatType.getShape(), concatType.getElementType());
        _concat.getOutput().setType(newConcatOutputType);

        const auto tilingCopyOp =
                builder.create<VPU::CopyOp>(_concat.getLoc(), concatType, _concat->getResult(0), memSpace);
        _concat->getResult(0).replaceAllUsesExcept(tilingCopyOp->getResult(0),
                                                   llvm::SmallPtrSet<mlir::Operation*, 1>{tilingCopyOp});
    }

    _log.trace("Inser copy after '{0}' at '{1}'", _concat->getName(), _concat->getLoc());
}

void InputConcatPattern::rewrite() {
    // Insert NCE operations before rewriting if any input has been marked with insertNCEOps = true
    insertNCEOperation();

    /*
        From DDR IR

        NCE      NCE     Const
         |        |        |
        Copy     Copy      |
           \      |       /
                Concat

        TO NNCMX IR

        NCE               NCE                       Const
         |                 |                          |
      SubView           SubView (added in VPUIP)  Copy(DDR->CMX)
         |                 |                          |
 (DistributedCast)  (DistributedCast)                 |
          \                 \                         /
                          Concat
    */

    // copy concat data out to DDR if it is input CMX concat only
    if (isInputConcatOnly) {
        insertCopyAfterConcat();
    }

    auto multiclusterIt = llvm::find_if(_inputParts, [](const InputConcatPart& inPart) {
        return inPart.isMultiCluster();
    });

    auto nceIt = llvm::find_if(_inputParts, [](const InputConcatPart& inPart) {
        return !inPart.isBlockArgOrConsant;
    });

    for (auto p : _inputParts | indexed) {
        const auto& operandIdx = p.index();
        auto& inputPart = p.value();
        auto origType = inputPart.concatOperand.getType().cast<NDTypeInterface>();
        auto inputCopyOp = _concat.getOperand(operandIdx).getDefiningOp<VPU::CopyOp>();

        if (inputPart.isMultiCluster()) {
            // If the inputPattern has MC layer, then the outputPattern must have MC layer so the concat could be
            // CMX-ed. Infer concat input type

            _log.trace("Removing clustered input Copy from NNCMX to DDR '{0}' at '{1}'",
                       inputPart.getCopyOp()->getName(), inputPart.getCopyOp()->getLoc());

            auto concatOutputType = _concat.getOutput().getType().cast<VPU::DistributedTypeInterface>();
            auto newConcatInputType =
                    getConcatDistributedType(concatOutputType, origType.getShape(), origType.getElementType());

            auto concatInPatternMode = newConcatInputType.cast<VPU::DistributedTypeInterface>()
                                               .getDistributedTypes()
                                               .front()
                                               .cast<VPU::DistributedTensorType>()
                                               .getDistribution()
                                               .getMode()
                                               .getValue();

            auto nceProducerOutput = inputCopyOp->getOperand(0);
            auto nceDistributedType = nceProducerOutput.getType().cast<VPU::DistributedTypeInterface>();

            auto copyPatternType =
                    getConcatDistributedType(nceDistributedType, origType.getShape(), origType.getElementType());

            auto copyPatternMode = copyPatternType.cast<VPU::DistributedTypeInterface>()
                                           .getDistributedTypes()
                                           .front()
                                           .cast<VPU::DistributedTensorType>()
                                           .getDistribution()
                                           .getMode()
                                           .getValue();

            if (concatInPatternMode != copyPatternMode) {
                _log.trace("Inserting DistributedCast at '{0}'", inputPart.getCopyOp()->getLoc());
                mlir::OpBuilder builder(_concat);
                auto distributedCastOp = builder.create<VPU::DistributedCastOp>(_concat->getLoc(), newConcatInputType,
                                                                                nceProducerOutput);
                _concat.setOperand(operandIdx, distributedCastOp.getOutput());
            } else {
                _concat.setOperand(operandIdx, nceProducerOutput);
            }

            continue;
        }

        // modify only current concat input as it may have multiple uses
        if (!inputPart.isBlockArgOrConsant) {
            _log.trace("Removing input Copy from NNCMX to DDR '{0}' at '{1}'", inputPart.getCopyOp()->getName(),
                       inputPart.getCopyOp()->getLoc());

            auto newConcatInput = inputCopyOp->getOperand(0);
            _concat.setOperand(operandIdx, newConcatInput);
            continue;
        }

        mlir::OpBuilder builder(_concat);
        builder.setInsertionPointAfterValue(inputPart.concatOperand);

        VPUX_THROW_WHEN(nceIt == _inputParts.end(), "Failed to get memory space");
        auto ncePart = *nceIt;
        const auto memSpace = ncePart.getNceOp()->getOperand(0).getType().cast<NDTypeInterface>().getMemSpace();

        if (multiclusterIt == _inputParts.end()) {
            _log.trace("Insert Copy from DDR to CMX for constant input '{0}'", inputPart.concatOperand);
            const auto newConcatInput =
                    builder.create<VPU::CopyOp>(_concat.getLoc(), inputPart.concatOperand, memSpace).getOutput();
            _concat.setOperand(operandIdx, newConcatInput);
            continue;
        }

        auto multiclusterPart = *multiclusterIt;

        auto distributedConcatOutType = _concat.getOutput().getType().cast<VPU::DistributedTypeInterface>();
        auto newType =
                getConcatDistributedType(distributedConcatOutType, origType.getShape(), origType.getElementType());

        _log.trace("Insert Cluster tiling Copy from DDR to CMX for constant input '{0}'", inputPart.concatOperand);
        const auto newConcatInputOp =
                builder.create<VPU::CopyOp>(_concat.getLoc(), newType, inputPart.concatOperand, memSpace);

        _concat.setOperand(operandIdx, newConcatInputOp->getResult(0));
    }
}

size_t InputConcatPattern::getConcatSize() {
    return getSize(_concat.getOutput().getType());
}

bool InputConcatPattern::concatFitsInCMX(size_t cmxSize) {
    // check if the concat can fit in CMX
    // in order to CMX a concat the entire output buffer + inputs for the
    // largest tile must fit in CMX at the same time
    size_t concatSize = getConcatSize();
    size_t maxUserSize = 0;
    size_t currUserSize;
    // from all users find the one with the largest size
    for (auto concatPart : _inputParts) {
        currUserSize = 0;
        // consts (weights table and activation window) already exists
        auto nceOp = concatPart.getNceOp();
        if (nceOp != nullptr) {
            llvm::DenseSet<mlir::Value> operands(nceOp->getOperands().begin(), nceOp->getOperands().end());
            for (auto input : operands) {
                currUserSize += getSize(input.getType());
            }
        }
        maxUserSize = std::max<size_t>(maxUserSize, currUserSize);
    }

    _log.trace("Concat size '{0}'", (concatSize + maxUserSize));
    // return concat size smaller than CMX size
    return (concatSize + maxUserSize) <= cmxSize;
}

bool isSupportedAndBeneficialToInsertNCEOp(InputConcatPart concatPart) {
    if (!concatPart.isMultiCluster()) {
        return true;
    }

    auto copyOutOp = concatPart.getCopyOp();
    if (copyOutOp == nullptr) {
        return false;
    }

    auto nceDistributedType = copyOutOp->getOperand(0).getType().cast<VPU::DistributedTypeInterface>();

    auto nceDistributionMode = nceDistributedType.getDistributedTypes()
                                       .front()
                                       .cast<VPU::DistributedTensorType>()
                                       .getDistribution()
                                       .getMode()
                                       .getValue();

    // It's not beneficial to insert an Average Pooling with DUPLICATED distribution
    return !(VPU::bitEnumContainsAny(nceDistributionMode, VPU::DistributionMode::DUPLICATED) ||
             VPU::bitEnumContainsAny(nceDistributionMode, VPU::DistributionMode::MULTICASTED));
}

bool InputConcatPattern::inputsHaveNotOnlyCopiesUsers() {
    // avoid concats which are complex, where the inputs to the concat are used
    // by other operations

    for (auto& concatPart : _inputParts) {
        auto nceOp = concatPart.getNceOp();
        if (nceOp == nullptr) {
            continue;
        }

        for (auto result : nceOp->getResults()) {
            for (auto user : result.getUsers()) {
                if (!mlir::isa<VPU::CopyOp>(user)) {
                    if (!isSupportedAndBeneficialToInsertNCEOp(concatPart)) {
                        return true;
                    }
                    concatPart.insertNCEOps = true;
                }
            }
        }
    }

    return false;
}

bool InputConcatPattern::insertNCEOperation() {
    mlir::OpBuilder builder(_concat);

    for (auto p : _inputParts | indexed) {
        const auto& operandIdx = p.index();
        auto& concatPart = p.value();

        if (!concatPart.insertNCEOps) {
            continue;
        }

        auto nceOp = concatPart.getNceOp();
        VPUX_THROW_WHEN(nceOp == nullptr, "Can't find NCE operation");

        auto copyOutOp = concatPart.getCopyOp();
        VPUX_THROW_WHEN(copyOutOp == nullptr, "Can't find output copy operation");

        const auto createPoolFunc = [nceOp](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) -> VPU::NCEAveragePoolOp {
            auto ctx = builder.getContext();
            const SmallVector<int64_t> neutralKernelStrides = {1, 1};
            const SmallVector<int64_t> pads = {0, 0};

            const auto kernelSizeAttr = getIntArrayAttr(ctx, neutralKernelStrides);
            const auto stridesAttr = getIntArrayAttr(ctx, neutralKernelStrides);
            const auto padBeginAttr = getIntArrayAttr(ctx, pads);
            const auto padEndAttr = getIntArrayAttr(ctx, pads);

            auto newOperandType = newOperands[0].getType();

            auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(nceOp);

            // Update Clamp in PPE to align with the original NCE task
            auto nceOpIface = mlir::dyn_cast<VPU::NCEOpInterface>(nceOp);
            VPUX_THROW_WHEN(nceOpIface == nullptr, "Can't find NCEOpInterface");

            const auto origPPETask = nceOpIface.getPPE();
            const auto& clampAdapter = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterClamp>();
            ppeAttr = clampAdapter.updateClamps(ppeAttr, origPPETask);
            const auto elemType = newOperandType.cast<vpux::NDTypeInterface>().getElementType();
            if (mlir::isa<mlir::quant::QuantizedType>(elemType)) {
                if (const auto quantParamsAdapter =
                            VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterQuantParams*>()) {
                    const auto kernelShape = parseIntArrayAttr<int64_t>(kernelSizeAttr);
                    ppeAttr = quantParamsAdapter->recomputeQuantParams(ppeAttr, elemType, elemType, kernelShape);
                }
            }
            if (const auto scaleAdapter = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterScale*>()) {
                ppeAttr = scaleAdapter->updateScale(ppeAttr, {1.0});
            }

            const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(padBeginAttr, padEndAttr));

            auto outputChannelsAttr = nceOp->hasAttr(VPU::outChanAttrName)
                                              ? nceOp->getAttr(VPU::outChanAttrName).cast<mlir::IntegerAttr>()
                                              : nullptr;

            return builder.create<VPU::NCEAveragePoolOp>(loc, newOperandType, newOperands[0], kernelSizeAttr,
                                                         stridesAttr, padAttr, ppeAttr,
                                                         /*multi_cluster_strategyAttr=*/nullptr, outputChannelsAttr);
        };

        mlir::Operation* newAvgPoolOp = nullptr;
        builder.setInsertionPointAfter(nceOp);

        mlir::Operation* newCopyOp = nullptr;
        const auto copyOutOpType = mlir::cast<vpux::NDTypeInterface>(copyOutOp->getResult(0).getType());
        const auto copyMemSpace = copyOutOpType.getMemSpace();

        newAvgPoolOp = createPoolFunc(builder, appendLoc(nceOp->getLoc(), "inserted_AVG_Pooling"), nceOp->getResult(0));

        if (!copyOutOp->hasOneUse()) {
            newCopyOp = builder.create<VPU::CopyOp>(copyOutOp->getLoc(), newAvgPoolOp->getResult(0), copyMemSpace);
        }

        if (!copyOutOp->hasOneUse()) {
            _concat.setOperand(operandIdx, newCopyOp->getResult(0));
        } else {
            copyOutOp->setOperand(0, newAvgPoolOp->getResult(0));
        }
        _log.trace("Inserted an identity AvgPooling {0}", *newAvgPoolOp);
    }

    return true;
}

bool InputConcatPattern::isMemConsistentPerCluster() {
    // CMX Concat is not supported when the memory is inconsistent for each single cluster
    // i.e., when distribution modes are SEGMENTED or OVERLAPPED and concatenation over H

    auto hasMultiCluster = llvm::any_of(_inputParts, [](InputConcatPart concatPart) {
        return concatPart.isMultiCluster();
    });

    if (!hasMultiCluster) {
        return true;
    }

    auto isOffsetOnH = [](mlir::ArrayAttr offset) {
        auto offsetVector = Shape(parseIntArrayAttr<int64_t>(offset));
        return offsetVector[Dims4D::Act::H] != 0;
    };

    auto isSingleOpSplitOnH = [](InputConcatPart concatPart) {
        if (!concatPart.isMultiCluster()) {
            return false;
        }
        const auto disType = concatPart.nceOp->getResult(0)
                                     .getType()
                                     .cast<VPU::DistributedTypeInterface>()
                                     .getDistributedTypes()
                                     .front()
                                     .cast<VPU::DistributedTensorType>();
        const auto disMode = disType.getDistribution().getMode().getValue();
        return disMode == VPU::DistributionMode::SEGMENTED || disMode == VPU::DistributionMode::OVERLAPPED;
    };

    bool isSplitOverH = llvm::any_of(_inputParts, isSingleOpSplitOnH);

    bool isConcatOverH = false;
    if (_concat.getStaticOffsetsAttr() != nullptr) {
        const auto concatDims = _concat.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();
        isConcatOverH = llvm::any_of(concatDims, isOffsetOnH);
    } else {
        const auto concatAxis = _concat.getPerAxis().value().getAxis().getValue().getSExtValue();
        isConcatOverH = concatAxis == Dims4D::Act::H.ind();
    }

    return !(isConcatOverH && isSplitOverH);
}

bool InputConcatPattern::areDistributionTypesConsistent() {
    const auto& it = llvm::find_if(_inputParts, [](InputConcatPart& inPart) {
        return inPart.isMultiCluster();
    });

    if (it == _inputParts.end()) {
        return true;
    }

    auto& multiclusterPart = *it;

    const auto distributedTypeInterfaceOutput =
            multiclusterPart.nceOp->getResult(0).getType().cast<VPU::DistributedTypeInterface>();
    if (!distributedTypeInterfaceOutput.containsDistributedTypes()) {
        return false;
    }
    const auto firstDistrType =
            distributedTypeInterfaceOutput.getDistributedTypes().front().cast<VPU::DistributedTensorType>();

    for (auto& part : _inputParts) {
        if (part.isBlockArgOrConsant) {
            continue;
        }

        if (!part.isMultiCluster()) {
            _log.trace("Can't concatenate distribution tensor with ranked tensor: `{0}` and `{1}`",
                       multiclusterPart.concatOperand, part.concatOperand);
            return false;
        }

        const auto distributedTypeInterfaceInput =
                part.nceOp->getResult(0).getType().cast<VPU::DistributedTypeInterface>();
        if (!distributedTypeInterfaceInput.containsDistributedTypes()) {
            return false;
        }
        const auto curType =
                distributedTypeInterfaceInput.getDistributedTypes().front().cast<VPU::DistributedTensorType>();

        if (!areDistributedTypesConcatenable(firstDistrType, curType)) {
            _log.trace("Not matching distributed tensor attributes between concat inputs: `{0}` and `{1}`",
                       firstDistrType, curType);
            return false;
        }
    }

    return true;
}

bool InputConcatPattern::areAnyInputsInPlace() {
    for (auto& inputPart : _inputParts) {
        auto nceEltwiseOp = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(inputPart.getNceOp());
        if (nceEltwiseOp == nullptr) {
            continue;
        }
        if (nceEltwiseOp.getIsInplace()) {
            return true;
        }
    }
    return false;
}

bool InputConcatPattern::inputPatternCanBeCMXed(size_t cmxSize) {
    // Check compatibility between input distribution types
    if (!areDistributionTypesConsistent()) {
        _log.trace("Distribution types are inconsistent");
        return false;
    }

    // Check if the memory is consistent per cluster
    if (!isMemConsistentPerCluster()) {
        _log.trace("Memory is inconsistent on each cluster");
        return false;
    }

    // assert that the concat will fit in CMX
    if (!concatFitsInCMX(cmxSize)) {
        _log.trace("Concat does not fit in cmx");
        return false;
    }

    if (inputsHaveNotOnlyCopiesUsers()) {
        _log.trace("Concat is complex");
        return false;
    }

    // TODO E#99223: remove limitation for in-place Eltwise
    if (areAnyInputsInPlace()) {
        _log.trace("Input is in-place, which is not yet supported");
        return false;
    }

    return true;
}

//
// OutputPattern
//

struct OutputConcatPart final : public NceBasedPart {
    VPU::SliceOp sliceOp;
    unsigned nceInput;

    OutputConcatPart(VPU::CopyOp copy, VPU::SliceOp slice, VPU::NCEOpInterface nce, unsigned idx)
            : NceBasedPart(copy, nce), sliceOp(slice), nceInput(idx) {
    }

    bool hasSliceOp() const {
        return sliceOp != nullptr;
    }
};

class OutputConcatPattern {
public:
    OutputConcatPattern(VPU::ConcatOp concat, ArrayRef<OutputConcatPart> outputParts, Logger log)
            : _concat(concat), _outputParts(outputParts.begin(), outputParts.end()), _log(log) {
        VPUX_THROW_WHEN(_outputParts.empty(), "Pattern have to have outputs");
    }

    ArrayRef<OutputConcatPart> getOutputParts() const;

    void rewrite();
    bool outputPatternCanBeCMXed(size_t cmxSize);

private:
    VPU::ConcatOp _concat;
    SmallVector<OutputConcatPart> _outputParts;
    Logger _log;

private:
    size_t getConcatSize();
    bool childOpsFitInCMX(size_t cmxSize);
    bool areConcatPartsTypesConsistent(ArrayRef<OutputConcatPart> parts) const;
    void moveConcatToCMX();
    void replaceSliceCopy();
};

ArrayRef<OutputConcatPart> OutputConcatPattern::getOutputParts() const {
    return _outputParts;
}

void OutputConcatPattern::rewrite() {
    /*
                            From DDR IR
         VPU.Concat
           /    \
     VPU.Slice   VPU.Slice                         VPU.Concat
         |        |                                  |
        Copy     Copy                               Copy
         |        |                                  |
        NCE      NCE                                NCE
                            TO NNCMX IR
         VPU.Concat                              VPU.Concat
           /    \                                    |
     VPU.Slice VPU.Slice                            NCE
         |        |
        NCE      NCE
    */
    moveConcatToCMX();
    replaceSliceCopy();
}

size_t OutputConcatPattern::getConcatSize() {
    return getSize(_concat.getOutput().getType());
}

bool OutputConcatPattern::areConcatPartsTypesConsistent(ArrayRef<OutputConcatPart> parts) const {
    // Check if all branches are of the same type
    // Either all or none should be in multi cluster mode
    size_t nceClusterTilingParts = 0;
    for (auto concatPart : parts) {
        if (concatPart.isMultiCluster()) {
            nceClusterTilingParts++;
        }
    }

    if (nceClusterTilingParts > 0 && nceClusterTilingParts != parts.size()) {
        return false;
    }

    return true;
}

bool OutputConcatPattern::childOpsFitInCMX(size_t cmxSize) {
    // check if the child operations - operations using the concat output buffer
    // will fit in CMX along with their inputs and output
    size_t concatSize = getConcatSize();
    size_t parallelConsumerCount = _outputParts.size();
    size_t maxConsumerSize = 0;
    for (auto& concatPart : _outputParts) {
        size_t consumerInputSize = 0;
        size_t consumerOutputSize = 0;
        // consts (weights table and activation window) already exists
        auto nceOp = concatPart.getNceOp();
        auto copyOp = concatPart.getCopyOp();
        auto sliceOp = copyOp->getOperand(0).getDefiningOp<VPU::SliceOp>();
        for (auto input : nceOp->getOperands()) {
            if (input.getDefiningOp() == copyOp && sliceOp == nullptr) {
                continue;
            }
            consumerInputSize += getSize(input.getType());
        }
        for (auto output : nceOp->getResults()) {
            consumerOutputSize += getSize(output.getType());
        }
        maxConsumerSize = std::max<size_t>(maxConsumerSize, consumerInputSize + consumerOutputSize);
    }

    if (parallelConsumerCount > 1) {
        // in cases of parallel consumers the graph level differences could be large and the
        // NNCMX buffer could be held for many cycles filling up NNCMX space. To avoid this
        // scenario, ensure that there is space for parallel consumer branches.
        // Note: multiplying by 2 since only 1 compute operation can be live at any given time,
        // during second consumer execution first will be freed and the third can be allocated.
        maxConsumerSize = 2 * maxConsumerSize;
    }

    _log.trace("Concat consumer max size '{0}'", (maxConsumerSize + concatSize));
    // return concat size greater than CMX size
    return (maxConsumerSize + concatSize) <= cmxSize;
}

bool OutputConcatPattern::outputPatternCanBeCMXed(size_t cmxSize) {
    // verify the following operation can fit in CMX
    if (!childOpsFitInCMX(cmxSize)) {
        _log.trace("Concat consumers do not fit in cmx");
        return false;
    }

    // Check if all output branches are of the same type
    if (!areConcatPartsTypesConsistent(_outputParts)) {
        _log.trace("Concat contains both single and multi cluster outputs");
        return false;
    }

    // Check compatibility between output distribution types
    if (!_outputParts[0].isMultiCluster()) {
        return true;
    }

    const auto distributedTypeInterfaceInput =
            _outputParts[0].nceOp->getOperand(_outputParts[0].nceInput).getType().cast<VPU::DistributedTypeInterface>();
    if (!distributedTypeInterfaceInput.containsDistributedTypes()) {
        return true;
    }
    auto inTypeDistributed =
            distributedTypeInterfaceInput.getDistributedTypes().front().cast<VPU::DistributedTensorType>();

    if (inTypeDistributed != nullptr) {
        for (auto concatPart : ArrayRef(_outputParts).drop_front()) {
            const auto distributedTypeInterfaceOutput =
                    concatPart.nceOp->getOperand(concatPart.nceInput).getType().cast<VPU::DistributedTypeInterface>();
            if (!distributedTypeInterfaceOutput.containsDistributedTypes()) {
                return false;
            }
            const auto curType =
                    distributedTypeInterfaceOutput.getDistributedTypes().front().cast<VPU::DistributedTensorType>();

            if (!areDistributedTypesConcatenable(inTypeDistributed, curType)) {
                _log.trace("Not matching distributed tensor attributes between concat outputs: `{0}` and `{1}`",
                           inTypeDistributed, curType);
                return false;
            }
        }
    }

    return true;
}

// Moves the concat to NNCMX, inserts DistributedCast if needed and returns the new output type
// of the Concat or DistributedCast
void OutputConcatPattern::moveConcatToCMX() {
    mlir::OpBuilder builder(_concat);
    builder.setInsertionPointAfter(_concat);
    const auto concatType = _concat.getOutput().getType().cast<vpux::NDTypeInterface>();
    _log.trace("Moving output to NNCMX for '{0}' at '{1}'", _concat->getName(), _concat->getLoc());
    if (!_outputParts[0].isMultiCluster()) {
        const auto memSpaceCMX = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(MemoryKind::CMX_NN), 0);
        auto newConcatType = concatType.changeMemSpace(memSpaceCMX);
        _concat.getOutput().setType(newConcatType);
        return;
    }
    // If the outputPattern has MC layer, set the same distribution for Concat output as for the following consumer

    auto outputCopyType = _outputParts.front().copyOp->getResult(0).getType().cast<VPU::DistributedTypeInterface>();
    auto newConcatOutputType =
            getConcatDistributedType(outputCopyType, concatType.getShape(), concatType.getElementType());
    _concat.getOutput().setType(newConcatOutputType);
}

void OutputConcatPattern::replaceSliceCopy() {
    VPUX_THROW_WHEN(_outputParts.empty(), "Concat has no output parts");
    mlir::OpBuilder builder(_outputParts[0].copyOp);
    for (auto concatPart : _outputParts) {
        // correct the tensor type for slice op
        if (concatPart.hasSliceOp()) {
            auto origSliceOp = concatPart.sliceOp;
            builder.setInsertionPoint(origSliceOp);
            _log.trace("Creating VPU.Slice '{0}' at '{1}'", origSliceOp->getName(), origSliceOp->getLoc());
            auto newSliceOp =
                    builder.create<VPU::SliceOp>(origSliceOp->getLoc(), origSliceOp->getOperand(0),
                                                 origSliceOp.getStaticOffsetsAttr(), origSliceOp.getStaticSizesAttr());
            origSliceOp.replaceAllUsesWith(newSliceOp.getResult());
        }

        _log.trace("Removing output Copy from DDR to NNCMX '{0}' at '{1}'. Operand: {2}",
                   concatPart.getCopyOp()->getName(), concatPart.getCopyOp()->getLoc(),
                   concatPart.getCopyOp()->getOperand(0));

        concatPart.getCopyOp()->getResult(0).replaceAllUsesWith(concatPart.getCopyOp()->getOperand(0));
    }
}

//
// CMXConcat
//

class CMXConcatPass final : public CMXConcatBase<CMXConcatPass> {
public:
    explicit CMXConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;

private:
    mlir::FailureOr<InputConcatPattern> getInputPattern(VPU::ConcatOp concat);
    mlir::FailureOr<OutputConcatPattern> getOutputPattern(VPU::ConcatOp concat);

    bool isSplitSupportedOnDPU(VPU::SliceOp sliceOp);
    bool isPotentialCMXConcat(VPU::ConcatOp concat);
    bool areInputOutputPatternsCompatible(InputConcatPattern& inputPattern, OutputConcatPattern& outputPattern);
};

mlir::FailureOr<InputConcatPattern> CMXConcatPass::getInputPattern(VPU::ConcatOp concat) {
    SmallVector<InputConcatPart> inputParts;
    llvm::SmallPtrSet<mlir::Value, 4> nceInputs;

    const auto getBlockArgOrConstant = [](mlir::Value input) -> mlir::Value {
        if (input.isa<mlir::BlockArgument>() || input.getDefiningOp<Const::DeclareOp>() != nullptr) {
            return input;
        }

        auto viewLike = input.getDefiningOp<VPU::ViewLikeOpInterface>();
        while (viewLike != nullptr) {
            auto maybeBlockArg = viewLike.getOperation()->getOperand(0);
            if (viewLike.getOperation()->getOperand(0).isa<mlir::BlockArgument>()) {
                return maybeBlockArg;
            }

            viewLike = maybeBlockArg.getDefiningOp<VPU::ViewLikeOpInterface>();
        }

        return nullptr;
    };

    const auto& logNest = _log.nest(2);
    for (auto input : concat.getOperands()) {
        const auto maybeBlockArgOrConstant = getBlockArgOrConstant(input);
        if (maybeBlockArgOrConstant != nullptr) {
            inputParts.push_back(InputConcatPart(input));
            continue;
        }

        auto inputCopyOp = input.getDefiningOp<VPU::CopyOp>();
        if (inputCopyOp == nullptr) {
            logNest.trace("InputPattern mismatch: Copy op is not found");
            return mlir::failure();
        }

        auto parentNCEOp = inputCopyOp.getInput().getDefiningOp<VPU::NCEOpInterface>();
        auto isDistributedType = [](mlir::Value val) {
            auto distributedIf = mlir::dyn_cast<VPU::DistributedTypeInterface>(val.getType());
            return ((distributedIf != nullptr) && (distributedIf.containsDistributedTypes()));
        };

        if (isDistributedType(inputCopyOp->getOperand(0)) || isDistributedType(inputCopyOp->getResult(0))) {
            if (parentNCEOp) {
                // ViewOp is inserted between NCEPermuteOp and ClusterTiling copy(CMX2CMX) when bufferizing for
                // NCEPermuteOp.
                // The ViewOp would block removing this CMX2CMX ClusterTiling copy and lead to unrolling error
                // because the input and ouput are both distributed.
                // So skip CMXConcat for NCEPermuteOp, see E#118060 for details.
                if (mlir::isa<VPU::NCEPermuteOp>(parentNCEOp)) {
                    logNest.trace("Skip CMXConcat for NCEPermuteOp");
                    return mlir::failure();
                }
            }
        }
        if (parentNCEOp == nullptr) {
            logNest.trace("InputPattern mismatch: NCE op is not found");
            return mlir::failure();
        }

        if (nceInputs.contains(input)) {
            return mlir::failure();
        }
        nceInputs.insert(input);
        inputParts.push_back(InputConcatPart(input, inputCopyOp, parentNCEOp));
    }

    auto hasNce = llvm::any_of(inputParts, [](InputConcatPart& part) {
        return !part.isBlockArgOrConsant;
    });

    if (!hasNce) {
        logNest.trace("All inputs are constant");
        return mlir::failure();
    }

    return InputConcatPattern(concat, inputParts, _log.nest(2));
}

mlir::FailureOr<OutputConcatPattern> CMXConcatPass::getOutputPattern(VPU::ConcatOp concat) {
    SmallVector<OutputConcatPart> outputParts;

    const auto& logNest = _log.nest(2);
    for (auto user : concat.getOutput().getUsers()) {
        auto outputSliceOp = mlir::dyn_cast<VPU::SliceOp>(user);
        auto outputCopyOp = mlir::dyn_cast<VPU::CopyOp>(user);

        // Store the CopyOp or ClusterTiling(CopyOp)
        SmallVector<mlir::Operation*> copyOutOps;
        if (outputSliceOp) {
            // case 1. if the child of concat is SliceOp
            if (!isSplitSupportedOnDPU(outputSliceOp)) {
                logNest.trace("OutputPattern mismatch: SliceOp is not supported on DPU");
                return mlir::failure();
            }
            if (!outputSliceOp->hasOneUse()) {
                logNest.trace("OutputPattern mismatch: SliceOp is not supported because of multiple uses");
                return mlir::failure();
            }
            for (auto copyOp : outputSliceOp.getResult().getUsers()) {
                outputCopyOp = mlir::dyn_cast<VPU::CopyOp>(copyOp);
                if (outputCopyOp) {
                    // match Concat->SliceOp->CopyOp
                    copyOutOps.push_back(outputCopyOp);
                    continue;
                }
                logNest.trace("OutputPattern mismatch: No CopyOp after Slice");
                return mlir::failure();
            }
        } else if (outputCopyOp) {
            // case 3. if the child of concat is CopyOp
            copyOutOps.push_back(outputCopyOp);
        } else {
            logNest.trace("OutputPattern mismatch: No CopyOp");
            return mlir::failure();
        }

        // Look for the NCEOps according to the CopyOps
        for (auto& op : copyOutOps) {
            for (auto opUser : op->getResult(0).getUsers()) {
                auto childNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(opUser);

                if (childNCEOp == nullptr) {
                    logNest.trace("OutputPattern mismatch: No NCEOp");
                    return mlir::failure();
                }
                auto index = getIndexOfInput(childNCEOp, op);
                outputParts.push_back(
                        OutputConcatPart(mlir::dyn_cast<VPU::CopyOp>(op), outputSliceOp, childNCEOp, index));
            }
        }
    }

    return OutputConcatPattern(concat, outputParts, _log);
}

bool CMXConcatPass::isSplitSupportedOnDPU(VPU::SliceOp sliceOp) {
    // Check if SubView performs a split along major dimension taking into account order in memory
    // For NCHW that would be split along C
    // For NHWC that would be split along H
    // Only such cases are supported by DPU IDU because only then input to DPU is a contiguous
    // block in memory. Otherwise this behavior needs to be performed by DMA
    const auto inputTypeShape = getShape(sliceOp.getOperand()).raw();
    const auto outputType = sliceOp.getResult().getType();

    auto shapedType = outputType.cast<vpux::NDTypeInterface>();
    const auto outputTypeShape = shapedType.getShape().raw();

    if (inputTypeShape.size() != outputTypeShape.size()) {
        return false;
    }

    size_t dimsDifference = 0;
    size_t dimsDifferenceCount = 0;
    const auto order = shapedType.getDimsOrder();

    for (size_t i = 0; i < inputTypeShape.size(); i++) {
        if (inputTypeShape[i] != outputTypeShape[i]) {
            dimsDifference = i;
            dimsDifferenceCount++;
        }
    }

    if (dimsDifferenceCount > 1) {
        return false;
    }

    if (static_cast<int32_t>(dimsDifference) == Dims4D::Act::C.ind() && order == DimsOrder::NCHW) {
        return true;
    }

    if (static_cast<int32_t>(dimsDifference) == Dims4D::Act::H.ind() && order == DimsOrder::NHWC) {
        return true;
    }

    return false;
}

bool CMXConcatPass::isPotentialCMXConcat(VPU::ConcatOp concat) {
    // if concat is a Result operation
    auto hasReturnUser = llvm::any_of(concat.getOutput().getUsers(), [](mlir::Operation* outputUser) {
        return mlir::isa<mlir::func::ReturnOp>(outputUser);
    });

    if (hasReturnUser) {
        _log.trace("Concat output is part of network output");
        return false;
    }

    // Check if the Concat op satisfies the CMX Concat conditions or not
    auto isSingleAxisConcat = [](mlir::ArrayAttr offset) {
        // If a concat has at least one static_offset attribute of 2 or more non-zero axis
        // it is considered as multiple-axis concat, vice versa
        // e.g., static_offset of a multiple-axis concat:
        // [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]
        auto offsetVector = parseIntArrayAttr<int64_t>(offset);
        return offsetVector.size() - std::count(offsetVector.begin(), offsetVector.end(), 0) <= 1;
    };

    if (!concat.getStaticOffsetsAttr()) {
        return true;
    }

    return llvm::all_of(concat.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>(), isSingleAxisConcat);
}

bool CMXConcatPass::areInputOutputPatternsCompatible(InputConcatPattern& inputPattern,
                                                     OutputConcatPattern& outputPattern) {
    // Check if the input and outputPattern satisfy
    // both are distributed types or both are normal types
    bool inputHasDistributed = false;
    bool outputHasDistributed = false;
    for (auto concatPart : inputPattern.getInputParts()) {
        if (concatPart.isMultiCluster()) {
            inputHasDistributed = true;
            break;
        }
    }
    for (auto concatPart : outputPattern.getOutputParts()) {
        if (concatPart.isMultiCluster()) {
            outputHasDistributed = true;
            break;
        }
    }
    if (inputHasDistributed != outputHasDistributed) {
        // different input output type
        return false;
    }

    const auto& inIt = llvm::find_if(inputPattern.getInputParts(), [](const InputConcatPart& inPart) {
        return inPart.isMultiCluster();
    });
    const auto& outIt = llvm::find_if(outputPattern.getOutputParts(), [](const OutputConcatPart& inPart) {
        return inPart.isMultiCluster();
    });

    if (inputHasDistributed && outputHasDistributed) {
        const auto inputType = (*inIt).nceOp->getResult(0)
                                       .getType()
                                       .cast<VPU::DistributedTypeInterface>()
                                       .getDistributedTypes()
                                       .front()
                                       .cast<VPU::DistributedTensorType>();
        const auto outputType = (*outIt).nceOp->getOperand(0)
                                        .getType()
                                        .cast<VPU::DistributedTypeInterface>()
                                        .getDistributedTypes()
                                        .front()
                                        .cast<VPU::DistributedTensorType>();

        if (!areDistributedTypesConcatenable(inputType, outputType)) {
            _log.trace("Distributed tensor attributes for concat input and output do not match: `{0}` and `{1}`",
                       inputType, outputType);
            return false;
        }

        // Only Slices on one axis is supported by pass
        auto getSplitAxis = [](VPU::SliceOp slice) -> size_t {
            const auto inputShape = getShape(slice.getOperand()).raw();
            const auto outputShape = slice.getResult().getType().cast<NDTypeInterface>().getShape().raw();

            for (size_t dim = 0; dim < inputShape.size(); dim++) {
                if (inputShape[dim] != outputShape[dim]) {
                    return dim;
                }
            }

            VPUX_THROW("SliceOp is invalid: there is no slice axis");
        };

        for (const auto& outPart : outputPattern.getOutputParts()) {
            if (!outPart.isMultiCluster()) {
                continue;
            }

            // Check that all Slices after the Concat have their axis different than the axis
            // of clustering, if the mode is SEGMENTED for input
            const auto distribution = inputType.getDistribution().getMode().getValue();
            if (outPart.hasSliceOp() && (distribution == VPU::DistributionMode::SEGMENTED ||
                                         distribution == VPU::DistributionMode::OVERLAPPED)) {
                const auto numTiles = parseIntArrayAttr<int64_t>(inputType.getDistribution().getNumTiles());
                const auto sliceAxis = getSplitAxis(outPart.sliceOp);

                if (numTiles[sliceAxis] != 1) {
                    return false;
                }
            }
        }
    }
    return true;
}

void CMXConcatPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto availableMem = VPU::getTotalCMXSize(module);
    const auto cmxSize = checked_cast<size_t>(availableMem.count());

    const auto& nestLog = _log.nest();

    func->walk([&](VPU::ConcatOp concat) {
        // check concat input pattern
        _log.trace("Got '{0}' at '{1}'", concat->getName(), concat->getLoc());
        if (!isPotentialCMXConcat(concat)) {
            nestLog.trace("Concat cannot be executed on CMX");
            return;
        }
        auto potentialInputPattern = getInputPattern(concat);
        if (mlir::failed(potentialInputPattern)) {
            nestLog.trace("Concat input pattern not valid");
            return;
        }

        auto inputPattern = potentialInputPattern.value();
        if (!inputPattern.inputPatternCanBeCMXed(cmxSize)) {
            nestLog.trace("Concat input pattern can not be cmx-ed");
            return;
        }

        // check concat output pattern
        auto potentialOutputPattern = getOutputPattern(concat);
        if (mlir::succeeded(potentialOutputPattern)) {
            auto outputPattern = potentialOutputPattern.value();
            if (!outputPattern.outputPatternCanBeCMXed(cmxSize)) {
                nestLog.trace("Concat output pattern can not be cmx-ed");
                return;
            }

            if (!areInputOutputPatternsCompatible(inputPattern, outputPattern)) {
                nestLog.trace("Concat input and output pattern type mismatch");
                return;
            }
            outputPattern.rewrite();
        } else {
            nestLog.trace("Concat output pattern not valid");
            if (inputPattern.inputConcatOnlyMeetRequirement()) {
                inputPattern.isInputConcatOnly = true;
            } else {
                nestLog.trace("Cannot move only Concat inputs to CMX");
                return;
            }
        }

        inputPattern.rewrite();
        _log.trace("Concat '{0}' at '{1}' will be moved to CMX", concat->getName(), concat->getLoc());
    });
}

}  // namespace

//
// createCMXConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCMXConcatPass(Logger log) {
    return std::make_unique<CMXConcatPass>(log);
}

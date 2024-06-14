//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"

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
    VPU::NCEClusterTilingOp copyClusterOp;
    VPU::NCEOpInterface nceOp;
    VPU::NCEClusterTilingOp nceClusterOp;

    NceBasedPart(VPU::CopyOp copy, VPU::NCEOpInterface nce)
            : copyOp(copy), copyClusterOp(nullptr), nceOp(nce), nceClusterOp(nullptr) {
    }

    NceBasedPart(VPU::CopyOp copy, VPU::NCEClusterTilingOp copyCluster, VPU::NCEOpInterface nce,
                 VPU::NCEClusterTilingOp nceCluster)
            : copyOp(copy), copyClusterOp(copyCluster), nceOp(nce), nceClusterOp(nceCluster) {
    }

    virtual ~NceBasedPart() = default;

    virtual bool isMultiCluster() const {
        return copyClusterOp != nullptr && nceClusterOp != nullptr;
    }

    virtual mlir::Operation* getCopyOp() {
        return copyClusterOp != nullptr ? copyClusterOp.getOperation() : copyOp.getOperation();
    }

    virtual mlir::Operation* getNceOp() {
        return nceClusterOp != nullptr ? nceClusterOp.getOperation() : nceOp.getOperation();
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

    InputConcatPart(mlir::Value operand)
            : NceBasedPart(nullptr, nullptr), concatOperand(operand), isBlockArgOrConsant(true) {
    }

    InputConcatPart(mlir::Value operand, VPU::CopyOp copy, VPU::NCEOpInterface nce)
            : NceBasedPart(copy, nce), concatOperand(operand), isBlockArgOrConsant(false) {
        if (copy == nullptr && nce == nullptr) {
            isBlockArgOrConsant = true;
        }
    }

    InputConcatPart(mlir::Value operand, VPU::CopyOp copy, VPU::NCEClusterTilingOp copyCluster, VPU::NCEOpInterface nce,
                    VPU::NCEClusterTilingOp nceCluster)
            : NceBasedPart(copy, copyCluster, nce, nceCluster), concatOperand(operand), isBlockArgOrConsant(false) {
        if (copy == nullptr && copyCluster == nullptr && nce == nullptr && nceCluster == nullptr) {
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

private:
    VPU::ConcatOp _concat;
    SmallVector<InputConcatPart> _inputParts;
    Logger _log;

private:
    size_t getConcatSize();
    bool concatFitsInCMX(size_t cmxSize);
    bool inputsHaveNotOnlyCopiesUsers() const;
    bool isMemConsistentPerCluster();
    bool areDistributionTypesConsistent();
    bool areAnyInputsInPlace();
};

ArrayRef<InputConcatPart> InputConcatPattern::getInputParts() const {
    return _inputParts;
}

void InputConcatPattern::rewrite() {
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

            auto nceProducerOutput = inputPart.getCopyOp()->getOperand(0);
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
                _log.trace("Inserting DistributedCast at '{1}'", inputPart.getCopyOp()->getLoc());
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

            auto newConcatInput = inputPart.getCopyOp()->getOperand(0);
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
        const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                 mlir::ValueRange newOperands) {
            auto outputTensorDistributedCopyOp = builder.create<VPU::CopyOp>(loc, newOperands[0], memSpace);
            builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
        };

        auto distributedConcatOutType = _concat.getOutput().getType().cast<VPU::DistributedTypeInterface>();
        auto newType =
                getConcatDistributedType(distributedConcatOutType, origType.getShape(), origType.getElementType());

        _log.trace("Insert Cluster tiling Copy from DDR to CMX for constant input '{0}'", inputPart.concatOperand);
        const auto newConcatInputOp = builder.create<NCEClusterTilingOp>(
                _concat.getLoc(), newType, inputPart.concatOperand, outputTensorBodyBuilder);

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
            for (auto input : nceOp->getOperands()) {
                currUserSize += getSize(input.getType());
            }
        }
        maxUserSize = std::max<size_t>(maxUserSize, currUserSize);
    }

    _log.trace("Concat size '{0}'", (concatSize + maxUserSize));
    // return concat size smaller than CMX size
    return (concatSize + maxUserSize) <= cmxSize;
}

bool InputConcatPattern::inputsHaveNotOnlyCopiesUsers() const {
    // avoid concats which are complex, where the inputs to the concat are used
    // by other operations

    const auto userIsCopyOp = [](mlir::Operation* user) {
        if (mlir::isa<VPU::CopyOp>(user)) {
            return true;
        }

        auto clusterTiling = mlir::dyn_cast<VPU::NCEClusterTilingOp>(user);
        if (clusterTiling == nullptr) {
            return false;
        }

        return clusterTiling.getInnerTaskOpOfType<VPU::CopyOp>() != nullptr;
    };

    for (auto concatPart : _inputParts) {
        auto nceOp = concatPart.getNceOp();
        if (nceOp == nullptr) {
            continue;
        }

        for (auto result : nceOp->getResults()) {
            for (auto user : result.getUsers()) {
                if (!userIsCopyOp(user)) {
                    return true;
                }
            }
        }
    }

    return false;
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
        const auto disType = concatPart.nceClusterOp.getResult(0)
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
            multiclusterPart.nceClusterOp.getResult(0).getType().cast<VPU::DistributedTypeInterface>();
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
                part.nceClusterOp.getResult(0).getType().cast<VPU::DistributedTypeInterface>();
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
        if (auto clusterTilingOp = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(inputPart.getNceOp())) {
            nceEltwiseOp = clusterTilingOp.getInnerTaskOpOfType<VPU::NCEEltwiseOp>();
        }
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
        // TODO implement complex concat
        // where part of the concatenated buffer is also used by another operation
        // visible in yolo-v4-tiny concatinate 4
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

    OutputConcatPart(VPU::CopyOp copy, VPU::NCEClusterTilingOp copyCluster, VPU::SliceOp slice, VPU::NCEOpInterface nce,
                     VPU::NCEClusterTilingOp nceCluster, unsigned idx)
            : NceBasedPart(copy, copyCluster, nce, nceCluster), sliceOp(slice), nceInput(idx) {
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

    const auto distributedTypeInterfaceInput = _outputParts[0]
                                                       .nceClusterOp.getOperand(_outputParts[0].nceInput)
                                                       .getType()
                                                       .cast<VPU::DistributedTypeInterface>();
    if (!distributedTypeInterfaceInput.containsDistributedTypes()) {
        return true;
    }
    auto inTypeDistributed =
            distributedTypeInterfaceInput.getDistributedTypes().front().cast<VPU::DistributedTensorType>();

    if (inTypeDistributed != nullptr) {
        for (auto concatPart : ArrayRef(_outputParts).drop_front()) {
            const auto distributedTypeInterfaceOutput = concatPart.nceClusterOp.getOperand(concatPart.nceInput)
                                                                .getType()
                                                                .cast<VPU::DistributedTypeInterface>();
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

    auto outputCopyType =
            _outputParts.front().copyClusterOp->getResult(0).getType().cast<VPU::DistributedTypeInterface>();
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
        auto inputClusterTilingOp = input.getDefiningOp<VPU::NCEClusterTilingOp>();
        if (inputClusterTilingOp) {
            if (auto innerOp = inputClusterTilingOp.getInnerTaskOpOfType<VPU::CopyOp>()) {
                inputCopyOp = innerOp;
            }
        }

        if (inputCopyOp == nullptr) {
            logNest.trace("InputPattern mismatch: Copy op is not found");
            return mlir::failure();
        }

        auto parentNCEOp = inputCopyOp.getInput().getDefiningOp<VPU::NCEOpInterface>();
        VPU::NCEClusterTilingOp parentNCEClusterTilingOp;

        if (inputClusterTilingOp) {
            auto inputClusterTiling = inputClusterTilingOp.getOperand(0);
            parentNCEClusterTilingOp = inputClusterTiling.getDefiningOp<VPU::NCEClusterTilingOp>();
            if (parentNCEClusterTilingOp) {
                if (auto innerOp = parentNCEClusterTilingOp.getInnerTaskOpOfType<VPU::NCEOpInterface>()) {
                    parentNCEOp = innerOp;
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
        }

        if (parentNCEOp == nullptr) {
            logNest.trace("InputPattern mismatch: NCE op is not found");
            return mlir::failure();
        }

        if (nceInputs.contains(input)) {
            return mlir::failure();
        }
        nceInputs.insert(input);
        inputParts.push_back(
                InputConcatPart(input, inputCopyOp, inputClusterTilingOp, parentNCEOp, parentNCEClusterTilingOp));
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
        auto outputClusterCopyOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(user);

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
                outputClusterCopyOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(copyOp);

                if (outputClusterCopyOp) {
                    if (outputClusterCopyOp.getInnerTaskOpOfType<VPU::CopyOp>()) {
                        // match Concat->SliceOp->ClusterTiling(CopyOp)
                        copyOutOps.push_back(outputClusterCopyOp);
                        continue;
                    }

                    logNest.trace("OutputPattern mismatch: No CopyOp in the ClusterTilingOp after Slice");
                    return mlir::failure();
                } else if (outputCopyOp) {
                    // match Concat->SliceOp->CopyOp
                    copyOutOps.push_back(outputCopyOp);
                    continue;
                }
                logNest.trace("OutputPattern mismatch: No CopyOp after Slice");
                return mlir::failure();
            }
        } else if (outputClusterCopyOp) {
            // case 2. if the child of concat is ClusterTilingOp
            if (outputClusterCopyOp.getInnerTaskOpOfType<VPU::CopyOp>() == nullptr) {
                logNest.trace("OutputPattern mismatch: No CopyOp in the ClusterTilingOp");
                return mlir::failure();
            }

            // match Concat->ClusterTiling(CopyOp)
            copyOutOps.push_back(outputClusterCopyOp);
        } else if (outputCopyOp) {
            // case 3. if the child of concat is CopyOp
            copyOutOps.push_back(outputCopyOp);
        } else {
            logNest.trace("OutputPattern mismatch: No CopyOp");
            return mlir::failure();
        }

        // Look for the NCEOps according to the CopyOps
        for (auto& op : copyOutOps) {
            llvm::DenseSet<mlir::Operation*> nceUsers;
            for (auto opUser : op->getResult(0).getUsers()) {
                auto childNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(opUser);
                auto childNCEClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(opUser);
                if (childNCEClusterOp) {
                    if (auto innerOp = childNCEClusterOp.getInnerTaskOpOfType<VPU::NCEOpInterface>()) {
                        childNCEOp = innerOp;
                    }
                }

                if (childNCEOp == nullptr) {
                    logNest.trace("OutputPattern mismatch: No NCEOp");
                    return mlir::failure();
                }

                if (nceUsers.find(opUser) != nceUsers.end()) {
                    // avoid multiple reads from the same location at the same time
                    logNest.trace("Concat input used twice by the same operation, can not cmx");
                    return mlir::failure();
                }
                nceUsers.insert(opUser);
                if (childNCEClusterOp) {
                    auto copyOp = mlir::dyn_cast<VPU::CopyOp>(op);
                    auto copyClusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(op);
                    if (copyClusterOp) {
                        if (auto innerOp = copyClusterOp.getInnerTaskOpOfType<VPU::CopyOp>()) {
                            copyOp = innerOp;
                        }
                    }

                    if (copyOp == nullptr) {
                        logNest.trace("OutputPattern mismatch: No CopyOp");
                        return mlir::failure();
                    }

                    auto index = getIndexOfInput(childNCEClusterOp, op);
                    outputParts.push_back(OutputConcatPart(copyOp, copyClusterOp, outputSliceOp, childNCEOp,
                                                           childNCEClusterOp, index));
                } else {
                    auto index = getIndexOfInput(childNCEOp, op);
                    outputParts.push_back(
                            OutputConcatPart(mlir::dyn_cast<VPU::CopyOp>(op), outputSliceOp, childNCEOp, index));
                }
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
        const auto inputType = (*inIt).nceClusterOp->getResult(0)
                                       .getType()
                                       .cast<VPU::DistributedTypeInterface>()
                                       .getDistributedTypes()
                                       .front()
                                       .cast<VPU::DistributedTensorType>();
        const auto outputType = (*outIt).nceClusterOp->getOperand(0)
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
        if (mlir::failed(potentialOutputPattern)) {
            nestLog.trace("Concat output pattern not valid");
            return;
        }

        auto outputPattern = potentialOutputPattern.value();
        if (!outputPattern.outputPatternCanBeCMXed(cmxSize)) {
            nestLog.trace("Concat output pattern can not be cmx-ed");
            return;
        }

        if (!areInputOutputPatternsCompatible(inputPattern, outputPattern)) {
            nestLog.trace("Concat input and output pattern type mismatch");
            return;
        }

        // rewrite from DDR to NNCMX
        _log.trace("Concat '{0}' at '{1}' will be moved to CMX", concat->getName(), concat->getLoc());
        outputPattern.rewrite();
        inputPattern.rewrite();
    });
}

}  // namespace

//
// createCMXConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCMXConcatPass(Logger log) {
    return std::make_unique<CMXConcatPass>(log);
}

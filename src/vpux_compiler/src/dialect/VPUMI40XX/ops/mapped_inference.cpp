#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"

using namespace vpux;

size_t countEmptyArray(mlir::ArrayAttr array, int64_t limit) {
    const auto emptyPredicate = [](const auto& item) -> bool {
        return item.template cast<mlir::ArrayAttr>().empty();
    };
    return std::count_if(array.begin(), array.begin() + limit, emptyPredicate);
}

size_t countZeroes(mlir::ArrayAttr array, int64_t limit) {
    const auto zeroPredicate = [](const int val) -> bool {
        return val == 0;
    };
    const auto arrAttr = parseIntArrayAttr<int64_t>(array);
    return std::count_if(arrAttr.begin(), arrAttr.begin() + limit, zeroPredicate);
}

mlir::ArrayAttr subArray(mlir::ArrayAttr attr, int64_t idx) {
    return attr[checked_cast<unsigned>(idx)].cast<mlir::ArrayAttr>();
}

mlir::Value vpux::VPUMI40XX::MappedInferenceOp::getListHead(VPURegMapped::TaskType taskType, int64_t tileIdx,
                                                            int64_t listIdx) {
    auto mutableRange = getListHeadMutable(taskType, tileIdx, listIdx);
    if (mutableRange.size() > 0) {
        size_t emptyTiles, emptyLists, majorOperand, minorOperand = 0;
        switch (taskType) {
        case VPURegMapped::TaskType::DMA:
            emptyTiles = countEmptyArray(getDmaCount(), tileIdx);
            majorOperand = tileIdx - emptyTiles;

            emptyLists = countZeroes(subArray(getDmaCount(), tileIdx), listIdx);
            minorOperand = listIdx - emptyLists;

            return getDmaTasks()[majorOperand].slice(minorOperand, 1)[0];
            break;
        case VPURegMapped::TaskType::DPUInvariant:
            emptyTiles = countZeroes(getInvariantCount(), tileIdx);
            return getInvariantTasks().slice(tileIdx - emptyTiles, 1)[0];
            break;
        case VPURegMapped::TaskType::DPUVariant:
            emptyTiles = countZeroes(getVariantCount(), tileIdx);
            return getVariantTasks().slice(tileIdx - emptyTiles, 1)[0];
            break;
        case VPURegMapped::TaskType::ActKernelInvocation:
            emptyTiles = countZeroes(getActKernelInvocationsCount(), tileIdx);
            return getActKernelInvocations().slice(tileIdx - emptyTiles, 1)[0];
            break;
        case VPURegMapped::TaskType::ActKernelRange:
            emptyTiles = countZeroes(getActKernelRangesCount(), tileIdx);
            return getActKernelRanges().slice(tileIdx - emptyTiles, 1)[0];
            break;
        default:
            return nullptr;
            break;
        };
    } else {
        return nullptr;
    }
}

mlir::MutableOperandRange vpux::VPUMI40XX::MappedInferenceOp::getListHeadMutable(VPURegMapped::TaskType taskType,
                                                                                 int64_t tileIdx, int64_t listIdx) {
    auto arrayIdx = [](mlir::ArrayAttr attr, int64_t idx) -> int64_t {
        return attr[checked_cast<unsigned>(idx)].cast<mlir::IntegerAttr>().getInt();
    };

    auto taskListSizeIsNotValid = [&arrayIdx](mlir::ArrayAttr array, int64_t tileIdx) -> bool {
        if (tileIdx >= static_cast<int64_t>(array.size()) || arrayIdx(array, tileIdx) == 0) {
            return true;
        }
        return false;
    };

    auto dmaTaskListIsNotValid = [&arrayIdx](mlir::ArrayAttr array, int64_t tileIdx, int64_t listIdx) -> bool {
        if (tileIdx >= static_cast<int64_t>(array.size()) || subArray(array, tileIdx).size() == 0 ||
            listIdx >= static_cast<int64_t>(subArray(array, tileIdx).size()) ||
            arrayIdx(subArray(array, tileIdx), listIdx) == 0) {
            return true;
        }

        return false;
    };

    size_t emptyTiles, emptyLists, majorOperand, minorOperand = 0;
    auto emptyOperandRange = mlir::MutableOperandRange(getOperation(), 0, 0);

    switch (taskType) {
    case VPURegMapped::TaskType::DMA:
        if (dmaTaskListIsNotValid(getDmaCount(), tileIdx, listIdx))
            return emptyOperandRange;

        emptyTiles = countEmptyArray(getDmaCount(), tileIdx);
        majorOperand = tileIdx - emptyTiles;

        emptyLists = countZeroes(subArray(getDmaCount(), tileIdx), listIdx);
        minorOperand = listIdx - emptyLists;

        return getDmaTasksMutable()[majorOperand].slice(checked_cast<unsigned int>(minorOperand), 1);
        break;
    case VPURegMapped::TaskType::DPUInvariant:
        if (taskListSizeIsNotValid(getInvariantCount(), tileIdx))
            return emptyOperandRange;

        emptyTiles = countZeroes(getInvariantCount(), tileIdx);
        return getInvariantTasksMutable().slice(checked_cast<unsigned int>(tileIdx - emptyTiles), 1);
        break;
    case VPURegMapped::TaskType::DPUVariant:
        if (taskListSizeIsNotValid(getVariantCount(), tileIdx))
            return emptyOperandRange;

        emptyTiles = countZeroes(getVariantCount(), tileIdx);
        return getVariantTasksMutable().slice(checked_cast<unsigned int>(tileIdx - emptyTiles), 1);
        break;
    case VPURegMapped::TaskType::ActKernelInvocation:
        if (taskListSizeIsNotValid(getActKernelInvocationsCount(), tileIdx))
            return emptyOperandRange;

        emptyTiles = countZeroes(getActKernelInvocationsCount(), tileIdx);
        return getActKernelInvocationsMutable().slice(checked_cast<unsigned int>(tileIdx - emptyTiles), 1);
        break;
    case VPURegMapped::TaskType::ActKernelRange:
        if (taskListSizeIsNotValid(getActKernelRangesCount(), tileIdx))
            return emptyOperandRange;

        emptyTiles = countZeroes(getActKernelRangesCount(), tileIdx);
        return getActKernelRangesMutable().slice(checked_cast<unsigned int>(tileIdx - emptyTiles), 1);
        break;
    default:
        return emptyOperandRange;
        break;
    };
}

//
// Dot Printer
//

DotNodeColor VPUMI40XX::MappedInferenceOp::getNodeColor() {
    return DotNodeColor::RED;
}

bool VPUMI40XX::MappedInferenceOp::printAttributes(llvm::raw_ostream&, llvm::StringRef, llvm::StringRef,
                                                   llvm::StringRef) {
    return true;
}

DOT::EdgeDir VPUMI40XX::MappedInferenceOp::getEdgeDirection(mlir::Operation*) {
    return DOT::EdgeDir::EDGE_REVERSE;
}

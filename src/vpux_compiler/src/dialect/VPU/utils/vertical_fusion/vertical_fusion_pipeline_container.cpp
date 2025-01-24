//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_pipeline_container.hpp"

using namespace vpux;
using namespace VPU;

VPU::ExecutorKind getExecutorByOperation(mlir::Operation* operation) {
    if (operation == nullptr) {
        return ExecutorKind::DMA_NN;
    }

    return mlir::isa<SWOpInterface>(operation) ? ExecutorKind::SHAVE_ACT : ExecutorKind::DPU;
}

TimelineInterval::TimelineInterval(StrategyCost begin, StrategyCost end, mlir::Operation* operation, int64_t index)
        : _mBegin(begin), _mEnd(end), _mOperation(operation), _mIndex(index) {
    _mExecutor = getExecutorByOperation(operation);
}

VFPipelineContainer::VFPipelineContainer() {
}

bool VFPipelineContainer::setPlaceInTimeline(mlir::Operation* operation, int64_t index, const StrategyCost& cost) {
    if (cost == 0) {
        return false;
    }

    if (_containerMapper.empty()) {
        _containerMapper.emplace_back(0, cost, operation, index);
        _lastInterval = _containerMapper.back();
        return true;
    }

    if (!_lastInterval.has_value()) {
        return false;
    }

    if (_lastInterval.value()._mIndex == index) {
        // add further to the timeline
        auto lastEnd = _lastInterval.value()._mEnd;
        _containerMapper.emplace_back(lastEnd, lastEnd + cost, operation, index);
        _lastInterval = _containerMapper.back();
        return true;
    }

    // else try to pipeline
    auto executor = getExecutorByOperation(operation);
    auto foundExecutor = llvm::find_if(_containerMapper | reversed, [&](auto item) {
        return item._mExecutor == executor;
    });

    auto foundByIndex = llvm::find_if(_containerMapper | reversed, [&](auto item) {
        return item._mIndex == index;
    });

    if (foundExecutor == _containerMapper.rend() && foundByIndex == _containerMapper.rend()) {
        return false;
    }

    auto lastEnd = std::numeric_limits<StrategyCost>::min();
    if (foundExecutor != _containerMapper.rend()) {
        lastEnd = std::max(lastEnd, foundExecutor->_mEnd);
    }
    if (foundByIndex != _containerMapper.rend()) {
        lastEnd = std::max(lastEnd, foundByIndex->_mEnd);
    }

    if (lastEnd + cost >= _lastInterval.value()._mEnd) {
        _containerMapper.emplace_back(lastEnd, lastEnd + cost, operation, index);
        _lastInterval = _containerMapper.back();
        return true;
    }

    _containerMapper.emplace_back(lastEnd, lastEnd + cost, operation, index);
    return true;
}

std::optional<int64_t> VFPipelineContainer::getLastIntervalIndex() const {
    if (!_lastInterval.has_value()) {
        return std::nullopt;
    }

    return _lastInterval.value()._mIndex;
}

SmallVector<std::pair<int64_t, mlir::Operation*>> VFPipelineContainer::getTimeLine() const {
    SmallVector<std::pair<int64_t, mlir::Operation*>> result;

    const auto filterOperation = [](auto& value) {
        return value._mOperation != nullptr;
    };

    const auto transformOperation = [](auto& value) {
        return std::make_pair(value._mIndex, value._mOperation);
    };

    llvm::transform(_containerMapper | filtered(filterOperation), std::back_inserter(result), transformOperation);

    return result;
}

bool VFPipelineContainer::isPipelineAvailable(int64_t index, mlir::Operation* operation, StrategyCost cost) const {
    if (!_lastInterval.has_value()) {
        return false;
    }

    if (_lastInterval.value()._mIndex == index) {
        return false;
    }

    auto executor = getExecutorByOperation(operation);
    if (_lastInterval.value()._mExecutor == executor) {
        return false;
    }

    auto foundExecutor = llvm::find_if(_containerMapper | reversed, [&](auto item) {
        return item._mExecutor == executor;
    });

    if (foundExecutor == _containerMapper.rend()) {
        return false;
    }

    return foundExecutor->_mEnd + cost <= _lastInterval.value()._mEnd;
}

StrategyCost VFPipelineContainer::getPrefetchAvailability() const {
    if (!_lastInterval.has_value()) {
        return 0;
    }

    auto foundExecutor = llvm::find_if(_containerMapper | reversed, [](auto item) {
        return item._mExecutor == VPU::ExecutorKind::DMA_NN;
    });

    auto currentCost = maxCost();
    auto prefetchedCost = 0;
    if (foundExecutor != _containerMapper.rend()) {
        prefetchedCost = foundExecutor->_mEnd;
    }

    return currentCost - prefetchedCost;
}

bool VFPipelineContainer::addOperation(mlir::Operation* operation, int64_t index, const StrategyCost& cost) {
    return setPlaceInTimeline(operation, index, cost);
}

bool VFPipelineContainer::addDMA(int64_t index, const StrategyCost& cost) {
    return setPlaceInTimeline(nullptr, index, cost);
}

StrategyCost VFPipelineContainer::maxCost() const {
    if (!_lastInterval.has_value()) {
        return 0;
    }
    return _lastInterval.value()._mEnd;
}

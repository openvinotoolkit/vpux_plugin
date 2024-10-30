//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/parser/parser.hpp"
#include "vpux/utils/profiling/parser/records.hpp"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace vpux::profiling;

namespace {

// Main synchronization primitive. First contain tasks, that update barrier, Second - that wait for barrier
using SynchronizationPoint = std::pair<RawProfilingRecords, RawProfilingRecords>;
using SynchronizationPointsContainer = std::vector<SynchronizationPoint>;

template <class IterableContainer>
std::string convertIterableToString(const IterableContainer& container) {
    if (container.empty()) {
        return "[]";
    }
    const auto last = std::prev(container.cend());
    std::stringstream ss;
    ss << "[";
    for (auto it = container.cbegin(); it != last; ++it) {
        ss << *it << ", ";
    }
    ss << *last << "]";
    return ss.str();
}

inline std::string to_string(const SynchronizationPoint& syncPoint) {
    const auto getType = [](const auto& vec) -> std::string {
        VPUX_THROW_WHEN(vec.empty(), "No records");
        return convertExecTypeToName(vec.front()->getExecutorType());
    };
    const auto printNames = [](const auto& x) {
        std::vector<std::string> names;
        for (const auto& t : x) {
            names.push_back(t->getTaskName());
        }
        return convertIterableToString(names);
    };

    return printNames(syncPoint.first) + " " + getType(syncPoint.first) + " -> " + getType(syncPoint.second) + " " +
           printNames(syncPoint.second);
}

RawProfilingRecords getRelatedTasksOfKind(
        RawProfilingRecord::BarrierIdType barrierId,
        const std::multimap<RawProfilingRecord::BarrierIdType, RawProfilingRecordPtr>& relatedTasks,
        ExecutorType execKind) {
    RawProfilingRecords tasks;
    auto range = relatedTasks.equal_range(barrierId);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second->getExecutorType() == execKind) {
            tasks.push_back(it->second);
        }
    }
    return tasks;
}

SynchronizationPointsContainer findSynchronizationPoints(const RawProfilingRecords& taskGroup1,
                                                         const RawProfilingRecords& taskGroup2,
                                                         SynchronizationPointKind pointKind) {
    using BarrierIdType = RawProfilingRecord::BarrierIdType;
    std::multimap<BarrierIdType, RawProfilingRecordPtr> barrierPredecessors;
    std::set<BarrierIdType> waitBarriers;
    std::multimap<BarrierIdType, RawProfilingRecordPtr> barrierSuccessors;
    std::set<BarrierIdType> updateBarriers;
    std::unordered_set<BarrierIdType> dpuUpdatedBarriers;

    for (const auto& tasksGroup : {taskGroup1, taskGroup2}) {
        for (const auto& task : tasksGroup) {
            for (const auto waitBarrier : task->getWaitBarriers()) {
                barrierSuccessors.insert(std::make_pair(waitBarrier, task));
                waitBarriers.insert(waitBarrier);
            }
            for (const auto updateBarrier : task->getUpdateBarriers()) {
                barrierPredecessors.insert(std::make_pair(updateBarrier, task));
                updateBarriers.insert(updateBarrier);
                if (task->getExecutorType() == ExecutorType::DPU) {
                    dpuUpdatedBarriers.insert(updateBarrier);
                }
            }
        }
    }

    ExecutorType predecessorExecType = ExecutorType::DMA_SW;
    ExecutorType successorExecType = ExecutorType::DPU;
    if (pointKind == SynchronizationPointKind::DPU_TO_DMA) {
        std::swap(predecessorExecType, successorExecType);
    }

    // Possible synchronization points occurs on covered from both directions barriers
    const auto commonBarriers = RawProfilingRecord::getBarriersIntersection(waitBarriers, updateBarriers);
    SynchronizationPointsContainer synchronizationPoints;
    size_t numUnsuitableSyncPoints = 0;
    for (const auto& commonBarrier : commonBarriers) {
        if (pointKind == SynchronizationPointKind::STRICT_DMA_TO_DPU && dpuUpdatedBarriers.count(commonBarrier) != 0) {
            ++numUnsuitableSyncPoints;
            continue;
        }
        RawProfilingRecords predecessors =
                getRelatedTasksOfKind(commonBarrier, barrierPredecessors, predecessorExecType);
        RawProfilingRecords successors = getRelatedTasksOfKind(commonBarrier, barrierSuccessors, successorExecType);
        if (!predecessors.empty() && !successors.empty()) {
            synchronizationPoints.push_back(std::make_pair(predecessors, successors));
        }
    }
    if (numUnsuitableSyncPoints != 0) {
        vpux::Logger::global().trace(
                "Found {0} synchronization points. {1} are unused because of requested SynchronizationPointKind",
                synchronizationPoints.size(), numUnsuitableSyncPoints);
    }
    return synchronizationPoints;
}

// Get a shift in time for the synchronization point. The shift is defined as a difference between the latest update
// task and the earliest wait task
std::vector<RawProfilingRecord::TimeType> getBarrierShiftEstimations(const SynchronizationPointsContainer& syncPoints,
                                                                     FrequenciesSetup frequenciesSetup,
                                                                     vpux::Logger& log, bool extraVerbosity = false) {
    using TimeType = RawProfilingRecord::TimeType;

    std::vector<double> syncShiftsEstimations;
    syncShiftsEstimations.reserve(syncPoints.size());
    for (const auto& syncPoint : syncPoints) {
        TimeType latestPreBarrierTask = std::numeric_limits<TimeType>::min();
        const auto predecessors = syncPoint.first;
        for (const auto& predecessor : predecessors) {
            latestPreBarrierTask = std::max(latestPreBarrierTask, predecessor->getFinishTime(frequenciesSetup));
        }

        TimeType earliestPostBarrierTask = std::numeric_limits<TimeType>::max();
        const auto successors = syncPoint.second;
        for (const auto& successor : successors) {
            earliestPostBarrierTask = std::min(earliestPostBarrierTask, successor->getStartTime(frequenciesSetup));
        }

        const auto shiftEstimation = latestPreBarrierTask - earliestPostBarrierTask;
        syncShiftsEstimations.push_back(shiftEstimation);
        if (extraVerbosity) {
            log.trace("{0}", to_string(syncPoint));
            log.trace(" {0} - {1} = {2}", latestPreBarrierTask, earliestPostBarrierTask, shiftEstimation);
        }
    }
    return syncShiftsEstimations;
}

}  // namespace

// Function return difference as delta = DMA - other, so later we can just add delta to convert from Other(DPU) to
// DMA timer
std::optional<int64_t> vpux::profiling::getDMA2OtherTimersShift(const RawProfilingRecords& dmaTasks,
                                                                const RawProfilingRecords& otherTasks,
                                                                FrequenciesSetup frequenciesSetup,
                                                                SynchronizationPointKind pointKind, vpux::Logger& log) {
    const auto inverseAlgorithm = pointKind == SynchronizationPointKind::DPU_TO_DMA;
    // For some reason in terms of shift estimation DPU2DMA works worse. Probably because of DMA queue starvation.In
    // case of DMA2DPU synchronization DPU tasks executes almost  immediately after barrier, while in opposite case DMA
    // queue should be filled before start
    VPUX_THROW_WHEN(inverseAlgorithm, "DPU2DMA algorithm is disabled");
    using TimeType = RawProfilingRecord::TimeType;

    const auto syncPoints = findSynchronizationPoints(dmaTasks, otherTasks, pointKind);
    if (syncPoints.empty()) {
        log.warning("Cannot find synchronization points for timers shift estimation. Tasks will be aligned on zero.");
        return std::nullopt;
    }
    const auto perBarrirShiftEstimation =
            getBarrierShiftEstimations(syncPoints, frequenciesSetup, log, /*extraVerbosity=*/true);

    std::optional<TimeType> rawTimerShift;
    for (TimeType shiftEstimate : perBarrirShiftEstimation) {
        if (!rawTimerShift.has_value()) {
            rawTimerShift = shiftEstimate;
        }
        if (!inverseAlgorithm) {
            rawTimerShift = std::max(rawTimerShift.value(), shiftEstimate);
        } else {
            rawTimerShift = std::min(rawTimerShift.value(), shiftEstimate);
        }
    }
    if (inverseAlgorithm) {
        rawTimerShift = -rawTimerShift.value();
    }

    const auto timersShift = static_cast<int64_t>(rawTimerShift.value());
    return timersShift;
}

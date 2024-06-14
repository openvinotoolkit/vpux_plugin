//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/reports/tasklist.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/location.hpp"
#include "vpux/utils/profiling/tasknames.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <string>

using vpux::checked_cast;
namespace vpux::profiling {

template <TaskInfo::ExecType Value>
bool isTaskType(const TaskInfo& task) {
    return task.exec_type == Value;
}

TaskList::TaskList() {
}

TaskList::TaskList(const std::vector<TaskInfo>& tasks): std::vector<TaskInfo>(tasks) {
}

template <TaskInfo::ExecType T>
TaskList TaskList::selectTasksOfType() const {
    TaskList selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isTaskType<T>);
    return selectedTasks;
}

TaskList TaskList::selectDPUtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::DPU>();
}

TaskList TaskList::selectUPAtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::UPA>();
}

TaskList TaskList::selectDMAtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::DMA>();
}

TaskList TaskList::selectSWtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::SW>();
}

TaskList TaskList::selectM2Itasks() const {
    return selectTasksOfType<TaskInfo::ExecType::M2I>();
}

TaskList TaskList::getSortedByStartTime() const {
    TaskList sorted(*this);
    sorted.sortByStartTime();
    return sorted;
}

TaskList TaskList::selectClusterLevelTasks() const {
    TaskList selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isClusterLevelProfilingTask);
    return selectedTasks;
}

TaskList TaskList::selectTasksFromCluster(unsigned clusterId) const {
    auto log = Logger::global();
    TaskList selectedTasks;

    for (const auto& task : *this) {
        std::string idStr = getClusterFromName(task.name);
        unsigned id;
        try {
            size_t idx;
            id = std::stoi(idStr, &idx);
            if (idx < idStr.size()) {  // Not all characters converted, ignoring
                log.warning("Not all characters converted while extracting cluster id from task ({0}). Task will "
                            "not be reported.",
                            task.name);
                continue;
            }
        } catch (...) {  // Could not extract cluster id
            log.warning("Could not extract cluster id for task ({0}). Task will not be reported.", task.name);
            continue;
        }

        if (id == clusterId) {
            selectedTasks.push_back(task);
        }
    }
    return selectedTasks;
}

void TaskList::sortByStartTime() {
    std::sort(begin(), end(), profilingTaskStartTimeComparator<TaskInfo>);
}

unsigned TaskList::getClusterCount() const {
    std::set<std::string> clusterLevelThreadNames;
    for (const auto& task : *this) {
        clusterLevelThreadNames.insert(getClusterFromName(task.name));
    }
    return checked_cast<unsigned>(clusterLevelThreadNames.size());
}

int TaskList::getSumOfDurations() const {
    return std::accumulate(begin(), end(), 0, [](const int& totalTime, const TaskInfo& task) {
        return totalTime + task.duration_ns;
    });
}

int TaskList::getStartTime() const {
    VPUX_THROW_WHEN(empty(), "Minimal time in empty TaskList is not defined.");

    auto minElementIt = min_element(begin(), end(), [](const TaskInfo& a, const TaskInfo& b) {
        return a.start_time_ns < b.start_time_ns;
    });
    return checked_cast<int>(minElementIt->start_time_ns);
}

int TaskList::getEndTime() const {
    VPUX_THROW_WHEN(empty(), "Maximal time in empty TaskList is not defined.");

    auto maxElementIt = max_element(begin(), end(), [](const TaskInfo& a, const TaskInfo& b) {
        return a.start_time_ns + a.duration_ns < b.start_time_ns + b.duration_ns;
    });
    return checked_cast<int>(maxElementIt->start_time_ns + maxElementIt->duration_ns);
}

int TaskList::getTotalDuration() const {
    if (empty()) {
        return 0;
    }
    return getEndTime() - getStartTime();
}

TaskList& TaskList::append(const TaskList& tasks) {
    insert(end(), tasks.begin(), tasks.end());
    return *this;
}

bool isVariantLevelProfilingTask(const TaskInfo& task) {
    const std::string variantSuffix = vpux::LOCATION_SEPARATOR + VARIANT_LEVEL_PROFILING_SUFFIX + "_";
    bool hasVariantInName = getTaskNameSuffixes(task.name).find(variantSuffix) != std::string::npos;
    return hasVariantInName;
}

bool isClusterLevelProfilingTask(const TaskInfo& task) {
    const std::string clusterSuffix = vpux::LOCATION_SEPARATOR + CLUSTER_LEVEL_PROFILING_SUFFIX + "_";
    bool hasClusterInName = getTaskNameSuffixes(task.name).find(clusterSuffix) != std::string::npos;
    return hasClusterInName && !isVariantLevelProfilingTask(task);
}

}  // namespace vpux::profiling

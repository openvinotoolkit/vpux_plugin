//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/reports/api.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/reports/stats.hpp"
#include "vpux/utils/profiling/reports/tasklist.hpp"
#include "vpux/utils/profiling/reports/ted.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"
#include "vpux/utils/profiling/tasknames.hpp"

#include <exception>
#include <iomanip>
#include <ostream>
#include <vector>

namespace vpux::profiling {

namespace {

/**
 * @brief Helper class to calculate placement of profiling tasks in
 * an optimal number of Perfetto UI threads
 *
 * Stores tasks end times for each thread.
 */
class TraceEventTimeOrderedDistribution {
public:
    /**
     * @brief Get the event thread Id assuring a non-overlapping placement among other existing tasks on the same
     * thread.
     *
     * Calls to this function assume that tasks were sorted in ascending order by taskStartTime
     * This function updates the state of object with the taskEndTime in the thread the task was assigned to.
     *
     * @return int - calculate thread id unique to the current process
     */
    int getThreadId(double taskStartTime, double duration);

    /// Return current number of threads
    unsigned size() const {
        return _lastTimestamps.size();
    }

private:
    std::vector<double> _lastTimestamps;
};

int TraceEventTimeOrderedDistribution::getThreadId(double taskStartTime, double duration) {
    double taskEndTime = taskStartTime + duration;
    for (size_t i = 0; i < _lastTimestamps.size(); ++i) {
        if (_lastTimestamps[i] <= taskStartTime) {
            _lastTimestamps[i] = taskEndTime;
            return i;
        }
    }
    _lastTimestamps.push_back(taskEndTime);
    return _lastTimestamps.size() - 1;
}

class TraceEventExporter {
public:
    TraceEventExporter(std::ostream& outStream, Logger& log);
    ~TraceEventExporter() noexcept(false);

    /**
     * @brief flush queued trace events to output stream.
     */
    void flushAsTraceEvents();

    void processTasks(const std::vector<TaskInfo>& tasks);
    void processLayers(const std::vector<LayerInfo>& layers);

private:
    TraceEventExporter(const TraceEventExporter&) = delete;
    TraceEventExporter& operator=(const TraceEventExporter&) = delete;

    /**
     * @brief helper function to ease exporting profiled tasks to JSON format
     *
     * @param tasks - list of tasks to be exported
     * @param processName - trace event process name
     * @param createNewProcess - if true, meta data about the current trace event process and threads assigned to the
     * tasks being processed are exported as meta trace events
     *
     * The function schedules tasks for output to out stream and generates meta type header trace events.
     * It internally manages trace events' thread IDs and names.
     */
    void processTraceEvents(const TaskList& tasks, const std::string& processName, bool createNewProcess = true);

    /**
     * @brief set tracing event process name for given process id.
     *
     * @param suffixStr - suffix added at the end of line. Except for the last tracing event, JSON events are separated
     * by commas
     * @param processId - trace event process identifier
     * @param suffixStr - end of line string for the process name trace event
     */
    void setTraceEventProcessName(const std::string& processName, int processId, const std::string& suffixStr = ",");

    void setTraceEventThreadName(const std::string& threadName, int threadId, int processId,
                                 const std::string& suffixStr = ",");

    /**
     * @brief Set the Tracing Event Process Sort Index
     *
     * @param processId - trace event process identifier
     * @param sortIndex - index defining the process ordering in the output report. (Some UIs do not respect this value)
     * @param suffixStr - suffix added at the end of line. Except for the last tracing event, JSON events are separated
     * by
     */
    void setTraceEventProcessSortIndex(int processId, unsigned sortIndex, const std::string& suffixStr = ",");

    /**
     * @brief Perform basic sanity checks on task name and duration
     *
     * @param task - task to check
     *
     * For reporting tasks it is assumed that all tasks should have
     * task name format compliant with:
     *
     *  originalLayerName?t_layerType/suffix1_value1/suffix2_value2...
     *
     * This method checks for existence of suffix separator (?) in task name and
     * asserts cluster_id suffix exists for the relevant task types.
     *
     * Warning is issued if task duration is not a positive integer.
     */
    void validateTaskNameAndDuration(const TaskInfo& task) const;

    std::vector<TraceEventDesc> _events;
    std::ostream& _outStream;
    Logger _log;
    int _processId = -1;
    int _threadId = -1;
};

constexpr auto CLUSTER_PROCESS_NAME = "Cluster";
constexpr auto DMA_PROCESS_NAME = "DMA";
constexpr auto LAYER_PROCESS_NAME = "Layers";
constexpr auto UPA_PROCESS_NAME = "UPA";
constexpr auto M2I_PROCESS_NAME = "M2I";

constexpr auto VARIANT_NAME = "Variants";
constexpr auto SHAVE_NAME = "Shave";
constexpr auto LAYER_THREAD_NAME = "Layers";

constexpr auto DMA_TASK_CATEGORY = "DMA";
constexpr auto DPU_TASK_CATEGORY = "DPU";
constexpr auto NONE_TASK_CATEGORY = "NONE";
constexpr auto SW_TASK_CATEGORY = "SW";
constexpr auto UPA_TASK_CATEGORY = "UPA";
constexpr auto M2I_TASK_CATEGORY = "M2I";

static const std::map<TaskInfo::ExecType, std::string> enumToStr = {{TaskInfo::ExecType::NONE, NONE_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::DPU, DPU_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::SW, SW_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::DMA, DMA_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::UPA, UPA_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::M2I, M2I_TASK_CATEGORY}

};

std::string getTraceEventThreadName(const TaskInfo& task) {
    switch (task.exec_type) {
    case TaskInfo::ExecType::DMA:
        return DMA_TASK_CATEGORY;
    case TaskInfo::ExecType::UPA:
        return SW_TASK_CATEGORY;  // we use SW task labels for UPA threads
    case TaskInfo::ExecType::SW:
        return std::string(SW_TASK_CATEGORY) + " / " + SHAVE_NAME;
    case TaskInfo::ExecType::DPU:
        return isVariantLevelProfilingTask(task) ? std::string(DPU_TASK_CATEGORY) + " / " + VARIANT_NAME
                                                 : DPU_TASK_CATEGORY;
    case TaskInfo::ExecType::M2I:
        return M2I_TASK_CATEGORY;
    default:
        VPUX_THROW("Unknown task category");
    }
}

void TraceEventExporter::processTasks(const std::vector<TaskInfo>& tasks) {
    for (auto& task : tasks) {
        validateTaskNameAndDuration(task);
    }

    //
    // Export DMA tasks
    //
    auto dmaTasks = TaskList(tasks).selectDMAtasks();
    processTraceEvents(dmaTasks, DMA_PROCESS_NAME, /* createNewProcess= */ true);

    //
    // Export cluster tasks (DPU and SW)
    //
    unsigned clusterCount = TaskList(tasks).getClusterCount();

    TaskList dpuTasks = TaskList(tasks).selectDPUtasks();
    TaskList swTasks = TaskList(tasks).selectSWtasks();

    for (unsigned clusterId = 0; clusterId < clusterCount; clusterId++) {
        std::string processName = std::string(CLUSTER_PROCESS_NAME) + " (" + std::to_string(clusterId) + ")";
        auto clusterDpuTasks = dpuTasks.selectTasksFromCluster(clusterId);
        processTraceEvents(clusterDpuTasks, processName, /* createNewProcess= */ true);
        auto clusterSwTasks = swTasks.selectTasksFromCluster(clusterId);
        processTraceEvents(clusterSwTasks, processName, /* createNewProcess= */ clusterDpuTasks.empty());
    }

    //
    // Export non-clustered SW tasks into separate UPA process
    //
    TaskList upaTasks = TaskList(tasks).selectUPAtasks();
    processTraceEvents(upaTasks, UPA_PROCESS_NAME, /* createNewProcess= */ true);

    VPUX_THROW_WHEN(!upaTasks.empty() && !swTasks.empty(),
                    "UPA and Shave tasks should be mutually exclusive but are found to coexist");

    TaskList m2iTasks = TaskList(tasks).selectM2Itasks();
    processTraceEvents(m2iTasks, M2I_PROCESS_NAME, /* createNewProcess= */ true);
}

void TraceEventExporter::processLayers(const std::vector<LayerInfo>& layers) {
    if (layers.empty()) {
        return;
    }

    ++_processId;
    _threadId = 0;

    TraceEventTimeOrderedDistribution layersDistr;
    for (auto& layer : layers) {
        TraceEventDesc ted;
        ted.name = layer.name;
        ted.category = "Layer";
        ted.pid = _processId;
        ted.tid = layersDistr.getThreadId(layer.start_time_ns, layer.duration_ns);
        // use ns-resolution integers to avoid round-off errors during fixed precision output to JSON
        ted.timestamp = layer.start_time_ns / 1000.;
        ted.duration = layer.duration_ns / 1000.;
        ted.customArgs.push_back({"Layer type", layer.layer_type});
        _events.push_back(ted);
    }

    setTraceEventProcessName(LAYER_PROCESS_NAME, _processId);
    setTraceEventProcessSortIndex(_processId, _processId);
    for (unsigned threadId = 0; threadId < layersDistr.size(); ++threadId) {
        setTraceEventThreadName(LAYER_THREAD_NAME, threadId, _processId);
    }
}

void TraceEventExporter::validateTaskNameAndDuration(const TaskInfo& task) const {
    // check existence of cluster_id suffix in clustered tasks
    if (task.exec_type == TaskInfo::ExecType::SW || task.exec_type == TaskInfo::ExecType::DPU) {
        bool hasClusterInName = !getClusterFromName(task.name).empty();
        VPUX_THROW_UNLESS(hasClusterInName, "Task {0} does not have assigned cluster_id", task.name);
    }

    // check task duration
    if (task.duration_ns <= 0) {
        _log.warning("Task {0} has duration {1} ns.", task.name, task.duration_ns);
    }
}

void TraceEventExporter::processTraceEvents(const TaskList& tasks, const std::string& processName,
                                            bool createNewProcess) {
    if (tasks.empty()) {  // don't need to output process details if there are no tasks to export
        return;
    }

    if (createNewProcess) {
        ++_processId;
        _threadId = -1;
        setTraceEventProcessName(processName, _processId);
        setTraceEventProcessSortIndex(_processId, _processId);
    }

    int lastThreadId = _threadId;
    ++_threadId;

    TraceEventTimeOrderedDistribution threadDistr;

    auto sortedTasks = tasks.getSortedByStartTime();
    for (const auto& task : sortedTasks) {
        auto thId = _threadId + threadDistr.getThreadId(task.start_time_ns, task.duration_ns);
        if (thId > lastThreadId) {
            setTraceEventThreadName(getTraceEventThreadName(task), thId, _processId);
            lastThreadId = thId;
        }

        TraceEventDesc ted;
        ted.name = task.name;
        ted.category = enumToStr.at(task.exec_type);
        ted.pid = _processId;
        ted.tid = thId;
        // use ns-resolution integers to avoid round-off errors during fixed precision output to JSON
        ted.timestamp = task.start_time_ns / 1000.;
        ted.duration = task.duration_ns / 1000.;
        _events.push_back(ted);
    }

    _threadId = lastThreadId;
}

TraceEventExporter::TraceEventExporter(std::ostream& outStream, Logger& log): _outStream(outStream), _log(log) {
    // Trace Events timestamps are in microseconds, set precision to preserve nanosecond resolution
    _outStream << std::setprecision(3) << "{\"traceEvents\":[" << std::endl;
}

TraceEventExporter::~TraceEventExporter() noexcept(false) {
    if (std::uncaught_exceptions() > 0) {
        // Got bigger probelm than closing the report so ignore the errors from here
        _outStream.exceptions(std::ios::goodbit);
    }
    // Hint for a classic Perfetto UI to use nanoseconds for display
    // JSON timestamps are expected to be in microseconds regardless
    _outStream << "\"displayTimeUnit\": \"ns\"\n"
               << "}\n";
    _outStream.flush();
}

void TraceEventExporter::flushAsTraceEvents() {
    if (!_events.empty()) {
        for (auto tedIt = _events.begin(); tedIt != std::prev(_events.end()); ++tedIt) {
            _outStream << *tedIt << ",\n";
        }
        _outStream << _events.back() << std::endl;
        _events.clear();
    }
    // close traceEvents block
    _outStream << "],\n";
    _outStream.flush();
}

void TraceEventExporter::setTraceEventProcessName(const std::string& processName, int processId,
                                                  const std::string& suffixStr) {
    _outStream << std::string(R"({"name": "process_name", "ph": "M", "pid":)") << processId
               << R"(, "args": {"name" : ")" << processName << R"("}})" << suffixStr << std::endl;
}

void TraceEventExporter::setTraceEventThreadName(const std::string& threadName, int threadId, int processId,
                                                 const std::string& suffixStr) {
    _outStream << std::string(R"({"name": "thread_name", "ph": "M", "pid":)") << processId << R"(, "tid":)" << threadId
               << R"(, "args": {"name" : ")" << threadName << R"("}})" << suffixStr << std::endl;
}

void TraceEventExporter::setTraceEventProcessSortIndex(int processId, unsigned sortIndex,
                                                       const std::string& suffixStr) {
    _outStream << std::string(R"({"name": "process_sort_index", "ph": "M", "pid":)") << processId
               << R"(, "args": {"sort_index" : ")" << sortIndex << R"("}})" << suffixStr << std::endl;
}

}  // namespace

void printProfilingAsTraceEvent(const std::vector<TaskInfo>& tasks, const std::vector<LayerInfo>& layers,
                                std::ostream& output, Logger& log) {
    TaskStatistics stats(tasks);

    {
        TraceEventExporter events(output, log);
        events.processTasks(tasks);
        events.processLayers(layers);
        events.flushAsTraceEvents();
        stats.printAsJson(output);
    }

    stats.log(log);
}

std::ostream& operator<<(std::ostream& os, const TraceEventDesc& event) {
    std::ios::fmtflags origFlags(os.flags());
    os << std::fixed << "{\"name\":\"" << event.name << "\", \"cat\":\"" << event.category << "\", \"ph\":\"X\", "
       << "\"ts\":" << event.timestamp << ", \"dur\":" << event.duration << ", \"pid\":" << event.pid
       << ", \"tid\":" << event.tid;
    if (!event.customArgs.empty()) {
        os << ", \"args\":{";
        bool isFirst = true;
        for (const auto& arg : event.customArgs) {
            os << (isFirst ? "" : ", ") << "\"" << arg.first << "\": \"" << arg.second << "\"";
            isFirst = false;
        }
        os << "}";
    }
    os << "}";
    os.flags(origFlags);
    return os;
}

}  // namespace vpux::profiling

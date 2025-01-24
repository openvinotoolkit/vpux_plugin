#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace {

constexpr uint32_t NPU_DEFAULT_INVARIANT_COUNT = 64;
constexpr uint32_t NPU_DEFAULT_VARIANT_COUNT = 128;
constexpr uint32_t NPU_DEFAULT_KERNEL_RANGE_COUNT = 64;
constexpr uint32_t NPU_DEFAULT_KERNEL_INVO_COUNT = 64;
constexpr uint32_t NPU_DEFAULT_MEDIA_COUNT = 4;
constexpr uint32_t NPU_DEFAULT_DMA_TASK_COUNT = 80;

struct TaskListKey {
    VPU::ArchKind archKind;
    VPURegMapped::TaskType taskType;
    bool operator==(const TaskListKey& other) const {
        return (archKind == other.archKind && taskType == other.taskType);
    }
};
struct TaskListKeyHash {
    std::size_t operator()(const TaskListKey& key) const noexcept {
        auto hashTask = std::hash<VPURegMapped::TaskType>{}(key.taskType);
        auto hashArch = std::hash<VPU::ArchKind>{}(key.archKind);
        // make sure the hash function is good enough for minimizing collision occurence (same output
        // for different key values)
        return hashTask ^ (hashArch << 3);
    }
};

const std::unordered_map<TaskListKey, uint32_t, TaskListKeyHash> taskListsDefaultCapacityMap = {
        {{VPU::ArchKind::NPU40XX, VPURegMapped::TaskType::DPUInvariant}, NPU_DEFAULT_INVARIANT_COUNT},
        {{VPU::ArchKind::NPU40XX, VPURegMapped::TaskType::DPUVariant}, NPU_DEFAULT_VARIANT_COUNT},
        {{VPU::ArchKind::NPU40XX, VPURegMapped::TaskType::ActKernelInvocation}, NPU_DEFAULT_KERNEL_INVO_COUNT},
        {{VPU::ArchKind::NPU40XX, VPURegMapped::TaskType::ActKernelRange}, NPU_DEFAULT_KERNEL_RANGE_COUNT},
        {{VPU::ArchKind::NPU40XX, VPURegMapped::TaskType::M2I}, NPU_DEFAULT_MEDIA_COUNT},
        {{VPU::ArchKind::NPU40XX, VPURegMapped::TaskType::DMA}, NPU_DEFAULT_DMA_TASK_COUNT}};

}  // namespace

namespace vpux {
namespace VPURegMapped {

size_t calcMinBitsRequirement(uint64_t value) {
    if (value == 0) {
        return 1;
    }
    if (value == std::numeric_limits<uint64_t>::max()) {
        return sizeof(uint64_t) * CHAR_BIT;
    }
    return checked_cast<size_t>(std::ceil(log2(value + 1)));
}

void updateRegMappedInitializationValues(std::map<std::string, std::map<std::string, RegFieldValue>>& values,
                                         const std::map<std::string, std::map<std::string, RegFieldValue>>& newValues) {
    for (auto newRegisterIter = newValues.begin(); newRegisterIter != newValues.end(); ++newRegisterIter) {
        auto correspondingRegisterIter = values.find(newRegisterIter->first);
        VPUX_THROW_UNLESS(correspondingRegisterIter != values.end(),
                          "updateRegMappedInitializationValues: Register with name {0} not found in provided values",
                          newRegisterIter->first);

        for (auto newFieldIter = newRegisterIter->second.begin(); newFieldIter != newRegisterIter->second.end();
             ++newFieldIter) {
            auto correspondingFieldIter = correspondingRegisterIter->second.find(newFieldIter->first);
            VPUX_THROW_UNLESS(correspondingFieldIter != correspondingRegisterIter->second.end(),
                              "updateRegMappedInitializationValues: Field with name {0} not found in provided values",
                              newFieldIter->first);

            // update field value
            correspondingFieldIter->second = newFieldIter->second;
        }
    }
}

std::optional<TaskBufferLayoutOp> getTaskBufferLayoutOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op->hasTrait<mlir::OpTrait::OneRegion>(),
                      "VPURegMapped::TaskBufferLayoutOp should only exist under OneRegion-type ops");

    auto taskBufferOpsRange = op->getRegion(0).getOps<TaskBufferLayoutOp>();
    if (taskBufferOpsRange.empty()) {
        return std::nullopt;
    }

    VPUX_THROW_WHEN(std::distance(taskBufferOpsRange.begin(), taskBufferOpsRange.end()) > 1,
                    "Only one VPURegMapped::TaskBufferLayoutOp should exist");
    return *(taskBufferOpsRange.begin());
}

uint32_t getDefaultTaskListCount(VPURegMapped::TaskType taskType, VPU::ArchKind archKind) {
    auto taskListCapacityIter = taskListsDefaultCapacityMap.find({archKind, taskType});
    VPUX_THROW_WHEN(taskListCapacityIter == taskListsDefaultCapacityMap.end(),
                    "getDefaultTaskListCount: Unknown task type {0} for arch {1}", taskType, archKind);

    return taskListCapacityIter->second;
}

}  // namespace VPURegMapped
}  // namespace vpux

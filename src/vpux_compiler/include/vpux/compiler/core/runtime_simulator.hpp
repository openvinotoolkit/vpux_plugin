//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/compiler/core/token_barrier_scheduler.hpp"

#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"

namespace vpux {

static constexpr int64_t MAX_DMA_ENGINES = 2;
static constexpr StringLiteral virtualIdAttrName = "virtualId";

class RuntimeSimulator final {
public:
    // struct TaskInfo {
    //     VPURT::TaskOp taskOp;
    //     SmallVector<int64_t> waitBarriers;
    //     SmallVector<int64_t> updateBarriers;

    //     TaskInfo() {
    //     }
    //     TaskInfo(VPURT::TaskOp taskOp): taskOp(taskOp) {
    //     }
    // };

    struct VirtualBarrierInfo {
        int64_t realId;
        int64_t producerCount;
        int64_t consumerCount;
        int64_t initProducerCount;
        int64_t initConsumerCount;

        VirtualBarrierInfo(): realId(), producerCount(), consumerCount(), initProducerCount(), initConsumerCount() {
        }
    };

    RuntimeSimulator(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log, int64_t numDmaEngines,
                     size_t numRealBarriers);
    void init();
    bool assignPhysicalIDs();
    void buildTaskLists();
    void acquireRealBarrier(VPURT::DeclareVirtualBarrierOp btask);
    bool isTaskReady(VPURT::TaskOp taskOp);
    void processTask(VPURT::TaskOp task);
    void returnRealBarrier(mlir::Operation* btask);
    void getAllBarriersProducersAndConsumers();
    void computeOpIndegree();
    void computeOpOutdegree();
    std::pair<int64_t, int64_t> getID(mlir::Operation* val) const;
    bool processTasks(std::vector<VPURT::TaskOp>& dma_task_list);
    bool fillBarrierTasks(std::list<VPURT::DeclareVirtualBarrierOp>& barrier_task_list);
    int64_t getVirtualId(VPURT::ConfigureBarrierOp op);

private:
    std::vector<VPURT::TaskOp> _nceTasks;
    std::vector<VPURT::TaskOp> _upaTasks;
    std::array<std::vector<VPURT::TaskOp>, MAX_DMA_ENGINES> _dmaTasks;
    std::list<VPURT::DeclareVirtualBarrierOp> _barrierOps;
    std::map<mlir::Operation*, std::pair<int64_t, int64_t>> _virtualToPhysicalBarrierMap;

    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;
    Logger _log;
    int64_t _numDmaEngines;
    size_t _numRealBarriers;

    struct active_barrier_info_t {
        size_t real_barrier_;
        size_t in_degree_;
        size_t out_degree_;
        active_barrier_info_t(size_t real, size_t in, size_t out)
                : real_barrier_(real), in_degree_(in), out_degree_(out) {
        }
    };

    typedef std::unordered_map<mlir::Operation*, active_barrier_info_t> active_barrier_table_t;
    typedef active_barrier_table_t::iterator active_barrier_table_iterator_t;
    typedef std::list<VPURT::DeclareVirtualBarrierOp>::iterator barrier_list_iterator_t;
    typedef std::vector<VPURT::TaskOp>::iterator taskInfo_iterator_t;
    typedef std::unordered_map<mlir::Operation*, size_t> in_degree_map_t;
    typedef std::unordered_map<mlir::Operation*, size_t> out_degree_map_t;

    active_barrier_table_t _active_barrier_table;
    std::list<size_t> _real_barrier_list;
    in_degree_map_t in_degree_map_;
    out_degree_map_t out_degree_map_;
    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>
            _configureTaskOpUpdateWaitMap;

    std::unordered_map<mlir::Operation*, SmallVector<mlir::Operation*>> _barrierProducersMap{};
    std::unordered_map<mlir::Operation*, SmallVector<mlir::Operation*>> _barrierConsumersMap{};
};

}  // namespace vpux

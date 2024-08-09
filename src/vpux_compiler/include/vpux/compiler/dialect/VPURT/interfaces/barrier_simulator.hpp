//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/ring_buffer.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_set>

namespace vpux {
namespace VPURT {

static constexpr int64_t MAX_DMA_ENGINES = 2;

struct BarrierConfig final {
    mlir::Location loc;

    int64_t realId = -1;
    int64_t nextSameId = -1;
    int64_t producerCount = 0;
    int64_t consumerCount = 0;

    explicit BarrierConfig(mlir::Location loc, int64_t realId = -1): loc(loc), realId(realId) {
    }
};

struct BarrierUserConfig final {
    std::unordered_set<int64_t> waits;
    std::unordered_set<int64_t> posts;
    int64_t startAfter = -1;
    int64_t cleanAfter = -1;
    int64_t virtualDep = -1;
    int64_t count = 0;

    BarrierUserConfig() = default;

    explicit BarrierUserConfig(int64_t virtualDep, int64_t count = 1): virtualDep(virtualDep), count(count) {
    }
};

//
// VirtualDependencyTracker
//

class VirtualDependencyTracker final {
public:
    using Range = std::pair<int64_t, int64_t>;

    struct Dependency final {
        Range consumer;
        Range producer;
    };

public:
    VirtualDependencyTracker(): _deps(1) {
    }

public:
    int64_t add(VPURT::TaskOp taskOp);
    int64_t add(ArrayRef<int64_t> waits, ArrayRef<int64_t> posts);

    int64_t clone(int64_t i);

public:
    bool validId(int64_t i) const {
        return i >= 0 && checked_cast<size_t>(i) < _ids.size();
    }
    bool validDep(int64_t i) const {
        return i >= 0 && checked_cast<size_t>(i) < _deps.size();
    }

    int64_t id(int64_t i) const {
        assert(validId(i));
        return _ids[checked_cast<size_t>(i)];
    }
    int64_t& id(int64_t i) {
        assert(validId(i));
        return _ids[checked_cast<size_t>(i)];
    }

    const Dependency& dep(int64_t i) const {
        assert(validDep(i));
        return _deps[checked_cast<size_t>(i)];
    }
    Dependency& dep(int64_t i) {
        assert(validDep(i));
        return _deps[checked_cast<size_t>(i)];
    }

public:
    void print(Logger log) const;

private:
    SmallVector<int64_t> _ids;
    SmallVector<Dependency> _deps;
};

//
// BarrierSimulator
//

class BarrierSimulator final {
public:
    explicit BarrierSimulator(mlir::func::FuncOp funcOp, bool wlm = false);

public:
    bool isDynamicBarriers() const {
        return _isDynamicBarriers;
    }

public:
    const BarrierConfig& getConfig(mlir::Value bar) const;

public:
    mlir::LogicalResult checkProducerCount(Logger log) const;
    mlir::LogicalResult checkProducerAndConsumerCount(Logger log) const;
    mlir::LogicalResult simulateBarriers(Logger log, std::optional<int64_t> numBarriers = std::nullopt,
                                         std::optional<bool> barrierLegalization = std::nullopt);
    SmallVector<mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>> getBarrierBatchesToLegalize();
    void linkNextIds(Logger log);

private:
    void parseBarriers(mlir::Operation* parentOp);
    void parseTasks(mlir::Operation* parentOp);
    void buildBarrierControlMap(mlir::func::FuncOp funcOp);

private:
    enum class Status { Success, Skip, Fail };

    Status processSim(const VirtualDependencyTracker::Dependency& dep, BarrierUserConfig& user, int64_t count,
                      StringRef taskType, int64_t index, SmallVectorImpl<int64_t>& toVirtual,
                      RingBuffer<int64_t>& nextReal, Logger log);

private:
    // DMA task index {queue index, task index in queue}
    using DmaTaskIdx = std::pair<int64_t, size_t>;

    int64_t _barrierProducerSlotCount = 0;
    int64_t _barrierTotalSlotCount = 0;
    int64_t _availableBarriers = 0;
    int64_t _usedBarriers = 0;

    DenseMap<mlir::Operation*, size_t> _barriersMap;
    SmallVector<BarrierConfig> _barriers;
    bool _isDynamicBarriers = false;
    bool _samePidTopoDep = false;

    SmallVector<SmallVector<BarrierUserConfig>> _dmaTasks;
    std::unordered_map<SmallVector<DmaTaskIdx>, bool> _multiQueueDmaTaskStatus;
    SmallVector<BarrierUserConfig> _nceTasks;
    SmallVector<BarrierUserConfig> _actTasks;
    SmallVector<BarrierUserConfig> _upaTasks;
    SmallVector<BarrierUserConfig> _m2iTasks;

    VirtualDependencyTracker _vdt;

    // barrier legalisation
    SmallVector<std::set<int64_t>> _barrierBatchesToLegalize;
    DenseMap<size_t, mlir::Operation*> _barrierVID;

    SmallVector<llvm::BitVector> _barrierControlMap;
};

}  // namespace VPURT
}  // namespace vpux

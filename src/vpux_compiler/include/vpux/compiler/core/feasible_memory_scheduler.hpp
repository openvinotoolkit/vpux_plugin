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

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseSet.h>

namespace vpux {

class FeasibleMemoryScheduler final {
public:
    using operationIdxType = size_t;
    // Operation type
    enum class EOpType { ORIGINAL_OP = 0, IMPLICIT_OP_READ = 1, IMPLICIT_OP_WRITE = 2 };
    // Operation state
    enum class EOpState { ACTIVE = 0, SPILLED = 1, CONSUMED = 2 };
    // Core struct in the feasible memory scheduler
    struct HeapElement {
        HeapElement(): op_(), time_(), opType_() {
        }
        HeapElement(operationIdxType op, size_t time = 0UL, EOpType op_type = EOpType::ORIGINAL_OP)
                : op_(op), time_(time), opType_(op_type) {
        }
        bool operator==(const HeapElement& other) const {
            return (op_ == other.op_) && (time_ == other.time_);
        }
        bool isOriginalOp() const {
            return (opType_ == EOpType::ORIGINAL_OP);
        }
        bool isImplicitWriteOp() const {
            return (opType_ == EOpType::IMPLICIT_OP_WRITE);
        }
        operationIdxType op_;
        size_t time_;
        EOpType opType_;
    };
    // Sort heap by smallest time
    struct MinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) {
            return a.time_ > b.time_;
        }
    };
    // Sort pair by the second arg
    struct SizeSort {
        bool operator()(const std::pair<operationIdxType, vpux::AddressType>& op1,
                        const std::pair<operationIdxType, vpux::AddressType>& op2) const {
            if (op1.second == op2.second) {
                return op1.first < op2.first;
            }

            return op1.second > op2.second;
        }
    };
    // Struct used during scheduling, containing op info
    struct OpOutputInfo {
        OpOutputInfo(EOpState state = EOpState::CONSUMED, size_t outstanding_consumers = 0UL)
                : state_(state), outstandingConsumers_(outstanding_consumers) {
        }
        bool active() const {
            return state_ == EOpState::ACTIVE;
        }
        bool spilled() const {
            return state_ == EOpState::SPILLED;
        }
        bool consumed() const {
            return state_ == EOpState::CONSUMED;
        }
        bool hasSingleOutstandingConsumer() const {
            return outstandingConsumers_ == 1UL;
        }
        void changeStateToActive() {
            state_ = EOpState::ACTIVE;
        }
        void changeStateToConsumed() {
            state_ = EOpState::CONSUMED;
        }
        void changeStateToSpilled() {
            state_ = EOpState::SPILLED;
        }
        void decrementConsumers() {
            VPUX_THROW_UNLESS(outstandingConsumers_ > 0UL, "Invalid number of consumers");
            --outstandingConsumers_;
            if (!outstandingConsumers_) {
                state_ = EOpState::CONSUMED;
            }
        }
        void incrementConsumers() {
            if (!outstandingConsumers_) {
                state_ = EOpState::SPILLED;
            }
            ++outstandingConsumers_;
        }
        EOpState state_;
        size_t outstandingConsumers_;
    };
    // Struct storing CMX address space
    struct IntervalInfo {
        void invalidate() {
            begin_ = std::numeric_limits<size_t>::max();
            end_ = std::numeric_limits<size_t>::min();
        }
        size_t length() const {
            VPUX_THROW_UNLESS(begin_ <= end_, "Invalid resource interval");
            return end_ - begin_ + 1;
        }
        IntervalInfo(): begin_(), end_() {
            invalidate();
        }
        IntervalInfo(size_t ibeg, size_t iend): begin_(ibeg), end_(iend) {
        }
        bool operator==(const IntervalInfo& other) const {
            return (begin_ == other.begin_) && (end_ == other.end_);
        }
        size_t begin_;
        size_t end_;
    };
    // Struct used to output the scheduled op info
    struct ScheduledOpInfo {
        ScheduledOpInfo(operationIdxType op, EOpType type, size_t time): op_(op), opType_(type), time_(time) {
        }
        ScheduledOpInfo(): op_(), opType_(), time_() {
        }
        bool operator==(const ScheduledOpInfo& other) const {
            return (other.op_ == op_) && (other.opType_ == opType_);
        }
        const ScheduledOpInfo& operator=(const HeapElement& helement) {
            op_ = helement.op_;
            opType_ = helement.opType_;
            return *this;
        }
        StringLiteral opTypeName() const {
            if (opType_ == EOpType::ORIGINAL_OP) {
                return StringLiteral("ORIGINAL");
            } else if (opType_ == EOpType::IMPLICIT_OP_READ) {
                return StringLiteral("SPILLED_READ");
            } else if (opType_ == EOpType::IMPLICIT_OP_WRITE) {
                return StringLiteral("SPILLED_WRITE");
            }
            return StringLiteral("UNDEFINED");
        }
        bool hasActiveResource() const {
            return (resourceInfo_.begin_ <= resourceInfo_.end_);
        }
        size_t beginResource() const {
            return resourceInfo_.begin_;
        }
        size_t endResource() const {
            return resourceInfo_.end_;
        }
        operationIdxType op_;
        EOpType opType_;
        size_t time_;
        IntervalInfo resourceInfo_;
    };

public:
    explicit FeasibleMemoryScheduler(mlir::Attribute& memSpace, MemLiveRangeInfo& liveRangeInfo,
                                     AsyncDepsInfo& depsInfo, Logger& log,
                                     LinearScan<mlir::Value, LinearScanHandler>& scan);

public:
    llvm::SmallVector<ScheduledOpInfo> generateSchedule();

private:
    bool init();
    void clearLists();
    void nextSchedulableOp();
    void getReadyDataList();
    void getReadyComputeList();
    llvm::SmallVector<operationIdxType> reduceInDegreeOfAdjacentOperations(operationIdxType opIdx);
    bool isReadyComputeOperationSchedulable(operationIdxType opIdx);
    SmallVector<mlir::Value> getSortedBuffers(operationIdxType opIdx);
    SmallVector<operationIdxType> getNonEmptyOpDemandList(operationIdxType opIdx);
    void scheduleInputOpForComputeOp(operationIdxType inputIdx);
    void allocateSortedBuffers(ArrayRef<mlir::Value> sortedBuffers);
    void scheduleComputeOp(operationIdxType opIdx);
    void scheduleAllPossibleReadyOpsAndUpdate(
            std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort>& readyList);
    void pushToStartTimeHeap(const HeapElement& elem);
    void pushToCompletionTimeHeap(const HeapElement& elem);
    HeapElement popFromStartTimeHeap();
    HeapElement popFromCompletionTimeHeap();
    HeapElement const* topElementGen(const llvm::SmallVector<HeapElement>& heap) const;
    bool isDataOp(operationIdxType opIdx);
    void unscheduleOp(const HeapElement& helement);
    bool isComputeOpWithSomeActiveInputs(operationIdxType opIdx);
    void distributeReadyOps(llvm::ArrayRef<operationIdxType> readyOps);
    llvm::SmallVector<HeapElement> popAllElementsAtThisTime(size_t time_step);
    void unscheduleAllCompletingOpsAtNextEarliestTime();
    void populateScheduledOps(HeapElement& scheduledOp);
    vpux::AddressType calculateOpSize(operationIdxType opIdx);
    void evictActiveOp(operationIdxType opIdx, mlir::Value* buffer);
    size_t evictionPriority(mlir::async::ExecuteOp& op);
    mlir::Value* chooseCandidateForEviction(llvm::SmallVector<mlir::Value*> orderedBuffers);
    mlir::async::ExecuteOp retrieveBufferOwner(mlir::Value* buffer);
    void forceScheduleActiveOpEviction();

private:
    Logger& _log;
    // CMX mem space, which will be allocated
    mlir::Attribute& _memSpace;
    // information about op buffers
    MemLiveRangeInfo& _liveRangeInfo;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    // heap with earliest operation start time
    llvm::SmallVector<HeapElement> _startTimeHeap;
    // heap with earlies operation completion time
    llvm::SmallVector<HeapElement> _completionTimeHeap;
    // operations with ACTIVE input
    std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort> _activeComputeOps;
    // compute operations with 0 in-degree
    std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort> _readyComputeOps;
    // data operations with 0 in-degree
    std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort> _readyDataOps;
    // operation in-degree, number of incoming edges
    std::unordered_map<operationIdxType, size_t> _inDegreeTable;
    // operation out-degree, number of outgoing edges
    std::unordered_map<operationIdxType, size_t> _outDegreeTable;
    // contains scheduled ops along with their status/type
    std::unordered_map<operationIdxType, OpOutputInfo> _opOutputTable;
    // container for the schedule output
    llvm::SmallVector<ScheduledOpInfo> _scheduledOps;
    // outputs of the graph
    llvm::DenseSet<operationIdxType> _outputOps;
    // schedule time
    size_t _currentTime;
};

}  // namespace vpux

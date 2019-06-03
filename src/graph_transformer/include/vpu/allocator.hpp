//
// Copyright (C) 2018-2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <unordered_set>
#include <list>
#include <vector>

#include <vpu/utils/enums.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/allocator/structs.hpp>
#include <vpu/allocator_shaves.hpp>

namespace vpu {

//
// UsedMemory
//

struct UsedMemory final {
    int BSS = 0;
    int CMX = 0;
    int blob = 0;
    int input = 0;
    int output = 0;
};

void printTo(std::ostream& os, const UsedMemory& usedMemory);
void printTo(DotLabel& lbl, const UsedMemory& usedMemory);

//
// AllocationResult
//

VPU_DECLARE_ENUM(AllocationStatus,
    OK,
    SHAVES_FAILED,
    DATA_FAILED)

struct AllocationResult final {
    AllocationStatus status = AllocationStatus::OK;
    Stage failedStage;
};

//
// DeallocationMode
//

//
// The following deallocation modes are supported to speed-up performance:
//   * JustFree - Usual data deallocation scheme
//   * MoveFromCMX  - Simple check and reallocation to DDR if tensor does not meet CMX requirements
//

VPU_DECLARE_ENUM(DeallocationMode,
    JustFree,
    MoveFromCMX)

//
// Allocator
//

class Allocator final {
public:
    Allocator();

    void setBatchSize(int batchSize) { _modelBatchSize = batchSize; }

    void reset();

    /**
     * Allocates memory for single data node
     */
    bool allocateData(const Data& data);
    void freeData(const Data& data, DeallocationMode mode = DeallocationMode::JustFree);

    void selfCheck();

    UsedMemory usedMemory() const;

    DataVector getAllocatedDatas(MemoryType memType) const;

    void setNeedToAllocNonIntermData() { _needToAllocNonIntermData = true; }
    /**
     * Allocates memory for the whole vector of data nodes
     */
    AllocationResult preprocess(const ModelPtr& model);

    DataSet& getCandidatesForCMX() { return _candidatesForCMX; }
    bool removeCMXCandidates(const Data& data);

    AllocatorForShaves& getAllocatorOfShaves() { return _allocatorOfShaves; }

private:
    allocator::MemChunk* allocateMem(MemoryType memType, int size, int inUse);
    void freeMem(allocator::MemChunk* chunk);

    allocator::MemChunk* addNewChunk(allocator::MemoryPool& pool, MemoryType memType, int offset, int pointer, int size, int inUse);
    allocator::MemChunk* checkMemPool(allocator::MemoryPool& pool, MemoryType memType, int size, int inUse);

    void extractDatas(MemoryType memType, const DataSet& from, DataVector& out) const;

private:
    int _modelBatchSize = 1;

    int _maxCmxSize = 0;

    allocator::MemoryPool _ddrMemoryPool;
    allocator::MemoryPool _cmxMemoryPool;
    EnumMap<MemoryType, allocator::MemoryPool*> _memPools;

    AllocatorForShaves _allocatorOfShaves;

    DataSet _allocatedData;
    DataSet _allocatedIntermData;

    DataMap<allocator::MemChunk*> _memChunksPerData;

    int _blobMemOffset = 0;
    int _inputMemOffset = 0;
    int _outputMemOffset = 0;

    /**
     * Means that Model::_datas list was changed in some way
     */
    bool _needToAllocNonIntermData = true;

    DataSet _candidatesForCMX;
};

int calcAllocationSize(const Data& data);

}  // namespace vpu

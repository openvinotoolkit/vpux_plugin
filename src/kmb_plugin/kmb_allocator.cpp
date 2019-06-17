//
// Copyright 2019 Intel Corporation.
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

#include <vpusmm.h>
#include <sys/mman.h>

#include "kmb_allocator.h"


void *vpu::KmbPlugin::KmbAllocator::lock(void *handle, InferenceEngine::LockOp) noexcept {
    return nullptr;
}

void vpu::KmbPlugin::KmbAllocator::unlock(void *handle) noexcept {
}

void *vpu::KmbPlugin::KmbAllocator::alloc(size_t size) noexcept {
    auto fd = vpusmm_alloc_dmabuf(size, VPUSMMType::VPUSMMTYPE_COHERENT);
    void *out = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return out;
}

bool vpu::KmbPlugin::KmbAllocator::free(void *handle) noexcept {
    return false;
}

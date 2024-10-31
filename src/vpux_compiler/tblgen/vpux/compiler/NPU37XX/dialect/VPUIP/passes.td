//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// UnrollClusterTiling
//

def UnrollClusterTiling : PassBase<"unroll-cluster-tiling", "vpux::FunctionPass"> {
    let summary = "Extracts child operations from ClusterTiling operation";

    let description = [{
        Legalizes chain of ClusterTiling operations according to the tile schema
    }];

    let constructor = "vpux::VPUIP::arch37xx::createUnrollClusterTilingPass()";
}

//
// AddSwKernelCacheHandlingOps
//

def AddSwKernelCacheHandlingOps : PassBase<"add-sw-kernel-cache-handling-ops", "vpux::FunctionPass"> {
    let summary = "Add cache-handling ops for SW Kernels";

    let description = [{
        This pass adds CACHE_INVALIDATE, CACHE_FLUSH or CACHE_FLUSH_INVALIDATE when needed.

        CACHE_INVALIDATE will be added when at least one input buffer:
        - is located in DDR and it is not produced by an ActSHAVE;
        - is a constant;
        such that the ActSHAVE cache is invalidated before the ActSHAVE Kernel reads from DDR through the cache.

        CACHE_FLUSH will be added when at least one output buffer is located in DDR and at least one of its
        consumers is not an ActSHAVE such that the ActSHAVE cache is flushed after the ActSHAVE Kernel writes
        to DDR through the cache.

        CACHE_FLUSH_INVALIDATE will be added when we have two consecutive SW Kernels and there is no CACHE_FLUSH
        needed after the first SW Kernel, but there is a CACHE_INVALIDATE needed before the second SW Kernel. So, in
        order to avoid invalidating the data produced by the first SW Kernel which was not yet synchronized back to DDR,
        CACHE_FLUSH_INVALIDATE will be added between the two SW Kernels.

        If there are two consecutive SW Kernels and a CACHE_FLUSH was added after the first one and a CACHE_INVALIDATE
        is needed before the second one, the resulting sequence of ops will be:
        [SW Kernel 1] - [CACHE_FLUSH] - [CACHE_INVALIDATE] - [SW Kernel 2].
    }];

    let constructor = "vpux::VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass()";
}

#endif

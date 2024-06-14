//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/logger.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_FeasibleMemorySchedulerSpilling = MLIR_UnitBase;

TEST_F(MLIR_FeasibleMemorySchedulerSpilling, RemoveComputeOpRelocationSpillsForInplace) {
    mlir::MLIRContext ctx(registry);

    // Create a simple IR with 2 CopyOps and 1 NCEOp which is an inplace
    // eltwise that uses input1 also as an output buffer
    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

        !Type_DDR = memref<1x32x32x32xf16, #NCHW, @DDR>
        !Type_CMX = memref<1x32x32x32xf16, #NCHW, [@CMX_NN, 0]>

        module @test {
            func.func @main(%arg0: !Type_DDR, %arg1: !Type_DDR) -> !Type_CMX {

                %buf_cmx_1 = memref.alloc() : !Type_CMX
                %buf_cmx_2 = memref.alloc() : !Type_CMX

                %t0, %r0 = async.execute -> !async.value<!Type_CMX> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64} {
                    %1 = VPUIP.Copy inputs(%arg0 : !Type_DDR) outputs(%buf_cmx_1 : !Type_CMX) -> !Type_CMX
                    async.yield %1 : !Type_CMX
                }

                %t1, %r1 = async.execute -> !async.value<!Type_CMX> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64} {
                    %1 = VPUIP.Copy inputs(%arg1 : !Type_DDR) outputs(%buf_cmx_2 : !Type_CMX) -> !Type_CMX
                    async.yield %1 : !Type_CMX
                }

                %t2, %r2 = async.execute [%t0, %t1] (%r0 as %arg2: !async.value<!Type_CMX>, %r1 as %arg3: !async.value<!Type_CMX>) -> !async.value<!Type_CMX> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64} {
                    %1 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%arg2 : !Type_CMX) weights(%arg3 : !Type_CMX) parent_input(%arg2 : !Type_CMX) parent_output(%buf_cmx_1 : !Type_CMX) outputs(%buf_cmx_1 : !Type_CMX) -> !Type_CMX  variants : {
                            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 31], outStart = [0, 0, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
                        } PPE : {
                            PPETask <ADD> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
                        }
                    async.yield %1: !Type_CMX
                }

                %r = async.await %r2 : !async.value<!Type_CMX>
                return %r : !Type_CMX
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto log = vpux::Logger::global();

    // Run helper utilities used by scheduler which allow it to gather data
    // represented in the IR
    AliasesInfo aliasesInfo{func};
    AsyncDepsInfo depsInfo{func};

    // Assume following buffers confgiuration of an eltwise inplace
    // input1: [    0 -  65536]
    // input2: [65536 - 131072]
    // output: [    0 -  65536] <- same buffer as input1, this buffer is spilled
    const size_t bufSize = 65536;
    const size_t buf1SpilledOffset = 0;
    const size_t buf2Offset = 65536;
    const int64_t newOutputOffset = 131072;

    VPUIP::NCEClusterTaskOp nceOp = nullptr;
    func.walk([&](mlir::Operation* op) {
        if (mlir::isa<VPUIP::NCEClusterTaskOp>(op)) {
            nceOp = mlir::cast<VPUIP::NCEClusterTaskOp>(op);
        }
    });
    ASSERT_TRUE(nceOp != nullptr);

    auto rootsInput1 = aliasesInfo.getRoots(nceOp.getInput());
    auto rootsInput2 = aliasesInfo.getRoots(nceOp.getWeights());
    ASSERT_TRUE(rootsInput1.size() == 1);
    ASSERT_TRUE(rootsInput2.size() == 1);

    auto buf1Spilled = *rootsInput1.begin();
    auto buf2 = *rootsInput2.begin();

    // Required components needed to construct FeasibleMemorySchedulerSpilling object
    const auto memKind = VPU::MemoryKind::CMX_NN;
    const auto secondLvlMemKind = VPU::MemoryKind::DDR;
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;
    LinearScan<mlir::Value, LinearScanHandler> scan(1024 * 1024, {}, alignment);

    FeasibleMemorySchedulerSpilling spilling(memKind, secondLvlMemKind, depsInfo, aliasesInfo, log, scan);

    // Initialize example scheduled ops structure that could have been produced during scheduling
    // This is a simple IR but by manually preparing it it is possible to create a scenario where there
    // is a case for performing relocation spilling optimization for inplace op
    // Below scheduledOps structure has following operations
    //  opId  info
    //  0     copyOp
    //  1     copyOp
    //  2     eltwise inplace
    //  3     spill write
    //  4     spill read
    // spill write/read pair can be optimized

    FeasibleMemoryScheduler::ScheduledOpInfoVec scheduledOps = {
            {0,                                                                 // opId
             FeasibleMemoryScheduler::EOpType::ORIGINAL_OP,                     // opType
             0,                                                                 // cycleBegin
             1,                                                                 // cycleEnd
             {},                                                                // inputResources
             {{buf1SpilledOffset, buf1SpilledOffset + bufSize, buf1Spilled}}},  // outputResources
            {1, FeasibleMemoryScheduler::EOpType::ORIGINAL_OP, 1, 2, {}, {{buf2Offset, buf2Offset + bufSize, buf2}}},
            {2,
             FeasibleMemoryScheduler::EOpType::ORIGINAL_OP,
             2,
             3,
             {{buf1SpilledOffset, buf1SpilledOffset + bufSize, buf1Spilled}, {buf2Offset, buf2Offset + bufSize, buf2}},
             {{buf1SpilledOffset, buf1SpilledOffset + bufSize, buf1Spilled}}},
            {2,
             FeasibleMemoryScheduler::EOpType::IMPLICIT_SPILL_WRITE_OP,
             3,
             4,
             {{buf1SpilledOffset, buf1SpilledOffset + bufSize, buf1Spilled}},
             {}},
            {2,
             FeasibleMemoryScheduler::EOpType::IMPLICIT_SPILL_READ_OP,
             4,
             5,
             {},
             {{newOutputOffset, newOutputOffset + bufSize, buf1Spilled}}}};

    // Perform spilling relocation optimization. After it scheduledOps size should be 3 as spill
    // operations should no longer be present
    spilling.removeComputeOpRelocationSpills(scheduledOps);

    EXPECT_EQ(scheduledOps.size(), 3);
    // nceOp has index 2
    EXPECT_EQ(scheduledOps[2].opType_, FeasibleMemoryScheduler::EOpType::ORIGINAL_OP);

    // Check if output of nceOp is no longer the same as first input
    EXPECT_NE(scheduledOps[2].beginOutputResource(0), scheduledOps[2].beginInputResource(0));
    EXPECT_NE(scheduledOps[2].endOutputResource(0), scheduledOps[2].endInputResource(0));

    // Verify output of nceOp now points to location previosuly assigned for spill-read op
    EXPECT_EQ(scheduledOps[2].beginOutputResource(0), newOutputOffset);
    EXPECT_EQ(scheduledOps[2].endOutputResource(0), newOutputOffset + bufSize);

    // Verify inPlace attribute is no longer present in nceOp
    EXPECT_FALSE(nceOp.getIsInplace().has_value());

    // Check both inputs and output buffers are allocated by different allocation operations
    auto input1AllocOp = scheduledOps[2].getInputBuffer(0).getDefiningOp<mlir::memref::AllocOp>();
    auto input2AllocOp = scheduledOps[2].getInputBuffer(1).getDefiningOp<mlir::memref::AllocOp>();
    auto outputAllocOp = scheduledOps[2].getOutputBuffer(0).getDefiningOp<mlir::memref::AllocOp>();

    ASSERT_TRUE(input1AllocOp != nullptr);
    ASSERT_TRUE(input2AllocOp != nullptr);
    ASSERT_TRUE(outputAllocOp != nullptr);
    EXPECT_NE(input1AllocOp, input2AllocOp);
    EXPECT_NE(input1AllocOp, outputAllocOp);
    EXPECT_NE(input2AllocOp, outputAllocOp);
}

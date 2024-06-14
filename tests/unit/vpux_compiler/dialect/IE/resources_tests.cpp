//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_Resources = MLIR_UnitBase;

TEST_F(MLIR_Resources, UsedMemInFunction) {
    mlir::MLIRContext ctx(registry);

    constexpr StringLiteral inputIR = R"(
        module @TwoFunctions {
            IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
                IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
                IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
            }
            IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

            func.func @emptyFunc() -> () {
                return
            }

            func.func @twoModulesFunc() {
                builtin.module @UsedMemory {
                    IE.MemoryResource 1000 bytes of @DDR
                    IE.MemoryResource 10 bytes of @CMX_NN
                }

                builtin.module @Unknown {
                }

                return
            }

            func.func @wrongModuleNameFunc() {
                builtin.module @WrongName {
                    IE.MemoryResource 1000 bytes of @DDR
                    IE.MemoryResource 10 bytes of @CMX_NN
                }

                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto emptyFunc = module.get().lookupSymbol<mlir::func::FuncOp>("emptyFunc");
    ASSERT_TRUE(emptyFunc != nullptr);

    auto expectedByteSize = Byte{10};
    EXPECT_EQ(IE::getUsedMemory(emptyFunc), SmallVector<IE::MemoryResourceOp>{});
    EXPECT_NO_THROW(IE::setUsedMemory(emptyFunc, VPU::MemoryKind::DDR, expectedByteSize));
    auto usedMemVec = IE::getUsedMemory(emptyFunc);
    EXPECT_EQ(usedMemVec.size(), 1);
    EXPECT_EQ(usedMemVec[0].getByteSize(), expectedByteSize.count());

    auto twoModulesFunc = module.get().lookupSymbol<mlir::func::FuncOp>("twoModulesFunc");
    ASSERT_TRUE(twoModulesFunc != nullptr);

    EXPECT_ANY_THROW(IE::getUsedMemory(twoModulesFunc)) << "Expected exactly one Temporary module.";
    EXPECT_ANY_THROW(IE::setUsedMemory(twoModulesFunc, VPU::MemoryKind::DDR, expectedByteSize))
            << "Expected exactly one Temporary module.";

    auto wrongModuleNameFunc = module.get().lookupSymbol<mlir::func::FuncOp>("wrongModuleNameFunc");
    ASSERT_TRUE(wrongModuleNameFunc != nullptr);

    EXPECT_ANY_THROW(IE::getUsedMemory(wrongModuleNameFunc)) << "Temporary module must have sym name \"UsedMemory\"";
    EXPECT_ANY_THROW(IE::setUsedMemory(wrongModuleNameFunc, VPU::MemoryKind::DDR, expectedByteSize))
            << "Temporary module must have sym name \"UsedMemory\"";
}

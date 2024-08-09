//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --move-declarations-to-top %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

func.func @MoveToTopOfBlock(%arg0: memref<16xf16>, %arg1: memref<8xf16>) -> memref<8xf16> {
    %0 = VPUIP.SubView %arg0 [0] [8] : memref<16xf16> to memref<8xf16>
    %decl0 = VPUIP.StaticAlloc<0> -> memref<8xf16>
    %cst2 = const.Declare memref<8xf16> = dense<2.000000e+00> : tensor<8xf16>
    %1 = IERT.Add inputs(%0 : memref<8xf16>, %cst2 : memref<8xf16>) outputs(%decl0 : memref<8xf16>) -> memref<8xf16>

    %cst4 = const.Declare memref<8xf16> = dense<4.000000e+00> : tensor<8xf16>
    %decl16 = VPUIP.StaticAlloc<16> -> memref<8xf16>
    %t2, %f2 = async.execute -> !async.value<memref<8xf16>> {
        %2 = IERT.Add inputs(%1 : memref<8xf16>, %cst4 : memref<8xf16>) outputs(%decl16 : memref<8xf16>) -> memref<8xf16>
        async.yield %2 : memref<8xf16>
    }
    %3 = async.await %f2 : !async.value<memref<8xf16>>

    %decl32 = VPUIP.StaticAlloc<32> -> memref<8xf16>
    %cst9 = const.Declare memref<8xf16> = dense<9.000000e+00> : tensor<8xf16>
    %t4, %f4 = async.execute -> !async.value<memref<8xf16>> {
        %4 = IERT.Add inputs(%3 : memref<8xf16>, %cst9 : memref<8xf16>) outputs(%decl32 : memref<8xf16>) -> memref<8xf16>
        async.yield %4 : memref<8xf16>
    }
    %5 = async.await %f4 : !async.value<memref<8xf16>>

    %6 = VPUIP.Copy inputs(%5 : memref<8xf16>) outputs(%arg1 : memref<8xf16>) -> memref<8xf16>
    return %6 : memref<8xf16>

    // CHECK-DAG:   [[CST2:%.+]] = const.Declare memref<8xf16> = dense<2.000000e+00> : tensor<8xf16>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare memref<8xf16> = dense<4.000000e+00> : tensor<8xf16>
    // CHECK-DAG:   [[CST9:%.+]] = const.Declare memref<8xf16> = dense<9.000000e+00> : tensor<8xf16>

    // CHECK-DAG:   [[DECL0:%.+]] = VPUIP.StaticAlloc<0> -> memref<8xf16>
    // CHECK-DAG:   [[DECL16:%.+]] = VPUIP.StaticAlloc<16> -> memref<8xf16>
    // CHECK-DAG:   [[DECL32:%.+]] = VPUIP.StaticAlloc<32> -> memref<8xf16>

    // CHECK:       [[VAR0:%.+]] = VPUIP.SubView %arg0 [0] [8] : memref<16xf16> to memref<8xf16>

    // CHECK:       [[VAR1:%.+]] = IERT.Add inputs([[VAR0]] : memref<8xf16>, [[CST2]] : memref<8xf16>) outputs([[DECL0]] : memref<8xf16>)

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute -> !async.value<memref<8xf16>> {
    // CHECK:           [[VAR2:%.+]] = IERT.Add inputs([[VAR1]] : memref<8xf16>, [[CST4]] : memref<8xf16>) outputs([[DECL16]] : memref<8xf16>)
    // CHECK:           async.yield [[VAR2]] : memref<8xf16>
    // CHECK:       }
    // CHECK:       [[VAR3:%.+]] = async.await [[F2]] : !async.value<memref<8xf16>>

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute -> !async.value<memref<8xf16>> {
    // CHECK:           [[VAR4:%.+]] = IERT.Add inputs([[VAR3]] : memref<8xf16>, [[CST9]] : memref<8xf16>) outputs([[DECL32]] : memref<8xf16>)
    // CHECK:           async.yield [[VAR4]] : memref<8xf16>
    // CHECK:       }
    // CHECK:       [[VAR5:%.+]] = async.await [[F4]] : !async.value<memref<8xf16>>

    // CHECK:       [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR5]] : memref<8xf16>) outputs(%arg1 : memref<8xf16>)
    // CHECK:       return [[VAR6]] : memref<8xf16>
}

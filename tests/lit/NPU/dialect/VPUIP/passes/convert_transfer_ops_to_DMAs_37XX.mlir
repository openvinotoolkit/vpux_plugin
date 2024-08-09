//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-transfer-ops-to-DMAs %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: @TimestampToDMA
func.func @TimestampToDMA(%arg0: memref<1xui64>) -> memref<1xui64> {
    %0 = VPUIP.StaticAlloc<0> -> memref<1xui64, @CMX_NN>
    %1 = VPUIP.Timestamp(%0 : memref<1xui64, @CMX_NN>) -> memref<1xui64, @CMX_NN>
    %2 = VPUIP.Copy inputs(%1 : memref<1xui64, @CMX_NN>) outputs(%arg0 : memref<1xui64>) -> memref<1xui64>
    return %2: memref<1xui64>

    // CHECK:       [[VAR0:%.*]] = VPUIP.StaticAlloc<0> -> memref<1xui64, @CMX_NN>
    // CHECK-NOT:       VPUIP.Timestamp
    // CHECK:       [[VAR1:%.*]] = VPURT.DeclareBuffer <Register>
    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:           inputs([[VAR1]] : memref<1xui64, @Register>)
    // CHECK-SAME:           outputs([[VAR0]] : memref<1xui64, @CMX_NN>)
    // CHECK:       [[VAR3:%.*]] = VPUIP.NNDMA
    // CHECK:           inputs([[VAR2]] : memref<1xui64, @CMX_NN>)
    // CHECK:           outputs(%arg0 : memref<1xui64>)
    // CHECK:       return [[VAR3]] : memref<1xui64>
}

//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @ConstantLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo :  {
        DataInfo "output" : tensor<1x2x2x2xf16>
    }

func.func @main(%arg0: memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16> {
    %0 = const.Declare memref<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x2x2x2xf16>) outputs(%arg0 : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
// expected-error@+1 {{function output at index=0 should be an alias of the output buffer, but it's not}}
    return %0: memref<1x2x2x2xf16>
}

}

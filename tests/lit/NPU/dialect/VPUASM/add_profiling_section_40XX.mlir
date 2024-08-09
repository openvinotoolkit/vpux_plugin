//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-profiling-section %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @AddProfilingSection {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x2x3x4xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2x3x4xf16>
    } profilingOutputsInfo : {
    }
    func.func @main() {
        ELF.Main @ELFMain {
            VPUASM.ProfilingMetadata @ProfilingMetadata {metadata = dense<1> : vector<184xui8>}
        }
        return
    }
}

// CHECK: ELF.CreateProfilingSection @".profiling" aligned(1) secFlags("SHF_NONE") {
// CHECK-NEXT: VPUASM.ProfilingMetadata @ProfilingMetadata {metadata = {{.*}} : vector<184xui8>}

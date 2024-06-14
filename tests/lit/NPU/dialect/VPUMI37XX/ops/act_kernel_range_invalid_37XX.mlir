//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch%" %s
// REQUIRES: arch-VPUX37XX

module @Test {
IE.CNNNetwork entryPoint : @AssignFullKernelPath inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x64x32x514xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x64x32x514xf16>
}

func.func @NoKernelText(%arg0: memref<1x64x32x514xf16, @DDR>, %arg1: memref<1x64x32x514xf16, @DDR>) -> memref<1x64x32x514xf16, @DDR> {
  %1 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
  %2 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
    // expected-error@+1 {{ActKernelRange with COMPUTE taskType should have KernelText, KernelArgs, KernelEntry}}
  %3 = VPUMI37XX.ActKernelRange kernel_args_index(%2 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%1 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  return %arg1 : memref<1x64x32x514xf16, @DDR>
}
}

// -----

module @Test {
IE.CNNNetwork entryPoint : @AssignFullKernelPath inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x64x32x514xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x64x32x514xf16>
}

func.func @NoKernelEntry(%arg0: memref<1x64x32x514xf16, @DDR>, %arg1: memref<1x64x32x514xf16, @DDR>) -> memref<1x64x32x514xf16, @DDR> {
  %1 = VPUMI37XX.DeclareKernelText kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
  %2 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
    // expected-error@+1 {{ActKernelRange with COMPUTE taskType should have KernelText, KernelArgs, KernelEntry}}
  %3 = VPUMI37XX.ActKernelRange kernel_text_index(%1 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%2 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  return %arg1 : memref<1x64x32x514xf16, @DDR>
}
}

// -----

module @Test {
IE.CNNNetwork entryPoint : @AssignFullKernelPath inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x64x32x514xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x64x32x514xf16>
}

func.func @NoKernelArgs(%arg0: memref<1x64x32x514xf16, @DDR>, %arg1: memref<1x64x32x514xf16, @DDR>) -> memref<1x64x32x514xf16, @DDR> {
  %1 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
  %2 = VPUMI37XX.DeclareKernelText kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
    // expected-error@+1 {{ActKernelRange with COMPUTE taskType should have KernelText, KernelArgs, KernelEntry}}
  %3 = VPUMI37XX.ActKernelRange kernel_text_index(%2 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%1 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  return %arg1 : memref<1x64x32x514xf16, @DDR>
}
}

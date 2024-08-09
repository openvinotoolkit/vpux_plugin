//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --assign-full-kernel-path-VPUMI37XX %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @Test {

IE.CNNNetwork entryPoint : @AssignFullKernelPath inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x64x32x514xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x64x32x514xf16>
}

func.func @AssignFullKernelPath(%arg0: memref<1x64x32x514xf16, @DDR>, %arg1: memref<1x64x32x514xf16, @DDR>) -> memref<1x64x32x514xf16, @DDR> {
  %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x64x32x514xf16, @DDR>
  %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x32x514xf16, @DDR>
  // CHECK: activation_mish_3720xx_lsu0_wo
  %3 = VPUMI37XX.DeclareKernelText kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
  // CHECK: activation_mish_3720xx_lsu0_wo
  %4 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
  // CHECK: activation_mish_3720xx_lsu0_wo
  %5 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_mish") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI37XX.ActKernelRange kernel_text_index(%3 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%5 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%4 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
  // CHECK: activation_mish
  %7 = VPUMI37XX.KernelParams inputs(%1 : memref<1x64x32x514xf16, @DDR>) outputs(%2 : memref<1x64x32x514xf16, @DDR>) kernel_type("activation_mish") kernel_params(dense_resource<__elided__> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
  %8 = VPUMI37XX.ActKernelInvocation range_index(%6 : <0:0:0>) params_index(%7 : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

  %9 = VPUMI37XX.ActKernelRange kernelTaskType(@CACHE_FLUSH_INVALIDATE) -> !VPURegMapped.Index<0:0:1>
  // CHECK: cache_op_flush_invalidate
  %10 = VPUMI37XX.KernelParams kernel_type("cache_op_flush_invalidate") kernel_params(dense<255> : vector<1xui8>) -> !VPURegMapped.Index<0:0:1>
  %11 = VPUMI37XX.ActKernelInvocation range_index(%9 : <0:0:1>) params_index(%10 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  return %arg1 : memref<1x64x32x514xf16, @DDR>
}
}

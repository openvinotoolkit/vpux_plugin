//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUMI37XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU37XX

IE.CNNNetwork entryPoint : @single_hswish inputsInfo : {
  DataInfo "input" : tensor<1x1000xf16>
} outputsInfo : {
  DataInfo "hswish" : tensor<1x1000xf16>
}
VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @single_hswish() {
  %0 = VPURegMapped.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
  %1 = VPURegMapped.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>
  %2 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
  %3 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>

  %4 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x1x1x1000xf16>
  %5 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x1x1x1000xf16>
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

  %8 = VPUMI37XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %9 = VPUMI37XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %10 = VPUMI37XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %11 = VPUMI37XX.KernelParams inputs(%6 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>

  %16 = VPUMI37XX.ActKernelRange taskLocation(%0 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%8 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%10 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%9 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %17 = VPUMI37XX.ActKernelInvocation taskLocation(%1 : !VPURegMapped.Index<0:0:0>) range_index(%16 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

  %18 = VPUMI37XX.MappedInference actKernelRanges(%16 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%17 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: func.func @single_hswish

//CHECK: ELF.CreateLogicalSection @[[SECMETA:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareTaskBuffer @[[TBRANGE:.*]] idx(!VPURegMapped.Index<0:0:0>) <ActKernelRange>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBINVO:.*]] idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation>

//CHECK: ELF.CreateLogicalSection @[[SECIN0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF0:.*]] !VPUASM.Buffer
//CHECK: ELF.CreateLogicalSection @[[SECOUT0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF1:.*]] !VPUASM.Buffer
//CHECK: ELF.CreateLogicalSection @[[SECCMX0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF2:.*]] !VPUASM.Buffer
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer

//CHECK: ELF.CreateSection @[[SECSHVTEXT:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareKernelText @[[SYMTEXT0:.*]] : "activation_hswish"
//CHECK: VPUASM.DeclareKernelEntry @[[SYMENTRY0:.*]] : "activation_hswish"
//CHECK: ELF.CreateSection @[[SECSHVDATA:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareKernelData @[[SYMDATA0:.*]] : "activation_hswish"
//CHECK: ELF.CreateSection @[[SECSHVPARAMS:.*]] aligned
//CHECK-NEXT: VPUASM.KernelParams @[[SYMPARAMS0:.*]] inputs([@[[SECCMX0]]::@[[SYMBUFF2]]]) outputs([@[[SECCMX0]]::@[[SYMBUFF3]]]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("activation_hswish")

//CHECK: ELF.CreateSection @[[SECSHVRANGE:.*]] aligned
//CHECK-NEXT: VPUASM.ActKernelRange @[[SYMACTRANGE0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBRANGE]])
    //CHECK-SAME: calls @[[SECSHVTEXT]]::@[[SYMTEXT0]] : @[[SYMENTRY0]]

//CHECK: ELF.CreateSection @[[SECSHVINVOCATION:.*]] aligned
//CHECK: VPUASM.ActKernelInvocation @[[SYMACTINVO0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBINVO]])
    //CHECK-SAME: -> @[[SECSHVRANGE]]::@[[SYMACTRANGE0]](kernel_data : @[[SECSHVDATA]]::@[[SYMDATA0]], kernel_params : @[[SECSHVPARAMS]]::@[[SYMPARAMS0]]) waits([]) updates([])

//CHECK: VPUASM.MappedInference_37XX @MappedInference : actKernelRanges(@[[SECSHVRANGE]]::@[[SYMACTRANGE0]]) actKernelInvocations(@[[SECSHVINVOCATION]]::@[[SYMACTINVO0]]) dmaCount([0, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(0)

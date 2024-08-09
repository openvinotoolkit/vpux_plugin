//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %s && prof_parser -b %t -p %data_path_npu%/profiling-0-37XX-MVN.bin -f json -vv | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#loc = loc(unknown)
#loc1 = loc("profiling_result")
module @MVN_case1 attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  module @UsedMemory {
    IE.MemoryResource 4096 bytes of @DDR loc(#loc)
  } loc(#loc)
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096] loc(#loc)
  module @VPU.SW {
    func.func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh", VPU.task_type = @COMPUTE} loc(#loc)
    func.func private @builtin_Swish(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, f64) attributes {VPU.kernel_code = "activation_swish.cpp", VPU.kernel_entry = "activation_swish", VPU.task_type = @COMPUTE} loc(#loc)
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1", VPU.task_type = @COMPUTE} loc(#loc)
    func.func private @builtin_Convert(memref<*xf32, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE} loc(#loc)
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"} loc(#loc)
  } loc(#loc)
  IE.TileResource {activity_factor = 0.01668275783741318 : f64} 2 of @NCE at 1.300000e+03 MHz {
    builtin.module @UsedMemory {
      IE.MemoryResource 7168 bytes of @CMX_NN loc(#loc)
    } loc(#loc)
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0 loc(#loc)
      } loc(#loc)
    } loc(#loc)
    IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware loc(#loc)
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64} loc(#loc)
    IE.ExecutorResource 2 of @SHAVE_ACT loc(#loc)
    IE.ExecutorResource 1 of @SHAVE_NN loc(#loc)
    IE.ExecutorResource 1 of @DPU loc(#loc)
  } loc(#loc)
  IE.ExecutorResource 2 of @DMA_NN loc(#loc)
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64} loc(#loc)
  IE.CNNNetwork {inferenceTiming = 58624 : i64} entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x4x512xf32> loc(#loc)
  } outputsInfo : {
    DataInfo "Div_0" : tensor<1x4x512xf32> loc(#loc)
  } profilingOutputsInfo : {
    DataInfo "profilingOutput" {
      VPUIP.ProfilingSection type 3 : 896 bytes from 0 loc(#loc)
      VPUIP.ProfilingSection type 4 : 256 bytes from 896 loc(#loc)
      VPUIP.ProfilingSection type 5 : 64 bytes from 1152 loc(#loc)
    } : tensor<304xui32> loc(#loc)
  } loc(#loc)
  func.func @main(%arg0: memref<1x4x512xf32, @DDR> loc(unknown), %arg1: memref<1x4x512xf32, @DDR> loc(unknown), %arg2: memref<304xui32> loc("profiling_result")) -> (memref<1x4x512xf32, @DDR>, memref<304xui32>) {
    %0 = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register> loc(#loc2)
    %1 = VPURT.DeclareBuffer <ProfilingOutput> [0] <1152> -> memref<1xui32> loc(#loc2)
    %2 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier loc(#loc3)
    %3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier loc(#loc3)
    %4 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier loc(#loc3)
    %5 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier loc(#loc36)
    %6 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier loc(#loc36)
    %7 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier loc(#loc37)
    %8 = VPURT.ConfigureBarrier<6> -> !VPURT.Barrier loc(#loc38)
    %9 = VPURT.ConfigureBarrier<7> -> !VPURT.Barrier loc(#loc9)
    %10 = VPURT.ConfigureBarrier<8> -> !VPURT.Barrier loc(#loc39)
    %11 = VPURT.ConfigureBarrier<9> -> !VPURT.Barrier loc(#loc39)
    %12 = VPURT.ConfigureBarrier<10> -> !VPURT.Barrier loc(#loc40)
    %13 = VPURT.ConfigureBarrier<11> -> !VPURT.Barrier loc(#loc40)
    %14 = VPURT.ConfigureBarrier<12> -> !VPURT.Barrier loc(#loc40)
    %15 = VPURT.ConfigureBarrier<13> -> !VPURT.Barrier loc(#loc41)
    %16 = VPURT.ConfigureBarrier<14> -> !VPURT.Barrier loc(#loc41)
    %17 = VPURT.ConfigureBarrier<15> {isFinalBarrier} -> !VPURT.Barrier loc(#loc16)
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<56xui32, [@CMX_NN, 0]> loc(#loc17)
    %19 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<56xui32, [@CMX_NN, 1]> loc(#loc17)
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<56xui32, [@CMX_NN, 0]> loc(#loc9)
    %21 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<56xui32, [@CMX_NN, 1]> loc(#loc9)
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc42)
    %23 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc43)
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc3)
    %25 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc3)
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc3)
    %27 = VPURT.DeclareBuffer <CMX_NN> [1] <768> -> memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc3)
    %28 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc44)
    %29 = VPURT.DeclareBuffer <CMX_NN> [1] <768> -> memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc45)
    %30 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc3)
    %31 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc3)
    %32 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc36)
    %33 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc36)
    %34 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc39)
    %35 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc39)
    %36 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %37 = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %38 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc39)
    %39 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc39)
    %40 = VPURT.DeclareBuffer <CMX_NN> [0] <3072> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc40)
    %41 = VPURT.DeclareBuffer <CMX_NN> [1] <3072> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc40)
    %42 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc40)
    %43 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc40)
    %44 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %45 = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %46 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc46)
    %47 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc47)
    %48 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc41)
    %49 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc41)
    %50 = VPURT.DeclareBuffer <CMX_NN> [0] <3072> -> memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc41)
    %51 = VPURT.DeclareBuffer <CMX_NN> [1] <3072> -> memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc41)
    %52 = VPURT.DeclareBuffer <CMX_NN> [0] <3072> -> memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc48)
    %53 = VPURT.DeclareBuffer <CMX_NN> [1] <3072> -> memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc49)
    %54 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc3)
    %55 = VPURT.DeclareBuffer <NetworkInput> [0] <4096> -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc3)
    %56 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc50)
    %57 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc51)
    %58 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc36)
    %59 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc36)
    %60 = VPURT.DeclareBuffer <CMX_NN> [0] <3840> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc52)
    %61 = VPURT.DeclareBuffer <CMX_NN> [1] <3840> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc53)
    %62 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc54)
    %63 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc55)
    %64 = VPURT.DeclareBuffer <CMX_NN> [0] <1792> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc56)
    %65 = VPURT.DeclareBuffer <CMX_NN> [1] <1792> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc57)
    %66 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc58)
    %67 = VPURT.DeclareBuffer <CMX_NN> [1] <768> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc59)
    %68 = VPURT.DeclareBuffer <CMX_NN> [0] <1792> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc60)
    %69 = VPURT.DeclareBuffer <CMX_NN> [1] <1792> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc61)
    %70 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc62)
    %71 = VPURT.DeclareBuffer <CMX_NN> [1] <768> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc63)
    %72 = VPURT.DeclareBuffer <CMX_NN> [0] <3840> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc64)
    %73 = VPURT.DeclareBuffer <CMX_NN> [1] <3840> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc65)
    %74 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc66)
    %75 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc67)
    %76 = VPURT.DeclareBuffer <CMX_NN> [0] <3840> -> memref<1x1x512x1xf16, [@CMX_NN, 0]> loc(#loc68)
    %77 = VPURT.DeclareBuffer <CMX_NN> [1] <3840> -> memref<1x1x512x1xf16, [@CMX_NN, 1]> loc(#loc69)
    %78 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x1x512x1xf16, [@CMX_NN, 0]> loc(#loc70)
    %79 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x1x512x1xf16, [@CMX_NN, 1]> loc(#loc71)
    %80 = VPURT.DeclareBuffer <CMX_NN> [0] <1792> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc72)
    %81 = VPURT.DeclareBuffer <CMX_NN> [1] <1792> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc73)
    %82 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc74)
    %83 = VPURT.DeclareBuffer <CMX_NN> [1] <768> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc75)
    %84 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<56xui32, @DDR> loc(#loc9)
    %85 = VPURT.DeclareBuffer <ProfilingOutput> [0] <224> -> memref<56xui32, @DDR> loc(#loc9)
    %86 = VPURT.DeclareBuffer <CMX_NN> [0] <1792> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc76)
    %87 = VPURT.DeclareBuffer <CMX_NN> [1] <1792> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc77)
    %88 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc78)
    %89 = VPURT.DeclareBuffer <CMX_NN> [1] <768> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc79)
    %90 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc80)
    %91 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc81)
    %92 = VPURT.DeclareBuffer <CMX_NN> [0] <3072> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc82)
    %93 = VPURT.DeclareBuffer <CMX_NN> [1] <3072> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc83)
    %94 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc84)
    %95 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc85)
    %96 = VPURT.DeclareBuffer <CMX_NN> [0] <3072> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc86)
    %97 = VPURT.DeclareBuffer <CMX_NN> [1] <3072> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc87)
    %98 = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc88)
    %99 = VPURT.DeclareBuffer <CMX_NN> [1] <1536> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc89)
    %100 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc90)
    %101 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc91)
    %102 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x2x256x1xf16, [@CMX_NN, 0]> loc(#loc92)
    %103 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x2x256x1xf16, [@CMX_NN, 1]> loc(#loc93)
    %104 = VPURT.DeclareBuffer <CMX_NN> [0] <3072> -> memref<1x2x256x1xf16, [@CMX_NN, 0]> loc(#loc94)
    %105 = VPURT.DeclareBuffer <CMX_NN> [1] <3072> -> memref<1x2x256x1xf16, [@CMX_NN, 1]> loc(#loc95)
    %106 = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 0]> loc(#loc96)
    %107 = VPURT.DeclareBuffer <CMX_NN> [1] <1536> -> memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 1]> loc(#loc97)
    %108 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 0]> loc(#loc98)
    %109 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 1]> loc(#loc99)
    %110 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc41)
    %111 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc41)
    %112 = VPURT.DeclareBuffer <CMX_NN> [0] <3008> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc100)
    %113 = VPURT.DeclareBuffer <CMX_NN> [1] <3008> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc101)
    %114 = VPURT.DeclareBuffer <ProfilingOutput> [0] <448> -> memref<56xui32, @DDR> loc(#loc17)
    %115 = VPURT.DeclareBuffer <ProfilingOutput> [0] <672> -> memref<56xui32, @DDR> loc(#loc17)
    %116 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc41)
    %117 = VPURT.DeclareBuffer <NetworkOutput> [0] <4096> -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc41)
    %118 = VPURT.DeclareBuffer <CMX_NN> [0] <544> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc102)
    %119 = VPURT.DeclareBuffer <CMX_NN> [1] <544> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc103)
    %120 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc104)
    %121 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc105)
    %122 = VPURT.DeclareBuffer <CMX_NN> [0] <608> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc106)
    %123 = VPURT.DeclareBuffer <CMX_NN> [1] <608> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc107)
    %124 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc108)
    %125 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc109)
    %126 = VPURT.DeclareBuffer <CMX_NN> [0] <672> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc110)
    %127 = VPURT.DeclareBuffer <CMX_NN> [1] <672> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc111)
    %128 = VPURT.DeclareBuffer <CMX_NN> [0] <704> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc112)
    %129 = VPURT.DeclareBuffer <CMX_NN> [1] <704> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc113)
    %130 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc114)
    %131 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc115)
    %132 = VPURT.DeclareBuffer <CMX_NN> [0] <2848> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc116)
    %133 = VPURT.DeclareBuffer <CMX_NN> [1] <2848> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc117)
    %134 = VPURT.DeclareBuffer <CMX_NN> [0] <2880> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc118)
    %135 = VPURT.DeclareBuffer <CMX_NN> [1] <2880> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc119)
    %136 = VPURT.DeclareBuffer <CMX_NN> [0] <2912> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc120)
    %137 = VPURT.DeclareBuffer <CMX_NN> [1] <2912> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc121)
    %138 = VPURT.DeclareBuffer <CMX_NN> [0] <2944> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc122)
    %139 = VPURT.DeclareBuffer <CMX_NN> [1] <2944> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc123)
    %140 = VPURT.DeclareBuffer <CMX_NN> [0] <2976> -> memref<8xui32, [@CMX_NN, 0]> loc(#loc124)
    %141 = VPURT.DeclareBuffer <CMX_NN> [1] <2976> -> memref<8xui32, [@CMX_NN, 1]> loc(#loc125)
    %142 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc126)
    %143 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc126)
    %144 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc126)
    %145 = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc126)
    %146 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc127)
    %147 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc127)
    %148 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc127)
    %149 = VPURT.DeclareBuffer <CMX_NN> [0] <264> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc127)
    %150 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc126)
    %151 = VPURT.DeclareBuffer <CMX_NN> [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc126)
    %152 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc126)
    %153 = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc126)
    %154 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc127)
    %155 = VPURT.DeclareBuffer <CMX_NN> [0] <272> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc127)
    %156 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc127)
    %157 = VPURT.DeclareBuffer <CMX_NN> [0] <280> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc127)
    %158 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc128)
    %159 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc128)
    %160 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc128)
    %161 = VPURT.DeclareBuffer <CMX_NN> [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc128)
    %162 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc129)
    %163 = VPURT.DeclareBuffer <CMX_NN> [0] <288> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc129)
    %164 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc129)
    %165 = VPURT.DeclareBuffer <CMX_NN> [0] <296> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc129)
    %166 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc130)
    %167 = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc130)
    %168 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc130)
    %169 = VPURT.DeclareBuffer <CMX_NN> [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc130)
    %170 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc131)
    %171 = VPURT.DeclareBuffer <CMX_NN> [0] <304> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc131)
    %172 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc131)
    %173 = VPURT.DeclareBuffer <CMX_NN> [0] <312> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc131)
    %174 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc132)
    %175 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc132)
    %176 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc132)
    %177 = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc132)
    %178 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc133)
    %179 = VPURT.DeclareBuffer <CMX_NN> [0] <320> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc133)
    %180 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc133)
    %181 = VPURT.DeclareBuffer <CMX_NN> [0] <328> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc133)
    %182 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc132)
    %183 = VPURT.DeclareBuffer <CMX_NN> [0] <80> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc132)
    %184 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc132)
    %185 = VPURT.DeclareBuffer <CMX_NN> [0] <88> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc132)
    %186 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc133)
    %187 = VPURT.DeclareBuffer <CMX_NN> [0] <336> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc133)
    %188 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc133)
    %189 = VPURT.DeclareBuffer <CMX_NN> [0] <344> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc133)
    %190 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc134)
    %191 = VPURT.DeclareBuffer <CMX_NN> [0] <96> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc134)
    %192 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc134)
    %193 = VPURT.DeclareBuffer <CMX_NN> [0] <104> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc134)
    %194 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc135)
    %195 = VPURT.DeclareBuffer <CMX_NN> [0] <352> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc135)
    %196 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc135)
    %197 = VPURT.DeclareBuffer <CMX_NN> [0] <360> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc135)
    %198 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc134)
    %199 = VPURT.DeclareBuffer <CMX_NN> [0] <112> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc134)
    %200 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc134)
    %201 = VPURT.DeclareBuffer <CMX_NN> [0] <120> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc134)
    %202 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16xui64, [@CMX_NN, 0]> loc(#loc29)
    %203 = VPURT.DeclareBuffer <ProfilingOutput> [0] <896> -> memref<16xui64> loc(#loc29)
    %204 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc135)
    %205 = VPURT.DeclareBuffer <CMX_NN> [0] <368> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc135)
    %206 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc135)
    %207 = VPURT.DeclareBuffer <CMX_NN> [0] <376> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc135)
    %208 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<16xui64, [@CMX_NN, 0]> loc(#loc30)
    %209 = VPURT.DeclareBuffer <ProfilingOutput> [0] <1024> -> memref<16xui64> loc(#loc30)
    %210 = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register> loc(#loc2)
    %211 = VPURT.DeclareBuffer <ProfilingOutput> [0] <1156> -> memref<1xui32> loc(#loc2)
    %212 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<224xui32> loc(#loc31)
    %213 = VPURT.DeclareBuffer <ProfilingOutput> [0] <896> -> memref<32xui64> loc(#loc31)
    %214 = VPURT.DeclareBuffer <ProfilingOutput> [0] <1152> -> memref<16xui32> loc(#loc31)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%0 : memref<1xui32, @Register>) outputs(%1 : memref<1xui32>) -> memref<1xui32> loc(#loc2)
    } loc(#loc2)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%142 : memref<1xui64, @Register>) outputs(%143 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc126)
    } loc(#loc126)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%54 : memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) outputs(%24 : memref<1x1x2x512xf32, [@CMX_NN, 0]>) -> memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc126)
    } loc(#loc126)
    VPURT.Task updates(%2 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 0 : i64>} inputs(%144 : memref<1xui64, @Register>) outputs(%145 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc126)
    } loc(#loc126)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%146 : memref<1xui64, @Register>) outputs(%147 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc127)
    } loc(#loc127)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%55 : memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) outputs(%25 : memref<1x1x2x512xf32, [@CMX_NN, 1]>) -> memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc127)
    } loc(#loc127)
    VPURT.Task updates(%2 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 8 : i64>} inputs(%148 : memref<1xui64, @Register>) outputs(%149 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc127)
    } loc(#loc127)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Convert inputs(%22 as %arg3: memref<1x1x2x512xf32, [@CMX_NN, 0]>) outputs(%28 as %arg4: memref<1x1x2x512xf16, [@CMX_NN, 0]>) profiling_data(%56 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x2x512xf16, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x2x512xf32, [@CMX_NN, 0]>, memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc136)
    } loc(#loc136)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Convert inputs(%23 as %arg3: memref<1x1x2x512xf32, [@CMX_NN, 1]>) outputs(%29 as %arg4: memref<1x1x2x512xf16, [@CMX_NN, 1]>) profiling_data(%57 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x2x512xf16, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x2x512xf32, [@CMX_NN, 1]>, memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc137)
    } loc(#loc137)
    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%150 : memref<1xui64, @Register>) outputs(%151 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc126)
    } loc(#loc126)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%26 : memref<1x1x2x512xf16, [@CMX_NN, 0]>) outputs(%30 : memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc126)
    } loc(#loc126)
    VPURT.Task updates(%4 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 1 : i64>} inputs(%152 : memref<1xui64, @Register>) outputs(%153 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc126)
    } loc(#loc126)
    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%154 : memref<1xui64, @Register>) outputs(%155 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc127)
    } loc(#loc127)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%27 : memref<1x1x2x512xf16, [@CMX_NN, 1]>) outputs(%31 : memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc127)
    } loc(#loc127)
    VPURT.Task updates(%4 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 9 : i64>} inputs(%156 : memref<1xui64, @Register>) outputs(%157 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc127)
    } loc(#loc127)
    VPURT.Task waits(%4 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%158 : memref<1xui64, @Register>) outputs(%159 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc128)
    } loc(#loc128)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%58 : memref<1x2x512x1xf16, @DDR>) outputs(%32 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc128)
    } loc(#loc128)
    VPURT.Task updates(%5 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 2 : i64>} inputs(%160 : memref<1xui64, @Register>) outputs(%161 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc128)
    } loc(#loc128)
    VPURT.Task waits(%4 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%162 : memref<1xui64, @Register>) outputs(%163 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc129)
    } loc(#loc129)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%59 : memref<1x2x512x1xf16, @DDR>) outputs(%33 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc129)
    } loc(#loc129)
    VPURT.Task updates(%5 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 10 : i64>} inputs(%164 : memref<1xui64, @Register>) outputs(%165 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc129)
    } loc(#loc129)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 1 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%62 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%66 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%118 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc138)
    } loc(#loc138)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 1 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%63 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%67 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%119 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc139)
    } loc(#loc139)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 2 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%60 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%64 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%120 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc140)
    } loc(#loc140)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 2 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%61 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%65 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%121 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc141)
    } loc(#loc141)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 3 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%70 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%74 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%122 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc142)
    } loc(#loc142)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 3 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%71 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%75 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%123 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc143)
    } loc(#loc143)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 4 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%68 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%72 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%124 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc144)
    } loc(#loc144)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 4 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%69 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%73 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%125 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc145)
    } loc(#loc145)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 5 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Swish inputs(%78 as %arg3: memref<1x1x512x1xf16, [@CMX_NN, 0]>) outputs(%82 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%126 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x1x512x1xf16, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc146)
    } loc(#loc146)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 5 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Swish inputs(%79 as %arg3: memref<1x1x512x1xf16, [@CMX_NN, 1]>) outputs(%83 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%127 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x1x512x1xf16, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc147)
    } loc(#loc147)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 6 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Swish inputs(%76 as %arg3: memref<1x1x512x1xf16, [@CMX_NN, 0]>) outputs(%80 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%128 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x1x512x1xf16, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc148)
    } loc(#loc148)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 7 : i64, dataIndex = 6 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Swish inputs(%77 as %arg3: memref<1x1x512x1xf16, [@CMX_NN, 1]>) outputs(%81 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%129 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x1x512x1xf16, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc149)
    } loc(#loc149)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<56xui32, [@CMX_NN, 0]>) outputs(%84 : memref<56xui32, @DDR>) -> memref<56xui32, @DDR> loc(#loc150)
    } loc(#loc150)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64} inputs(%21 : memref<56xui32, [@CMX_NN, 1]>) outputs(%85 : memref<56xui32, @DDR>) -> memref<56xui32, @DDR> loc(#loc151)
    } loc(#loc151)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%88 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%92 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%130 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc152)
    } loc(#loc152)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%89 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%93 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%131 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc153)
    } loc(#loc153)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 1 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%86 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%90 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%132 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc154)
    } loc(#loc154)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 1 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%87 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%91 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%133 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc155)
    } loc(#loc155)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%96 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%100 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%134 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc156)
    } loc(#loc156)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%97 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%101 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%135 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc157)
    } loc(#loc157)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 3 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%94 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%98 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%136 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc158)
    } loc(#loc158)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 3 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_MVN inputs(%95 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%99 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%137 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc159)
    } loc(#loc159)
    VPURT.Task waits(%10 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%166 : memref<1xui64, @Register>) outputs(%167 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc130)
    } loc(#loc130)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%34 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) outputs(%38 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc130)
    } loc(#loc130)
    VPURT.Task updates(%11 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 3 : i64>} inputs(%168 : memref<1xui64, @Register>) outputs(%169 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc130)
    } loc(#loc130)
    VPURT.Task waits(%10 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%170 : memref<1xui64, @Register>) outputs(%171 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc131)
    } loc(#loc131)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%35 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) outputs(%39 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc131)
    } loc(#loc131)
    VPURT.Task updates(%11 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%172 : memref<1xui64, @Register>) outputs(%173 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc131)
    } loc(#loc131)
    VPURT.Task waits(%11 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%174 : memref<1xui64, @Register>) outputs(%175 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc132)
    } loc(#loc132)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%36 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%40 : memref<1x4x256x1xf16, [@CMX_NN, 0]>) -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc132)
    } loc(#loc132)
    VPURT.Task updates(%12 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 4 : i64>} inputs(%176 : memref<1xui64, @Register>) outputs(%177 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc132)
    } loc(#loc132)
    VPURT.Task waits(%11 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%178 : memref<1xui64, @Register>) outputs(%179 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc133)
    } loc(#loc133)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%37 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%41 : memref<1x4x256x1xf16, [@CMX_NN, 1]>) -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc133)
    } loc(#loc133)
    VPURT.Task updates(%12 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 12 : i64>} inputs(%180 : memref<1xui64, @Register>) outputs(%181 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc133)
    } loc(#loc133)
    VPURT.Task waits(%12 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Tanh inputs(%104 as %arg3: memref<1x2x256x1xf16, [@CMX_NN, 0]>) outputs(%108 as %arg4: memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%138 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x2x256x1xf16, [@CMX_NN, 0]>, memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc160)
    } loc(#loc160)
    VPURT.Task waits(%12 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Tanh inputs(%105 as %arg3: memref<1x2x256x1xf16, [@CMX_NN, 1]>) outputs(%109 as %arg4: memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%139 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x2x256x1xf16, [@CMX_NN, 1]>, memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc161)
    } loc(#loc161)
    VPURT.Task waits(%12 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 5 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Tanh inputs(%102 as %arg3: memref<1x2x256x1xf16, [@CMX_NN, 0]>) outputs(%106 as %arg4: memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%140 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x2x256x1xf16, [@CMX_NN, 0]>, memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc162)
    } loc(#loc162)
    VPURT.Task waits(%12 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 5 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Tanh inputs(%103 as %arg3: memref<1x2x256x1xf16, [@CMX_NN, 1]>) outputs(%107 as %arg4: memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%141 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x2x256x1xf16, [@CMX_NN, 1]>, memref<1x2x256x1xf16, {order = #NCHW, strides = [1024, 256, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc163)
    } loc(#loc163)
    VPURT.Task waits(%13 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%182 : memref<1xui64, @Register>) outputs(%183 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc132)
    } loc(#loc132)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%42 : memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs(%44 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc132)
    } loc(#loc132)
    VPURT.Task updates(%14 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 5 : i64>} inputs(%184 : memref<1xui64, @Register>) outputs(%185 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc132)
    } loc(#loc132)
    VPURT.Task waits(%13 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%186 : memref<1xui64, @Register>) outputs(%187 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc133)
    } loc(#loc133)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%43 : memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs(%45 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc133)
    } loc(#loc133)
    VPURT.Task updates(%14 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 13 : i64>} inputs(%188 : memref<1xui64, @Register>) outputs(%189 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc133)
    } loc(#loc133)
    VPURT.Task waits(%14 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%190 : memref<1xui64, @Register>) outputs(%191 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc134)
    } loc(#loc134)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%110 : memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) outputs(%48 : memref<1x1x2x512xf16, [@CMX_NN, 0]>) -> memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc134)
    } loc(#loc134)
    VPURT.Task updates(%15 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 6 : i64>} inputs(%192 : memref<1xui64, @Register>) outputs(%193 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc134)
    } loc(#loc134)
    VPURT.Task waits(%14 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%194 : memref<1xui64, @Register>) outputs(%195 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc135)
    } loc(#loc135)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%111 : memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) outputs(%49 : memref<1x1x2x512xf16, [@CMX_NN, 1]>) -> memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc135)
    } loc(#loc135)
    VPURT.Task updates(%15 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 14 : i64>} inputs(%196 : memref<1xui64, @Register>) outputs(%197 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc135)
    } loc(#loc135)
    VPURT.Task waits(%15 : !VPURT.Barrier) updates(%16 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 6 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Convert inputs(%46 as %arg3: memref<1x1x2x512xf16, [@CMX_NN, 0]>) outputs(%52 as %arg4: memref<1x1x2x512xf32, [@CMX_NN, 0]>) profiling_data(%112 : memref<8xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x2x512xf32, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x2x512xf16, [@CMX_NN, 0]>, memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc164)
    } loc(#loc164)
    VPURT.Task waits(%15 : !VPURT.Barrier) updates(%16 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 14 : i64, clusterSize = 7 : i64, dataIndex = 6 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>} @VPU.SW::@builtin_Convert inputs(%47 as %arg3: memref<1x1x2x512xf16, [@CMX_NN, 1]>) outputs(%53 as %arg4: memref<1x1x2x512xf32, [@CMX_NN, 1]>) profiling_data(%113 : memref<8xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x2x512xf32, [@CMX_NN, 1]>, memref<8xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x2x512xf16, [@CMX_NN, 1]>, memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc165)
    } loc(#loc165)
    VPURT.Task waits(%16 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64} inputs(%18 : memref<56xui32, [@CMX_NN, 0]>) outputs(%114 : memref<56xui32, @DDR>) -> memref<56xui32, @DDR> loc(#loc166)
    } loc(#loc166)
    VPURT.Task waits(%16 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64} inputs(%19 : memref<56xui32, [@CMX_NN, 1]>) outputs(%115 : memref<56xui32, @DDR>) -> memref<56xui32, @DDR> loc(#loc167)
    } loc(#loc167)
    VPURT.Task waits(%16 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%198 : memref<1xui64, @Register>) outputs(%199 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc134)
    } loc(#loc134)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%50 : memref<1x1x2x512xf32, [@CMX_NN, 0]>) outputs(%116 : memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc134)
    } loc(#loc134)
    VPURT.Task {
      %215 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 7 : i64>} inputs(%200 : memref<1xui64, @Register>) outputs(%201 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc134)
    } loc(#loc134)
    VPURT.Task updates(%17 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 0 : i64} inputs(%202 : memref<16xui64, [@CMX_NN, 0]>) outputs(%203 : memref<16xui64>) -> memref<16xui64> loc(#loc29)
    } loc(#loc29)
    VPURT.Task waits(%16 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%204 : memref<1xui64, @Register>) outputs(%205 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc135)
    } loc(#loc135)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%51 : memref<1x1x2x512xf32, [@CMX_NN, 1]>) outputs(%117 : memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc135)
    } loc(#loc135)
    VPURT.Task {
      %215 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 15 : i64>} inputs(%206 : memref<1xui64, @Register>) outputs(%207 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc135)
    } loc(#loc135)
    VPURT.Task updates(%17 : !VPURT.Barrier) {
      %215 = VPUIP.NNDMA {port = 1 : i64} inputs(%208 : memref<16xui64, [@CMX_NN, 0]>) outputs(%209 : memref<16xui64>) -> memref<16xui64> loc(#loc30)
    } loc(#loc30)
    VPURT.Task {
      %215 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%210 : memref<1xui32, @Register>) outputs(%211 : memref<1xui32>) -> memref<1xui32> loc(#loc2)
    } loc(#loc2)
    return %arg1, %arg2 : memref<1x4x512xf32, @DDR>, memref<304xui32> loc(#loc168)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("PROFWORKPOINT_READ")
#loc3 = loc("converted_to_f16")
#loc4 = loc("MVN_0")
#loc5 = loc("t_MVN")
#loc6 = loc("MVN_1")
#loc7 = loc("Swish_0")
#loc8 = loc("t_Swish")
#loc9 = loc("actshaveProfilingCMX2DDR0")
#loc10 = loc("MVN_3")
#loc11 = loc("Tanh_0")
#loc12 = loc("t_Tanh")
#loc13 = loc("Div_0")
#loc14 = loc("t_Reshape")
#loc15 = loc("converted_to_f32")
#loc16 = loc("finishing_barrier")
#loc17 = loc("actshaveProfilingCMX2DDR14")
#loc18 = loc("_input_cluster_0")
#loc19 = loc("_input_cluster_1")
#loc20 = loc("_outputBuff_cluster_0")
#loc21 = loc("_outputBuff_cluster_1")
#loc22 = loc("_profilingBuff_cluster_0")
#loc23 = loc("_profilingBuff_cluster_1")
#loc24 = loc("tile_1")
#loc25 = loc("tile_0")
#loc26 = loc("MVN_2")
#loc27 = loc("_cluster_0")
#loc28 = loc("_cluster_1")
#loc29 = loc("dmaProfilingCMX2DDR0")
#loc30 = loc("dmaProfilingCMX2DDR128")
#loc31 = loc("newProfilingBuffer")
#loc32 = loc("cluster_0")
#loc33 = loc("cluster_1")
#loc34 = loc("output")
#loc35 = loc("t_Output")
#loc36 = loc(fused[#loc4, #loc5])
#loc37 = loc(fused[#loc6, #loc5])
#loc38 = loc(fused[#loc7, #loc8])
#loc39 = loc(fused[#loc10, #loc5])
#loc40 = loc(fused[#loc11, #loc12])
#loc41 = loc(fused[#loc13, #loc14, #loc15])
#loc42 = loc(fused[#loc3, #loc18])
#loc43 = loc(fused[#loc3, #loc19])
#loc44 = loc(fused[#loc3, #loc20])
#loc45 = loc(fused[#loc3, #loc21])
#loc46 = loc(fused[#loc13, #loc14, #loc15, #loc18])
#loc47 = loc(fused[#loc13, #loc14, #loc15, #loc19])
#loc48 = loc(fused[#loc13, #loc14, #loc15, #loc20])
#loc49 = loc(fused[#loc13, #loc14, #loc15, #loc21])
#loc50 = loc(fused[#loc3, #loc22])
#loc51 = loc(fused[#loc3, #loc23])
#loc52 = loc(fused[#loc4, #loc5, #loc24, #loc18])
#loc53 = loc(fused[#loc4, #loc5, #loc24, #loc19])
#loc54 = loc(fused[#loc4, #loc5, #loc25, #loc18])
#loc55 = loc(fused[#loc4, #loc5, #loc25, #loc19])
#loc56 = loc(fused[#loc4, #loc5, #loc24, #loc20])
#loc57 = loc(fused[#loc4, #loc5, #loc24, #loc21])
#loc58 = loc(fused[#loc4, #loc5, #loc25, #loc20])
#loc59 = loc(fused[#loc4, #loc5, #loc25, #loc21])
#loc60 = loc(fused[#loc6, #loc5, #loc24, #loc18])
#loc61 = loc(fused[#loc6, #loc5, #loc24, #loc19])
#loc62 = loc(fused[#loc6, #loc5, #loc25, #loc18])
#loc63 = loc(fused[#loc6, #loc5, #loc25, #loc19])
#loc64 = loc(fused[#loc6, #loc5, #loc24, #loc20])
#loc65 = loc(fused[#loc6, #loc5, #loc24, #loc21])
#loc66 = loc(fused[#loc6, #loc5, #loc25, #loc20])
#loc67 = loc(fused[#loc6, #loc5, #loc25, #loc21])
#loc68 = loc(fused[#loc7, #loc8, #loc24, #loc18])
#loc69 = loc(fused[#loc7, #loc8, #loc24, #loc19])
#loc70 = loc(fused[#loc7, #loc8, #loc25, #loc18])
#loc71 = loc(fused[#loc7, #loc8, #loc25, #loc19])
#loc72 = loc(fused[#loc7, #loc8, #loc24, #loc20])
#loc73 = loc(fused[#loc7, #loc8, #loc24, #loc21])
#loc74 = loc(fused[#loc7, #loc8, #loc25, #loc20])
#loc75 = loc(fused[#loc7, #loc8, #loc25, #loc21])
#loc76 = loc(fused[#loc26, #loc5, #loc24, #loc18])
#loc77 = loc(fused[#loc26, #loc5, #loc24, #loc19])
#loc78 = loc(fused[#loc26, #loc5, #loc25, #loc18])
#loc79 = loc(fused[#loc26, #loc5, #loc25, #loc19])
#loc80 = loc(fused[#loc26, #loc5, #loc24, #loc20])
#loc81 = loc(fused[#loc26, #loc5, #loc24, #loc21])
#loc82 = loc(fused[#loc26, #loc5, #loc25, #loc20])
#loc83 = loc(fused[#loc26, #loc5, #loc25, #loc21])
#loc84 = loc(fused[#loc10, #loc5, #loc24, #loc18])
#loc85 = loc(fused[#loc10, #loc5, #loc24, #loc19])
#loc86 = loc(fused[#loc10, #loc5, #loc25, #loc18])
#loc87 = loc(fused[#loc10, #loc5, #loc25, #loc19])
#loc88 = loc(fused[#loc10, #loc5, #loc24, #loc20])
#loc89 = loc(fused[#loc10, #loc5, #loc24, #loc21])
#loc90 = loc(fused[#loc10, #loc5, #loc25, #loc20])
#loc91 = loc(fused[#loc10, #loc5, #loc25, #loc21])
#loc92 = loc(fused[#loc11, #loc12, #loc24, #loc18])
#loc93 = loc(fused[#loc11, #loc12, #loc24, #loc19])
#loc94 = loc(fused[#loc11, #loc12, #loc25, #loc18])
#loc95 = loc(fused[#loc11, #loc12, #loc25, #loc19])
#loc96 = loc(fused[#loc11, #loc12, #loc24, #loc20])
#loc97 = loc(fused[#loc11, #loc12, #loc24, #loc21])
#loc98 = loc(fused[#loc11, #loc12, #loc25, #loc20])
#loc99 = loc(fused[#loc11, #loc12, #loc25, #loc21])
#loc100 = loc(fused[#loc13, #loc14, #loc15, #loc22])
#loc101 = loc(fused[#loc13, #loc14, #loc15, #loc23])
#loc102 = loc(fused[#loc4, #loc5, #loc25, #loc22])
#loc103 = loc(fused[#loc4, #loc5, #loc25, #loc23])
#loc104 = loc(fused[#loc4, #loc5, #loc24, #loc22])
#loc105 = loc(fused[#loc4, #loc5, #loc24, #loc23])
#loc106 = loc(fused[#loc6, #loc5, #loc25, #loc22])
#loc107 = loc(fused[#loc6, #loc5, #loc25, #loc23])
#loc108 = loc(fused[#loc6, #loc5, #loc24, #loc22])
#loc109 = loc(fused[#loc6, #loc5, #loc24, #loc23])
#loc110 = loc(fused[#loc7, #loc8, #loc25, #loc22])
#loc111 = loc(fused[#loc7, #loc8, #loc25, #loc23])
#loc112 = loc(fused[#loc7, #loc8, #loc24, #loc22])
#loc113 = loc(fused[#loc7, #loc8, #loc24, #loc23])
#loc114 = loc(fused[#loc26, #loc5, #loc25, #loc22])
#loc115 = loc(fused[#loc26, #loc5, #loc25, #loc23])
#loc116 = loc(fused[#loc26, #loc5, #loc24, #loc22])
#loc117 = loc(fused[#loc26, #loc5, #loc24, #loc23])
#loc118 = loc(fused[#loc10, #loc5, #loc25, #loc22])
#loc119 = loc(fused[#loc10, #loc5, #loc25, #loc23])
#loc120 = loc(fused[#loc10, #loc5, #loc24, #loc22])
#loc121 = loc(fused[#loc10, #loc5, #loc24, #loc23])
#loc122 = loc(fused[#loc11, #loc12, #loc25, #loc22])
#loc123 = loc(fused[#loc11, #loc12, #loc25, #loc23])
#loc124 = loc(fused[#loc11, #loc12, #loc24, #loc22])
#loc125 = loc(fused[#loc11, #loc12, #loc24, #loc23])
#loc126 = loc(fused[#loc3, #loc27])
#loc127 = loc(fused[#loc3, #loc28])
#loc128 = loc(fused[#loc4, #loc5, #loc27])
#loc129 = loc(fused[#loc4, #loc5, #loc28])
#loc130 = loc(fused[#loc10, #loc5, #loc27])
#loc131 = loc(fused[#loc10, #loc5, #loc28])
#loc132 = loc(fused[#loc11, #loc12, #loc27])
#loc133 = loc(fused[#loc11, #loc12, #loc28])
#loc134 = loc(fused[#loc13, #loc14, #loc15, #loc27])
#loc135 = loc(fused[#loc13, #loc14, #loc15, #loc28])
#loc136 = loc(fused[#loc3, #loc32])
#loc137 = loc(fused[#loc3, #loc33])
#loc138 = loc(fused[#loc4, #loc5, #loc25, #loc32])
#loc139 = loc(fused[#loc4, #loc5, #loc25, #loc33])
#loc140 = loc(fused[#loc4, #loc5, #loc24, #loc32])
#loc141 = loc(fused[#loc4, #loc5, #loc24, #loc33])
#loc142 = loc(fused[#loc6, #loc5, #loc25, #loc32])
#loc143 = loc(fused[#loc6, #loc5, #loc25, #loc33])
#loc144 = loc(fused[#loc6, #loc5, #loc24, #loc32])
#loc145 = loc(fused[#loc6, #loc5, #loc24, #loc33])
#loc146 = loc(fused[#loc7, #loc8, #loc25, #loc32])
#loc147 = loc(fused[#loc7, #loc8, #loc25, #loc33])
#loc148 = loc(fused[#loc7, #loc8, #loc24, #loc32])
#loc149 = loc(fused[#loc7, #loc8, #loc24, #loc33])
#loc150 = loc(fused[#loc9, #loc27])
#loc151 = loc(fused[#loc9, #loc28])
#loc152 = loc(fused[#loc26, #loc5, #loc25, #loc32])
#loc153 = loc(fused[#loc26, #loc5, #loc25, #loc33])
#loc154 = loc(fused[#loc26, #loc5, #loc24, #loc32])
#loc155 = loc(fused[#loc26, #loc5, #loc24, #loc33])
#loc156 = loc(fused[#loc10, #loc5, #loc25, #loc32])
#loc157 = loc(fused[#loc10, #loc5, #loc25, #loc33])
#loc158 = loc(fused[#loc10, #loc5, #loc24, #loc32])
#loc159 = loc(fused[#loc10, #loc5, #loc24, #loc33])
#loc160 = loc(fused[#loc11, #loc12, #loc25, #loc32])
#loc161 = loc(fused[#loc11, #loc12, #loc25, #loc33])
#loc162 = loc(fused[#loc11, #loc12, #loc24, #loc32])
#loc163 = loc(fused[#loc11, #loc12, #loc24, #loc33])
#loc164 = loc(fused[#loc13, #loc14, #loc15, #loc32])
#loc165 = loc(fused[#loc13, #loc14, #loc15, #loc33])
#loc166 = loc(fused[#loc17, #loc27])
#loc167 = loc(fused[#loc17, #loc28])
#loc168 = loc(fused[#loc34, #loc35])


// CHECK: {"traceEvents":[
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":0, "args": {"name" : "DMA"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":0, "args": {"sort_index" : "0"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":0, "tid":0, "args": {"name" : "DMA"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":0, "tid":1, "args": {"name" : "DMA"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":1, "args": {"name" : "Cluster (0)"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":1, "args": {"sort_index" : "1"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":0, "args": {"name" : "SW / Shave"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":1, "args": {"name" : "SW / Shave"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":2, "args": {"name" : "Cluster (1)"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":2, "args": {"sort_index" : "2"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":0, "args": {"name" : "SW / Shave"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":1, "args": {"name" : "SW / Shave"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":3, "args": {"name" : "Layers"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":3, "args": {"sort_index" : "3"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":3, "tid":0, "args": {"name" : "Layers"}},
// CHECK-NEXT: {"name":"converted_to_f16?_cluster_0", "cat":"DMA", "ph":"X", "ts":0.000, "dur":0.703, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"converted_to_f16?_cluster_1", "cat":"DMA", "ph":"X", "ts":2.864, "dur":0.703, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"converted_to_f16?_cluster_0", "cat":"DMA", "ph":"X", "ts":19.062, "dur":0.312, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"converted_to_f16?_cluster_1", "cat":"DMA", "ph":"X", "ts":19.219, "dur":0.312, "pid":0, "tid":1},
// CHECK-NEXT: {"name":"MVN_0?t_MVN/_cluster_0", "cat":"DMA", "ph":"X", "ts":19.765, "dur":0.520, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"MVN_0?t_MVN/_cluster_1", "cat":"DMA", "ph":"X", "ts":19.922, "dur":0.520, "pid":0, "tid":1},
// CHECK-NEXT: {"name":"MVN_3?t_MVN/_cluster_0", "cat":"DMA", "ph":"X", "ts":48.359, "dur":0.312, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"MVN_3?t_MVN/_cluster_1", "cat":"DMA", "ph":"X", "ts":48.515, "dur":0.312, "pid":0, "tid":1},
// CHECK-NEXT: {"name":"Tanh_0?t_Tanh/_cluster_0", "cat":"DMA", "ph":"X", "ts":49.062, "dur":0.520, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"Tanh_0?t_Tanh/_cluster_1", "cat":"DMA", "ph":"X", "ts":49.219, "dur":0.520, "pid":0, "tid":1},
// CHECK-NEXT: {"name":"Tanh_0?t_Tanh/_cluster_0", "cat":"DMA", "ph":"X", "ts":54.609, "dur":0.312, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"Tanh_0?t_Tanh/_cluster_1", "cat":"DMA", "ph":"X", "ts":54.765, "dur":0.312, "pid":0, "tid":1},
// CHECK-NEXT: {"name":"Div_0?t_Reshape/converted_to_f32/_cluster_0", "cat":"DMA", "ph":"X", "ts":55.312, "dur":0.703, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"Div_0?t_Reshape/converted_to_f32/_cluster_1", "cat":"DMA", "ph":"X", "ts":55.469, "dur":0.703, "pid":0, "tid":1},
// CHECK-NEXT: {"name":"Div_0?t_Reshape/converted_to_f32/_cluster_0", "cat":"DMA", "ph":"X", "ts":60.781, "dur":0.651, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"Div_0?t_Reshape/converted_to_f32/_cluster_1", "cat":"DMA", "ph":"X", "ts":60.937, "dur":0.338, "pid":0, "tid":1},
// CHECK-NEXT: {"name":"converted_to_f16?cluster_0", "cat":"SW", "ph":"X", "ts":12.526, "dur":5.390, "pid":1, "tid":0, "args":{"Active cycles:": "5237", "Stall cycles:": "805"}},
// CHECK-NEXT: {"name":"MVN_0?t_MVN/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":20.833, "dur":9.765, "pid":1, "tid":0, "args":{"Active cycles:": "9496", "Stall cycles:": "1367"}},
// CHECK-NEXT: {"name":"MVN_0?t_MVN/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":21.094, "dur":9.635, "pid":1, "tid":1, "args":{"Active cycles:": "9490", "Stall cycles:": "1715"}},
// CHECK-NEXT: {"name":"MVN_1?t_MVN/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":31.849, "dur":2.708, "pid":1, "tid":0, "args":{"Active cycles:": "2629", "Stall cycles:": "962"}},
// CHECK-NEXT: {"name":"MVN_1?t_MVN/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":31.979, "dur":2.838, "pid":1, "tid":1, "args":{"Active cycles:": "2635", "Stall cycles:": "1042"}},
// CHECK-NEXT: {"name":"Swish_0?t_Swish/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":36.015, "dur":3.098, "pid":1, "tid":0, "args":{"Active cycles:": "2772", "Stall cycles:": "876"}},
// CHECK-NEXT: {"name":"Swish_0?t_Swish/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":36.276, "dur":2.968, "pid":1, "tid":1, "args":{"Active cycles:": "2761", "Stall cycles:": "1075"}},
// CHECK-NEXT: {"name":"MVN_2?t_MVN/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":40.364, "dur":2.812, "pid":1, "tid":0, "args":{"Active cycles:": "2818", "Stall cycles:": "1148"}},
// CHECK-NEXT: {"name":"MVN_2?t_MVN/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":40.625, "dur":2.812, "pid":1, "tid":1, "args":{"Active cycles:": "2536", "Stall cycles:": "1053"}},
// CHECK-NEXT: {"name":"MVN_3?t_MVN/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":44.427, "dur":2.994, "pid":1, "tid":0, "args":{"Active cycles:": "2984", "Stall cycles:": "1203"}},
// CHECK-NEXT: {"name":"MVN_3?t_MVN/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":44.765, "dur":2.916, "pid":1, "tid":1, "args":{"Active cycles:": "2644", "Stall cycles:": "1140"}},
// CHECK-NEXT: {"name":"Tanh_0?t_Tanh/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":50.286, "dur":3.515, "pid":1, "tid":0, "args":{"Active cycles:": "3271", "Stall cycles:": "999"}},
// CHECK-NEXT: {"name":"Tanh_0?t_Tanh/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":50.547, "dur":3.385, "pid":1, "tid":1, "args":{"Active cycles:": "3262", "Stall cycles:": "1242"}},
// CHECK-NEXT: {"name":"Div_0?t_Reshape/converted_to_f32/cluster_0", "cat":"SW", "ph":"X", "ts":56.693, "dur":3.385, "pid":1, "tid":0, "args":{"Active cycles:": "3261", "Stall cycles:": "851"}},
// CHECK-NEXT: {"name":"converted_to_f16?cluster_1", "cat":"SW", "ph":"X", "ts":12.396, "dur":5.390, "pid":2, "tid":0, "args":{"Active cycles:": "5239", "Stall cycles:": "693"}},
// CHECK-NEXT: {"name":"MVN_0?t_MVN/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":20.963, "dur":9.895, "pid":2, "tid":0, "args":{"Active cycles:": "9508", "Stall cycles:": "1488"}},
// CHECK-NEXT: {"name":"MVN_0?t_MVN/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":21.224, "dur":9.765, "pid":2, "tid":1, "args":{"Active cycles:": "9466", "Stall cycles:": "1694"}},
// CHECK-NEXT: {"name":"MVN_1?t_MVN/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":32.109, "dur":2.578, "pid":2, "tid":0, "args":{"Active cycles:": "2562", "Stall cycles:": "1095"}},
// CHECK-NEXT: {"name":"MVN_1?t_MVN/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":32.396, "dur":2.734, "pid":2, "tid":1, "args":{"Active cycles:": "2640", "Stall cycles:": "1013"}},
// CHECK-NEXT: {"name":"Swish_0?t_Swish/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":36.146, "dur":2.708, "pid":2, "tid":0, "args":{"Active cycles:": "2762", "Stall cycles:": "967"}},
// CHECK-NEXT: {"name":"Swish_0?t_Swish/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":36.458, "dur":2.526, "pid":2, "tid":1, "args":{"Active cycles:": "2319", "Stall cycles:": "616"}},
// CHECK-NEXT: {"name":"MVN_2?t_MVN/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":40.104, "dur":2.942, "pid":2, "tid":0, "args":{"Active cycles:": "2859", "Stall cycles:": "963"}},
// CHECK-NEXT: {"name":"MVN_2?t_MVN/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":40.234, "dur":3.072, "pid":2, "tid":1, "args":{"Active cycles:": "2898", "Stall cycles:": "1096"}},
// CHECK-NEXT: {"name":"MVN_3?t_MVN/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":44.297, "dur":2.994, "pid":2, "tid":0, "args":{"Active cycles:": "2909", "Stall cycles:": "1093"}},
// CHECK-NEXT: {"name":"MVN_3?t_MVN/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":44.557, "dur":2.994, "pid":2, "tid":1, "args":{"Active cycles:": "2741", "Stall cycles:": "1082"}},
// CHECK-NEXT: {"name":"Tanh_0?t_Tanh/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":50.156, "dur":3.385, "pid":2, "tid":0, "args":{"Active cycles:": "3295", "Stall cycles:": "719"}},
// CHECK-NEXT: {"name":"Tanh_0?t_Tanh/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":50.416, "dur":3.255, "pid":2, "tid":1, "args":{"Active cycles:": "3266", "Stall cycles:": "1056"}},
// CHECK-NEXT: {"name":"Div_0?t_Reshape/converted_to_f32/cluster_1", "cat":"SW", "ph":"X", "ts":56.562, "dur":3.385, "pid":2, "tid":0, "args":{"Active cycles:": "3267", "Stall cycles:": "688"}},
// CHECK-NEXT: {"name":"converted_to_f16", "cat":"Layer", "ph":"X", "ts":0.000, "dur":19.531, "pid":3, "tid":0, "args":{"Layer type": "<unknown>", "Shave time:": "10us 780ns", "DMA time:": "2us 30ns"}},
// CHECK-NEXT: {"name":"MVN_0", "cat":"Layer", "ph":"X", "ts":19.765, "dur":11.224, "pid":3, "tid":0, "args":{"Layer type": "MVN", "Shave time:": "39us 60ns", "DMA time:": "1us 40ns"}},
// CHECK-NEXT: {"name":"MVN_1", "cat":"Layer", "ph":"X", "ts":31.849, "dur":3.281, "pid":3, "tid":0, "args":{"Layer type": "MVN", "Shave time:": "10us 858ns"}},
// CHECK-NEXT: {"name":"Swish_0", "cat":"Layer", "ph":"X", "ts":36.015, "dur":3.229, "pid":3, "tid":0, "args":{"Layer type": "Swish", "Shave time:": "11us 300ns"}},
// CHECK-NEXT: {"name":"MVN_2", "cat":"Layer", "ph":"X", "ts":40.104, "dur":3.333, "pid":3, "tid":0, "args":{"Layer type": "MVN", "Shave time:": "11us 638ns"}},
// CHECK-NEXT: {"name":"MVN_3", "cat":"Layer", "ph":"X", "ts":44.297, "dur":4.530, "pid":3, "tid":0, "args":{"Layer type": "MVN", "Shave time:": "11us 898ns", "DMA time:": "624ns"}},
// CHECK-NEXT: {"name":"Tanh_0", "cat":"Layer", "ph":"X", "ts":49.062, "dur":6.015, "pid":3, "tid":0, "args":{"Layer type": "Tanh", "Shave time:": "13us 540ns", "DMA time:": "1us 664ns"}},
// CHECK-NEXT: {"name":"Div_0", "cat":"Layer", "ph":"X", "ts":55.312, "dur":6.120, "pid":3, "tid":0, "args":{"Layer type": "Reshape", "Shave time:": "6us 770ns", "DMA time:": "2us 395ns"}}
// CHECK-NEXT: ],
// CHECK-NEXT: "taskStatistics": {
// CHECK-NEXT: "total duration":61.432,
// CHECK-NEXT: "DMA duration":5.676,
// CHECK-NEXT: "DPU duration":0.000,
// CHECK-NEXT: "SW duration":36.195,
// CHECK-NEXT: "M2I duration":0.000,
// CHECK-NEXT: "DMA-DPU overlap":0.000,
// CHECK-NEXT: "DMA-SW overlap":0.000,
// CHECK-NEXT: "SW-DPU overlap":0.000,
// CHECK-NEXT: "all tasks union":41.871,
// CHECK-NEXT: "total idle":19.561,
// CHECK-NEXT: "SW duration without DPU overlap":36.195,
// CHECK-NEXT: "DMA duration without overlaps":5.676,
// CHECK-NEXT: "Sum of DMA task durations":7.753,
// CHECK-NEXT: "Sum of DPU task durations":0.000,
// CHECK-NEXT: "Sum of SW task durations":115.844,
// CHECK-NEXT: "Sum of M2I task durations":0.000
// CHECK-NEXT: },
// CHECK-NEXT: "workpoint": { "freq": 1300.0, "status": "OK" },
// CHECK-NEXT: "displayTimeUnit": "ns"
// CHECK-NEXT: }

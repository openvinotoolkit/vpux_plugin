//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --ungroup-bounded-buffers --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

module @TestCopy attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {
  // CHECK-LABEL: main
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_213" : tensor<2x4x20x20xf16>
    DataInfo "vpu_shape_Parameter_213" : tensor<4xsi32>
    // CHECK: DataInfo "Parameter_213" : tensor<2x4x20x20xf16>
    // CHECK: DataInfo "vpu_shape_Parameter_213" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "Relu_214" : tensor<2x4x20x20xf16>
    DataInfo "vpu_shape_Relu_214" : tensor<4xsi32>
    // CHECK: DataInfo "Relu_214" : tensor<2x4x20x20xf16>
    // CHECK: DataInfo "vpu_shape_Relu_214" : tensor<4xsi32>
  }

  // CHECK-LABEL: main
  func.func @main(%arg0: memref<2x4x20x20xf16>, %arg1: memref<4xsi32>, %arg2: memref<2x4x20x20xf16>, %arg3: memref<4xsi32>) -> (memref<2x4x20x20xf16>, memref<4xsi32>) {
    // CHECK-SAME: [[IN_DATA:%.+]]: memref<2x4x20x20xf16>, [[IN_SHAPE:%.+]]: memref<4xsi32>,
    // CHECK-SAME: [[OUT_DATA:%.+]]: memref<2x4x20x20xf16>, [[OUT_SHAPE:%.+]]: memref<4xsi32>

    %DATA = memref.alloc() : memref<2x4x20x20xf16>
    %SHAPE = memref.alloc() : memref<4xsi32>
    // CHECK: [[DATA:%.*]] = memref.alloc
    // CHECK: [[SHAPE:%.*]] = memref.alloc

    %IN_BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%arg0, %arg1) :
        memref<2x4x20x20xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    // CHECKA: [[IN_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[IN_DATA]], [[IN_SHAPE]])
    %BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%DATA, %SHAPE) :
        memref<2x4x20x20xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    // CHECKA: [[BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[DATA]], [[SHAPE]])

    %COPY = VPUIP.Copy inputs(%IN_BOUNDED_BUFFER: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>)
                       outputs (%BOUNDED_BUFFER: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>)
                       -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    // CHECKA: [[DATA1:%.*]], [[SHAPE1:%.*]] = VPUIP.UngroupBoundedBuffer([[IN_BOUNDED_BUFFER]])
    // CHECKA: [[DATA2:%.*]], [[SHAPE2:%.*]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFFER]])
    // CHECK: [[DATA_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[IN_DATA]]
    // CHECK-SAME: outputs([[DATA]]
    // CHECK: [[SHAPE_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[IN_SHAPE]]
    // CHECK-SAME: outputs([[SHAPE]]
    // CHECKA: [[COPY_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[DATA_COPY]], [[SHAPE_COPY]])

    %OUT_DATA, %OUT_SHAPE = VPUIP.UngroupBoundedBuffer(%COPY) :
        !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
        -> memref<2x4x20x20xf16>, memref<4xsi32>
    // CHECKA: [[DDR_OUT_DATA:%.*]], [[DDR_OUT_SHAPE:%.*]] = VPUIP.UngroupBoundedBuffer

    %RESULT_DATA = VPUIP.Copy inputs(%OUT_DATA: memref<2x4x20x20xf16>) outputs(%arg2 : memref<2x4x20x20xf16>) -> memref<2x4x20x20xf16>
    %RESULT_SHAPE = VPUIP.Copy inputs(%OUT_SHAPE: memref<4xsi32>) outputs(%arg3 : memref<4xsi32>) -> memref<4xsi32>
    // CHECK: [[DATA_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[DATA_COPY]]
    // CHECK-SAME: outputs([[OUT_DATA]]
    // CHECK: [[SHAPE_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[SHAPE_COPY]]
    // CHECK-SAME: outputs([[OUT_SHAPE]]

    return %RESULT_DATA, %RESULT_SHAPE: memref<2x4x20x20xf16>, memref<4xsi32>
    // CHECK: return [[DATA_RESULT]], [[SHAPE_RESULT]]
  }
}

// -----

module @TestSwKernel attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]

  module @VPU.SW {
    func.func private @builtin_ReLU(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "relu_fp16.cpp", VPU.kernel_entry = "relu_fp16"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  // CHECK-LABEL: main
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_213" : tensor<2x4x20x20xf16>
    DataInfo "vpu_shape_Parameter_213" : tensor<4xsi32>
    // CHECK: DataInfo "Parameter_213" : tensor<2x4x20x20xf16>
    // CHECK: DataInfo "vpu_shape_Parameter_213" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "Relu_214" : tensor<2x4x20x20xf16>
    DataInfo "vpu_shape_Relu_214" : tensor<4xsi32>
    // CHECK: DataInfo "Relu_214" : tensor<2x4x20x20xf16>
    // CHECK: DataInfo "vpu_shape_Relu_214" : tensor<4xsi32>
  }

  // CHECK-LABEL: main
  func.func @main(%arg0: memref<2x4x20x20xf16>, %arg1: memref<4xsi32>, %arg2: memref<2x4x20x20xf16>, %arg3: memref<4xsi32>) -> (memref<2x4x20x20xf16>, memref<4xsi32>) {
    // CHECK-SAME: [[IN_DATA:%.+]]: memref<2x4x20x20xf16>, [[IN_SHAPE:%.+]]: memref<4xsi32>,
    // CHECK-SAME: [[OUT_DATA:%.+]]: memref<2x4x20x20xf16>, [[OUT_SHAPE:%.+]]: memref<4xsi32>

    %IN_BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%arg0, %arg1) :
        memref<2x4x20x20xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>

    // CMX Input
    %ALLOC0 = memref.alloc() : memref<2x4x20x20xf16, [@CMX_NN, 0]>
    %ALLOC1 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[ALLOC0:%.*]] = memref.alloc
    // CHECK: [[ALLOC1:%.*]] = memref.alloc
    %CMX_IN_BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%ALLOC0, %ALLOC1) :
        memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<4xsi32,[@CMX_NN, 0]>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    %COPY_IN = VPUIP.Copy inputs(%IN_BOUNDED_BUFFER : !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>)
                          outputs(%CMX_IN_BOUNDED_BUFFER : !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
                          -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK: [[COPY_IN:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[IN_DATA]]
    // CHECK-SAME: outputs([[ALLOC0]]
    // CHECK: [[SHAPE_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[IN_SHAPE]]
    // CHECK-SAME: outputs([[ALLOC1]]

    %ALLOC2 = memref.alloc() : memref<2x4x20x20xf16, [@CMX_NN, 0]>
    %ALLOC3 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[ALLOC2:%.*]] = memref.alloc
    // CHECK: [[ALLOC3:%.*]] = memref.alloc
    %CMX_OUT_BOUNDED_BUFFER = VPUIP.GroupBoundedBuffer(%ALLOC2, %ALLOC3) :
        memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    %KERNEL_OUT = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReLU
        inputs(%CMX_IN_BOUNDED_BUFFER as %arg4: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
        outputs(%CMX_OUT_BOUNDED_BUFFER as %arg5: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) on tile 0
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>{
            VPUIP.SW.Kernel.run(%arg4, %arg5) : !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>,
                                                !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
        }
    // CHECK: [[KERNEL_OUT:%.*]], [[OUTPUT_DIMS:%.*]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>} @VPU.SW::@builtin_ReLU
    // CHECK-SAME: inputs([[ALLOC0]] as %arg4
    // CHECK-SAME: outputs([[ALLOC2]] as %arg5
    // CHECK-SAME: -> (memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>){
    // CHECK:   VPUIP.SW.Kernel.run(%arg4, %arg5) : memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<2x4x20x20xf16, [@CMX_NN, 0]>
    // CHECK: }

    %ALLOC4 = memref.alloc() : memref<2x4x20x20xf16>
    %ALLOC5 = memref.alloc() : memref<4xsi32>
    // CHECK: [[ALLOC4:%.*]] = memref.alloc
    // CHECK: [[ALLOC5:%.*]] = memref.alloc
    %OUTPUT = VPUIP.GroupBoundedBuffer(%ALLOC4, %ALLOC5) :
        memref<2x4x20x20xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    %COPY_OUTPUT  = VPUIP.Copy inputs(%KERNEL_OUT: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
                          outputs(%OUTPUT: !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>)
                          -> !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK: [[DATA_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[KERNEL_OUT]]
    // CHECK-SAME: outputs([[ALLOC4]]
    // CHECK: [[SHAPE_COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[OUTPUT_DIMS]]
    // CHECK-SAME: outputs([[ALLOC5]]

    %OUT_DATA, %OUT_SHAPE = VPUIP.UngroupBoundedBuffer(%COPY_OUTPUT) :
        !VPUIP.BoundedBuffer<data=memref<2x4x20x20xf16>, dynamic_shape=memref<4xsi32>>
        -> memref<2x4x20x20xf16>, memref<4xsi32>

    %RESULT_DATA = VPUIP.Copy inputs(%OUT_DATA: memref<2x4x20x20xf16>) outputs(%arg2 : memref<2x4x20x20xf16>) -> memref<2x4x20x20xf16>
    %RESULT_SHAPE = VPUIP.Copy inputs(%OUT_SHAPE: memref<4xsi32>) outputs(%arg3 : memref<4xsi32>) -> memref<4xsi32>
    // CHECK: [[DATA_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[DATA_COPY]]
    // CHECK-SAME: outputs([[OUT_DATA]]
    // CHECK: [[SHAPE_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[SHAPE_COPY]]
    // CHECK-SAME: outputs([[OUT_SHAPE]]

    return %RESULT_DATA, %RESULT_SHAPE: memref<2x4x20x20xf16>, memref<4xsi32>
    // CHECK: return [[DATA_RESULT]], [[SHAPE_RESULT]]
  }
}

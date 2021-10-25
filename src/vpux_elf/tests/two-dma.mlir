#map = affine_map<(d0, d1, d2, d3) -> (d0 * 24 + d1 * 12 + d2 * 4 + d3)>
module @Convert attributes {VPUIP.arch = "KMB"}  {
  IERT.RunTimeResources availableMemory :  {
    MemoryResource 524288000 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
    MemoryResource 917504 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
  } usedMemory :  {
    MemoryResource 48 bytes of "DDR"
  } executors :  {
    ExecutorResource 1 of "DMA_NN" 
    ExecutorResource 16 of "SHAVE_UPA" 
    ExecutorResource {VPUIP.processorFrequency = 7.000000e+02 : f64} 4 of "NCE_Cluster"  {
      ExecutorResource 5 of "NCE_PerClusterDPU" 
    }
  }
  VPUIP.Graph options : "NONE" version : {contextStr = "VPUX Compiler", hash = "", majorV = 3 : i64, minorV = 11 : i64, patchV = 0 : i64}
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "Parameter_6" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "Convert_7" : tensor<1x2x3x4xf16>
  }
  func @main(%arg0: memref<1x2x3x4xf16, #map>, %arg1: memref<1x2x3x4xf16, #map>) -> memref<1x2x3x4xf16, #map> {
    %0 = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x2x3x4xf16, #map>
    %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, #map>) outputs(%1 : memref<1x2x3x4xf16, #map>) updates(%0 : !VPUIP.Barrier) -> memref<1x2x3x4xf16, #map>
    %3 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x2x3x4xf16, #map>
    %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x2x3x4xf16, #map>) outputs(%arg1 : memref<1x2x3x4xf16, #map>) waits(%0 : !VPUIP.Barrier) -> memref<1x2x3x4xf16, #map>
    return %arg1 : memref<1x2x3x4xf16, #map>
  }
}

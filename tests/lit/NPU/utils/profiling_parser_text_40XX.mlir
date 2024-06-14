//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-40XX.bin -f text | FileCheck %s
// REQUIRES: arch-VPUX40XX

//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_1                         Time(us): 0.20         Start(us): 0.00    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_2                         Time(us): 0.15         Start(us): 0.00    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_3                         Time(us): 0.15         Start(us): 0.00    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_4                         Time(us): 0.15         Start(us): 0.00    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_5                         Time(us): 0.15         Start(us): 0.00    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_0                         Time(us): 0.20         Start(us): 3.23    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0            Time(us): 1.89         Start(us): 7.32    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1            Time(us): 1.89         Start(us): 7.32    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_2            Time(us): 1.89         Start(us): 7.32    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_3            Time(us): 1.89         Start(us): 7.32    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_4            Time(us): 1.89         Start(us): 7.32    
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_5            Time(us): 1.89         Start(us): 7.32    
//CHECK: Task(DPU): relu1?t_Relu/cluster_5                                              Time(us): 1.52         Start(us): 9.25    
//CHECK: Task(DPU): relu1?t_Relu/cluster_0                                              Time(us): 1.66         Start(us): 9.27    
//CHECK: Task(DPU): relu1?t_Relu/cluster_2                                              Time(us): 1.66         Start(us): 9.27    
//CHECK: Task(DPU): relu1?t_Relu/cluster_1                                              Time(us): 1.66         Start(us): 9.28    
//CHECK: Task(DPU): relu1?t_Relu/cluster_4                                              Time(us): 1.65         Start(us): 9.28    
//CHECK: Task(DPU): relu1?t_Relu/cluster_3                                              Time(us): 1.60         Start(us): 9.29    
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_2                         Time(us): 3.64         Start(us): 10.94   
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_3                         Time(us): 3.64         Start(us): 10.94   
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_0                         Time(us): 3.64         Start(us): 10.94   
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_1                         Time(us): 3.64         Start(us): 10.94   
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_5                         Time(us): 1.84         Start(us): 10.97   
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_4                         Time(us): 1.84         Start(us): 10.97   
//CHECK: Task(DPU): relu2?t_Relu/cluster_1                                              Time(us): 0.98         Start(us): 14.59   
//CHECK: Task(DPU): relu2?t_Relu/cluster_4                                              Time(us): 0.98         Start(us): 14.59   
//CHECK: Task(DPU): relu2?t_Relu/cluster_2                                              Time(us): 0.98         Start(us): 14.59   
//CHECK: Task(DPU): relu2?t_Relu/cluster_3                                              Time(us): 0.98         Start(us): 14.59   
//CHECK: Task(DPU): relu2?t_Relu/cluster_5                                              Time(us): 0.98         Start(us): 14.59   
//CHECK: Task(DPU): relu2?t_Relu/cluster_0                                              Time(us): 0.94         Start(us): 14.63   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_5               Time(us): 9.84          Cycles:0(2394)  Start(us): 36.72   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_2               Time(us): 9.84          Cycles:0(2149)  Start(us): 36.72   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_3               Time(us): 9.90          Cycles:0(2595)  Start(us): 36.72   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_4               Time(us): 9.84          Cycles:0(2757)  Start(us): 36.72   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_2               Time(us): 10.16         Cycles:0(2715)  Start(us): 36.98   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_4               Time(us): 9.58          Cycles:0(2772)  Start(us): 36.98   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_0               Time(us): 9.58          Cycles:0(2400)  Start(us): 36.98   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_0               Time(us): 9.95          Cycles:0(2938)  Start(us): 37.18   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_1               Time(us): 9.38          Cycles:0(2430)  Start(us): 37.18   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_0/cluster_3               Time(us): 9.43          Cycles:0(2940)  Start(us): 37.18   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_1               Time(us): 9.38          Cycles:0(2929)  Start(us): 37.18   
//CHECK: Task(SW): relu2?t_Relu/converted_to_f32/tile_1/cluster_5               Time(us): 9.38          Cycles:0(2567)  Start(us): 37.18   
//CHECK: Layer: conv1/WithoutBiases                      Type: Convolution          DPU: 12.38    SW: 0.00     DMA: 0.00        Start: 0.00
//CHECK: Layer: relu1                                    Type: Relu                 DPU: 9.76     SW: 0.00     DMA: 0.00        Start: 9.25
//CHECK: Layer: conv2/WithoutBiases                      Type: Convolution          DPU: 18.25    SW: 0.00     DMA: 0.00        Start: 10.94
//CHECK: Layer: relu2                                    Type: Relu                 DPU: 5.83     SW: 116.25   DMA: 0.00        Start: 14.59
//CHECK: Total time: 162.46us, Real: 47.13us

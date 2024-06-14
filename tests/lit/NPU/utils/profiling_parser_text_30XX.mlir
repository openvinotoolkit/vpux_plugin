//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %data_path_npu%/profiling-30XX.mlir.txt
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-30XX.bin -f text | FileCheck %s
// REQUIRES: arch-VPUX30XX

//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_expand_copy_1_13                 Time(us): 17.50    Start(us): 0.00
//CHECK: Task(UPA): conv1/WithoutBiases?t_Convolution                                   Time(us): 37.68    Cycles:136370(68247)    Start(us): 23.17
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                                   Time(us): 4.03     Start(us): 65.59
//CHECK: Task(UPA): conv1/WithoutBiases?t_Convolution                                   Time(us): 154.49   Cycles:221617(1141568)  Start(us): 69.63
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]     Time(us): 1.36          Start(us): 69.85
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0          Time(us): 2.58     Start(us): 228.86
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1          Time(us): 2.85     Start(us): 231.59
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_2          Time(us): 2.72     Start(us): 234.59
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_3          Time(us): 2.45     Start(us): 237.46
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0        Time(us): 42.81    Start(us): 239.92
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1        Time(us): 43.78    Start(us): 240.03
//CHECK: Task(DMA): pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]     Time(us): 0.56     Start(us): 240.06
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2        Time(us): 45.58    Start(us): 240.23
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3        Time(us): 48.82    Start(us): 240.36
//CHECK: Task(DPU): pool1?t_MaxPool/cluster_1                                           Time(us): 7.21     Start(us): 289.56
//CHECK: Task(DPU): pool1?t_MaxPool/cluster_3                                           Time(us): 5.39     Start(us): 289.64
//CHECK: Task(DPU): pool1?t_MaxPool/cluster_2                                           Time(us): 8.78     Start(us): 289.67
//CHECK: Task(DPU): pool1?t_MaxPool/cluster_0                                           Time(us): 6.86     Start(us): 289.83
//CHECK: Task(DMA): pool1?t_MaxPool/_cluster_0                                                     Time(us): 2.02     Start(us): 299.67
//CHECK: Task(DMA): pool1?t_MaxPool/_cluster_1                                                     Time(us): 2.02     Start(us): 301.90
//CHECK: Task(DMA): pool1?t_MaxPool/_cluster_2                                                     Time(us): 2.15     Start(us): 304.28
//CHECK: Task(DMA): pool1?t_MaxPool/_cluster_3                                                     Time(us): 1.45     Start(us): 306.67
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 6.50     Start(us): 308.39
//CHECK: Task(DMA): pool1?t_MaxPool/_unrolled_permuteDMA                                Time(us): 36.39    Start(us): 315.04
//CHECK: Task(DMA): pool1?t_MaxPool/_unrolled_permuteDMA                                Time(us): 36.39    Start(us): 351.58
//CHECK: Task(DMA): pool1?t_MaxPool/_unrolled_permuteDMA                                Time(us): 36.40    Start(us): 388.11
//CHECK: Task(DMA): pool1?t_MaxPool/_unrolled_permuteDMA                                Time(us): 18.85    Start(us): 424.66
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 7.10     Start(us): 443.65
//CHECK: Task(UPA): output?t_Output                                                     Time(us): 63.48    Cycles:186916(229204)   Start(us): 454.15
//CHECK: Layer: conv1/WithoutBiases                      Type: Convolution          DPU: 181.00    SW: 192.17   DMA: 33.49          Start: 0.00
//CHECK: Layer: pool1                                    Type: MaxPool              DPU: 28.24     SW: 0.00     DMA: 149.84         Start: 240.06
//CHECK: Layer: output                                   Type: Convert              DPU: 0.00     SW: 63.48    DMA: 0.00           Start: 454.15
//CHECK: Total time: 648.23us, Real: 517.64us

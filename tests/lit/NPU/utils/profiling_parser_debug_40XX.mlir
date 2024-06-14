//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-40XX.bin -f debug | FileCheck %s
// REQUIRES: arch-VPUX40XX

//CHECK:    Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset    IDU dur         IDU tstamp  IDU WL ID  IDU DPU ID    ODU dur         ODU tstamp  ODU WL ID  ODU DPU ID  Task
//CHECK:        0       0           dpu          0                  0                  0         bb           239b9eab         10           0        172   239b9ead                 10          0  conv1/WithoutBiases?t_Convolution/cluster_0/variant_0
//CHECK:        1      20           dpu          0                  0                 20        9ee           239b9f12         11           0        dc2   239b9f1c                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0/variant_0
//CHECK:        2      40           dpu          0                  0                 40        664           239b9f2e         12           0        7d8   239b9f32                 12          0  relu1?t_Relu/cluster_0/variant_0
//CHECK:        3      60           dpu          0                  0                 60        320           239b9f3a         13           0        40d   239b9f3d                 13          0  relu1?t_Relu/cluster_0/variant_1
//CHECK:        4      80           dpu          0                  0                 80       16f1           239b9f7a         14           0       1a1e   239b9f83                 14          0  conv2/WithoutBiases?t_Convolution/cluster_0/variant_0
//CHECK:        5      a0           dpu          0                  0                 a0        306           239b9f8c         15           0        6ee   239b9f96                 15          0  relu2?t_Relu/cluster_0/variant_0
//CHECK:        6      c0           dpu          0                  1                  0         ba           239b9e6d         10           0        172   239b9e6f                 10          0  conv1/WithoutBiases?t_Convolution/cluster_1/variant_0
//CHECK:        7      e0           dpu          0                  1                 20        9ed           239b9f12         11           0        dc1   239b9f1c                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1/variant_0
//CHECK:        8     100           dpu          0                  1                 40        653           239b9f2e         12           0        7ca   239b9f32                 12          0  relu1?t_Relu/cluster_1/variant_0
//CHECK:        9     120           dpu          0                  1                 60        324           239b9f3a         13           0        40d   239b9f3d                 13          0  relu1?t_Relu/cluster_1/variant_1
//CHECK:       10     140           dpu          0                  1                 80       16f1           239b9f7a         14           0       1a1e   239b9f83                 14          0  conv2/WithoutBiases?t_Convolution/cluster_1/variant_0
//CHECK:       11     160           dpu          0                  1                 a0        2f4           239b9f8b         15           0        6d8   239b9f96                 15          0  relu2?t_Relu/cluster_1/variant_0
//CHECK:       12     180           dpu          0                  2                  0         b9           239b9e6d         10           0        166   239b9e6e                 10          0  conv1/WithoutBiases?t_Convolution/cluster_2/variant_0
//CHECK:       13     1a0           dpu          0                  2                 20        9ed           239b9f12         11           0        dc1   239b9f1c                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_2/variant_0
//CHECK:       14     1c0           dpu          0                  2                 40        65b           239b9f2e         12           0        7d1   239b9f32                 12          0  relu1?t_Relu/cluster_2/variant_0
//CHECK:       15     1e0           dpu          0                  2                 60        322           239b9f3a         13           0        40c   239b9f3d                 13          0  relu1?t_Relu/cluster_2/variant_1
//CHECK:       16     200           dpu          0                  2                 80       16f4           239b9f7a         14           0       1a21   239b9f83                 14          0  conv2/WithoutBiases?t_Convolution/cluster_2/variant_0
//CHECK:       17     220           dpu          0                  2                 a0        2f1           239b9f8b         15           0        6d5   239b9f96                 15          0  relu2?t_Relu/cluster_2/variant_0
//CHECK:       18     240           dpu          0                  3                  0         b9           239b9e6d         10           0        166   239b9e6e                 10          0  conv1/WithoutBiases?t_Convolution/cluster_3/variant_0
//CHECK:       19     260           dpu          0                  3                 20        9ed           239b9f12         11           0        dc1   239b9f1c                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_3/variant_0
//CHECK:       20     280           dpu          0                  3                 40        644           239b9f2e         12           0        7bf   239b9f32                 12          0  relu1?t_Relu/cluster_3/variant_0
//CHECK:       21     2a0           dpu          0                  3                 60        319           239b9f3a         13           0        404   239b9f3c                 13          0  relu1?t_Relu/cluster_3/variant_1
//CHECK:       22     2c0           dpu          0                  3                 80       16f4           239b9f7a         14           0       1a21   239b9f83                 14          0  conv2/WithoutBiases?t_Convolution/cluster_3/variant_0
//CHECK:       23     2e0           dpu          0                  3                 a0        2ed           239b9f8b         15           0        6d2   239b9f96                 15          0  relu2?t_Relu/cluster_3/variant_0
//CHECK:       24     300           dpu          0                  4                  0         b9           239b9e6d         10           0        165   239b9e6e                 10          0  conv1/WithoutBiases?t_Convolution/cluster_4/variant_0
//CHECK:       25     320           dpu          0                  4                 20        9ec           239b9f12         11           0        dc0   239b9f1c                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_4/variant_0
//CHECK:       26     340           dpu          0                  4                 40        651           239b9f2e         12           0        7cd   239b9f32                 12          0  relu1?t_Relu/cluster_4/variant_0
//CHECK:       27     360           dpu          0                  4                 60        320           239b9f3a         13           0        40d   239b9f3d                 13          0  relu1?t_Relu/cluster_4/variant_1
//CHECK:       28     380           dpu          0                  4                 80        b65           239b9f5c         14           0        d64   239b9f61                 14          0  conv2/WithoutBiases?t_Convolution/cluster_4/variant_0
//CHECK:       29     3a0           dpu          0                  4                 a0        2f4           239b9f8b         15           0        6d8   239b9f96                 15          0  relu2?t_Relu/cluster_4/variant_0
//CHECK:       30     3c0           dpu          0                  5                  0         b9           239b9e6d         10           0        165   239b9e6e                 10          0  conv1/WithoutBiases?t_Convolution/cluster_5/variant_0
//CHECK:       31     3e0           dpu          0                  5                 20        9eb           239b9f12         11           0        dbf   239b9f1c                 11          0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_5/variant_0
//CHECK:       32     400           dpu          0                  5                 40        5c0           239b9f2c         12           0        723   239b9f30                 12          0  relu1?t_Relu/cluster_5/variant_0
//CHECK:       33     420           dpu          0                  5                 60        2e7           239b9f38         13           0        3bf   239b9f3a                 13          0  relu1?t_Relu/cluster_5/variant_1
//CHECK:       34     440           dpu          0                  5                 80        b6a           239b9f5c         14           0        d69   239b9f61                 14          0  conv2/WithoutBiases?t_Convolution/cluster_5/variant_0
//CHECK:       35     460           dpu          0                  5                 a0        2ed           239b9f8b         15           0        6d0   239b9f96                 15          0  relu2?t_Relu/cluster_5/variant_0

//CHECK:    Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset              Begin   Duration      Stall   Executed      Clock     Branch  Task
//CHECK:        0     480      actshave          0                  0                  0           239ba135         bf        b7a        f31       295d         26  relu2?t_Relu/converted_to_f32/tile_0/cluster_0
//CHECK:        1     4a0      actshave          0                  0                 20           239ba131         b8        960        8a1       2753         18  relu2?t_Relu/converted_to_f32/tile_1/cluster_0
//CHECK:        2     4c0      actshave          0                  1                  0           239ba135         b4        97e        8a1       2662         18  relu2?t_Relu/converted_to_f32/tile_0/cluster_1
//CHECK:        3     4e0      actshave          0                  1                 20           239ba135         b4        b71        8a1       2665         18  relu2?t_Relu/converted_to_f32/tile_1/cluster_1
//CHECK:        4     500      actshave          0                  2                  0           239ba131         c3        a9b        f31       29aa         26  relu2?t_Relu/converted_to_f32/tile_0/cluster_2
//CHECK:        5     520      actshave          0                  2                 20           239ba12c         bd        865        8a1       2862         18  relu2?t_Relu/converted_to_f32/tile_1/cluster_2
//CHECK:        6     540      actshave          0                  3                  0           239ba135         b5        b7c        8a1       2674         18  relu2?t_Relu/converted_to_f32/tile_0/cluster_3
//CHECK:        7     560      actshave          0                  3                 20           239ba12c         be        a23        8a1       2866         18  relu2?t_Relu/converted_to_f32/tile_1/cluster_3
//CHECK:        8     580      actshave          0                  4                  0           239ba131         b8        ad4        8a1       275d         18  relu2?t_Relu/converted_to_f32/tile_0/cluster_4
//CHECK:        9     5a0      actshave          0                  4                 20           239ba12c         bd        ac5        8a1       2866         18  relu2?t_Relu/converted_to_f32/tile_1/cluster_4
//CHECK:       10     5c0      actshave          0                  5                  0           239ba12c         bd        95a        8a1       2868         18  relu2?t_Relu/converted_to_f32/tile_0/cluster_5
//CHECK:       11     5e0      actshave          0                  5                 20           239ba135         b4        a07        8a1       267d         18  relu2?t_Relu/converted_to_f32/tile_1/cluster_5

//CHECK:    Index  Offset        Engine        PLL Value          CFGID
//CHECK:        0     600           pll               4a              6
//CHECK:        1     604           pll               4a              6

//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegister() {
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<UINT test at 0 size 8 = 0>)
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<SINT test at 0 size 8 = 0xFF>)
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<FP test at 0 size 64 = 0x200000>)
    return
  }

// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<UINT test at 0 size 8 = 0>)
// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<SINT test at 0 size 8 = 0xFF>)
// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<FP test at 0 size 64 = 0x200000>)

// -----

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegisterRequiresVersion() {
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<UINT test at 0 size 8 = 0>)
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<SINT test at 0 size 8 = 0xFF requires 1:2:3>)
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<FP test at 0 size 64 = 0x200000 requires 4:5:6>)
    return
  }

// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<UINT test at 0 size 8 = 0>)
// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<SINT test at 0 size 8 = 0xFF requires 1:2:3>)
// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<FP test at 0 size 64 = 0x200000 requires 4:5:6>)

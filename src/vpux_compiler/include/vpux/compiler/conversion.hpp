//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"  // E#106904: IERT doesn't have a dialect header
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>

#include <memory>

namespace vpux {

//
// LowerIE2IERT
//

//
// Performs full lowering from the IE Dialect to IERT Dialect.
//
// This pipeline performs full IR lowering from IE Dialect to IERT Dialect,
// including Function types, call graph and return operations.
//

void buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createBufferizeIEPass(Logger log = Logger::global());

//
// LowerIE2VPU
//

std::unique_ptr<mlir::Pass> createConvertIEToVPUM2IPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertLayers2VPUPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDynamicQuantToVPUNCEPass(Logger log = Logger::global());

//
// LowerVPU2VPUIP
//

//
// Performs full lowering from the VPU Dialect to VPUIP Dialect.
//
// This pipeline performs full IR lowering from VPU Dialect to VPUIP Dialect,
// including Function types, call graph and return operations.
//

std::unique_ptr<mlir::Pass> createOneShotBufferizeVPU2VPUIPPass();
std::unique_ptr<mlir::Pass> createInPlaceBufferizationAnalyzePass();
std::unique_ptr<mlir::Pass> createAdjustDynamicOpsBeforeBufferizationPass();
std::unique_ptr<mlir::Pass> createAddBuffersForNetResults(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertSWLayers2AffinePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAffine2LLVMPass(Logger log = Logger::global());

// ELF back-end lowerings
std::unique_ptr<mlir::Pass> createConvertVPUIP2VPUMI37XXPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVPUMI37XX2VPUASMPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVPUMI37XX2ELFPass(Logger log = Logger::global());

// NPUReg40XX ELF specific passes
std::unique_ptr<mlir::Pass> createConvertVPUIP2VPUMI40XXPass(Logger log = Logger::global(),
                                                             bool enableMemorySideCache = false);
std::unique_ptr<mlir::Pass> createConvertVPUMI40XX2VPUASMPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVPUIPDPU2NPUReg40XXPass(
        Logger log = Logger::global(), VPU::DPUDryRunMode dpuDryRunMode = VPU::DPUDryRunMode::NONE);
std::unique_ptr<mlir::Pass> createConvertVPUASM2NPUReg40XXPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVPUASM2NPUReg40XXRelocsPass(Logger log = Logger::global(),
                                                                     bool enableWLM = false);

//
// registerConversionPipelines
//

void registerConversionPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/conversion/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/conversion/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace vpux

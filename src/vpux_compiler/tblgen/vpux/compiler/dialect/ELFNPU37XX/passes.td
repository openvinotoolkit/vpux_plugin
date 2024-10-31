//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELFNPU37XX_PASSES
#define VPUX_COMPILER_DIALECT_ELFNPU37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// Cleanup ELF Sections
//

def RemoveEmptyELFSections : PassBase<"remove-empty-ELFNPU37XX-sections", "vpux::FunctionPass"> {
    let summary = "Cleans up empty sections in ELF";

    let description = [{
        The pass is part of the ELF pipeline.

        Current lowering logic may pre-create sections, without yet knowing if any op will eventually land in it. This
        causes some empty sections to be placed in the ELF. This pass should clean them up.
    }];

    let constructor = "vpux::ELFNPU37XX::createRemoveEmptyELFSectionsPass()";

    let dependentDialects = [
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
    ];
}

//
// UpdateELFSectionFlags
//

def UpdateELFSectionFlags : PassBase<"update-ELFNPU37XX-section-flags", "vpux::ModulePass"> {
    let summary = "Update ELF section flags";

    let description = [{
        The pass is part of the ELF pipeline.

        The pass will update the ELFNPU37XX section flags (secFlags attribute) for each relevant section based on which
        processors will access the given section at run time. This information is retrieved from the ops contained
        within the section via the ELFNPU37XX_BinaryOpInterface::getAccessingProcs interface method.

        A relevant section in the context of this pass is one that requires allocation and/or loading by the loader.
    }];

    let constructor = "vpux::ELFNPU37XX::createUpdateELFSectionFlagsPass()";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect",
        "vpux::VPUMI37XX::VPUMI37XXDialect",
    ];
}

#endif

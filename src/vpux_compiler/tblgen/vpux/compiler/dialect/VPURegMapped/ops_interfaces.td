//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def VPURegMapped_TaskOpInterface : OpInterface<"TaskOpInterface"> {
    let description = "Task Op interface";
    let cppNamespace = "vpux::VPURegMapped";
    let methods = [
        StaticInterfaceMethod<
            "Get task type",
            "vpux::VPURegMapped::TaskType", "getTaskType"
        >,
        InterfaceMethod<
            "Get index",
            "vpux::VPURegMapped::IndexType", "getIndexType", (ins), [{}], [{
                return $_op.getType();
            }]
        >,
        InterfaceMethod<
            "Get the single result of the op",
            "mlir::Value", "getResult", (ins), [{
                return $_op.getResult();
            }]
        >,
        InterfaceMethod<
            "Set task location",
            "void", "setTaskLocation", (ins "mlir::Value":$location), [{}], [{
                $_op.getTaskLocationMutable().assign(location);
            }]
        >,

        InterfaceMethod<
            "Get the location projection of this task",
            "mlir::Value", "getTaskLocation", (ins), [{}], [{
                return $_op.taskLocation();
            }]
        >,
        InterfaceMethod<
            "Get the previous task inside list",
            "vpux::VPURegMapped::TaskOpInterface", "getPreviousTask", (ins),
            [{
                auto prev = $_op.getPreviousTask();
                if(prev)
                    return mlir::cast<VPURegMapped::TaskOpInterface>(prev.getDefiningOp());
                else
                    return nullptr;
            }]
        >,
        InterfaceMethod<
            "Get next task",
            "::vpux::VPURegMapped::TaskOpInterface", "getNextTask", (ins), [{
                const auto users = $_op.getResult().getUsers();
                const auto isNextTask = [&](auto operation) {
                    auto task = ::mlir::dyn_cast_if_present<Interface>(operation);
                    return task && task.getPreviousTask() == $_self;
                };
                const auto nextTaskIter = ::llvm::find_if(users, isNextTask);
                return nextTaskIter == ::std::end(users) ? nullptr : ::mlir::cast<Interface>(*nextTaskIter);
            }]
        >,
        InterfaceMethod<
            "Get the previous task Value",
            "mlir::Value", "getPreviousValue", (ins),
            [{
                return $_op.getPreviousTask();
            }]
        >,

        InterfaceMethod<
            "Set the previous task of this Op",
            "void", "setPreviousTask", (ins "mlir::Value":$previousTask), [{}], [{
                $_op.getPreviousTaskMutable().assign(previousTask);
            }]
        >,
        InterfaceMethod<
            "Set the state of the op to specify that it's previous task hard links to this task",
            "void", "enableHardLink", (ins ), [{}], [{
                $_op.getOperation()->setAttr("HardLinkedAttrName", mlir::UnitAttr::get($_op.getOperation()->getContext()));
            }]
        >,
        InterfaceMethod<
            "Disable the hardLinked state of this task",
            "void", "disableHardLink", (ins), [{}], [{
                $_op.getOperation()->removeAttr("HardLinkedAttrName");
            }]
        >,
        InterfaceMethod<
            "Returns true if this op is hard-linked by it's previous task",
            "bool", "isHardLinked", (ins) , [{}], [{
                return $_op.getOperation()->hasAttr("HardLinkedAttrName");
            }]
        >,
        InterfaceMethod<
            "Returns true if the underlying Op supports hard linking capabilities",
            "bool", "supportsHardLink", (ins), [{}], [{
                return false;
            }]
        >
    ];
}

#endif

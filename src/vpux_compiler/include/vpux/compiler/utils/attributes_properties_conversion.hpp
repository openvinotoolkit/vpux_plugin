#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Location.h>

namespace vpux {

/**
 * This function provides a convenient way of populating the properties struct of a
 * specific operation with the values of a DictionaryAttr.
 */
template <class OpType>
typename OpType::Properties toProperties(mlir::Attribute attr) {
    typename OpType::Properties properties;

    // we need to provide an emitError function
    auto emitError = [&]() {
        return mlir::emitError(mlir::UnknownLoc::get(attr.getContext()), "Invalid properties: ");
    };

    VPUX_THROW_UNLESS(OpType::setPropertiesFromAttr(properties, attr, emitError).succeeded(),
                      "Can't set properties from attribute '{0}'", attr);

    return properties;
}

}  // namespace vpux

#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(VPUMI40XX)
add_vpux_ops_interface(VPUMI40XX dialect/VPUMI40XX/)
add_vpux_attr_interface(VPUMI40XX dialect/VPUMI40XX/)
add_vpux_attribute(VPUMI40XX ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_type(VPUMI40XX)
add_vpux_pass(VPUMI40XX VPUMI40XX dialect/VPUMI40XX/)

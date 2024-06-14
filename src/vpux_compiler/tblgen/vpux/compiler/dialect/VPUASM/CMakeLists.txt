#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(VPUASM)
add_vpux_ops_interface(VPUASM dialect/VPUASM/)
add_vpux_attribute(VPUASM ENABLE_VPUX_ATTR)
add_vpux_type(VPUASM)
add_vpux_pass(VPUASM VPUASM dialect/VPUASM/)

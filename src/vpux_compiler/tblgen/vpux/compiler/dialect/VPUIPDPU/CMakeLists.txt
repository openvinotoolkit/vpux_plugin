#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(VPUIPDPU)
add_vpux_attribute(VPUIPDPU ENABLE_VPUX_ENUMS)
add_vpux_pass(VPUIPDPU VPUIPDPU dialect/VPUIPDPU/)

# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(options)

if((ARM OR AARCH64) AND (NOT DEFINED MCM_COMPILER_EXPORT_FILE OR NOT EXISTS ${MCM_COMPILER_EXPORT_FILE}))
    ie_option(ENABLE_MCM_COMPILER "Enable MCM compiler build" OFF)
else()
    ie_option(ENABLE_MCM_COMPILER "Enable MCM compiler build" ON)
endif()

if(ARM OR AARCH64)
    ie_option(ENABLE_HDDL2 "Enable HDDL2 Plugin" OFF)
    ie_option(ENABLE_VPUAL "Enable VPUAL" ON)
else()
    ie_option(ENABLE_HDDL2 "Enable HDDL2 Plugin" ON)
    ie_option(ENABLE_VPUAL "Enable VPUAL" OFF)
endif()

ie_option(ENABLE_KMB_SAMPLES "Enable KMB samples" ON)

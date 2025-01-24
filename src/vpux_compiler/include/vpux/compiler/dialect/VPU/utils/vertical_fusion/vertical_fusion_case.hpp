//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/core/attributes/dim.hpp"

namespace vpux {
namespace VPU {

/*
  Candidate for VF
*/
class VFCase {
public:
    /*
     Destructor of VF case
    */
    virtual ~VFCase();

    /*
     Constructor of VF case
    */
    explicit VFCase(const VFConfig& config, Dim axis);

    /*
     Move constructor
    */
    VFCase(VFCase&& vfCase);

    /*
     Move assignment operator
    */
    VFCase& operator=(VFCase&& other);

    /*
     Copy constructor
    */
    VFCase(const VFCase& vfCase);

    /*
     Copy assignment operator
    */
    VFCase& operator=(const VFCase& other);

    /*
     Set number of tiles
    */
    void setTilingNumber(int64_t number);

    /*
     Set VF scheduling
    */
    void setScheduling(std::shared_ptr<IVFScheduling> vfScheduling);

    /*
     Set VF tiling storage
    */
    void setTilingStorage(std::unique_ptr<TilingOperationStorage> vfStorage);

    /*
     Get VF cost
    */
    StrategyCost getCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log);

    /*
     Check if VF case has been initialized with scheduling
    */
    bool isInitialized();

    /*
     Get VF config
    */
    VFConfig& getConfig();

    /*
     Generate VF tiling
    */
    mlir::ArrayAttr getTiling() const;

    /*
     Set Scheduling and tiling to VF
    */
    void approveScheduling();

    /*
     Get current tiling number
    */
    int64_t getTilingNumber() const;

private:
    /*
    Add CMX write spills
    */
    void addCMXWriteSpills(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction);

    /*
     Clear cached data
    */
    void clearCache();

    /*
     VF data
    */
    VFConfig _config;

    /*
     Axis for tiling
    */
    Dim _axis;

    /*
     Number of tiles
    */
    int64_t _tilingNumber = 1;

    /*
     VF Scheduling
    */
    std::shared_ptr<IVFScheduling> _vfScheduling;

    /*
     VF TilingOperationStorage
    */
    std::unique_ptr<TilingOperationStorage> _vfTilingStorage;

    /*
     Cached VF cost
    */
    std::optional<StrategyCost> _cachedCost;
};

}  // namespace VPU
}  // namespace vpux

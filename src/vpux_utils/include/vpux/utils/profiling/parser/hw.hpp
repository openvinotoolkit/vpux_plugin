//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Profiling HW interface

#pragma once

#include <cstdint>

struct ActShaveData_t {
    uint64_t begin;
    uint32_t duration;
    uint32_t stallCycles;
    uint32_t executedInstructions;
    uint32_t clockCycles;
    uint32_t branchTaken;
    uint32_t reserved32;
};

struct UpaData_t {
    uint64_t begin;
    uint64_t end;
    uint32_t stallCycles;
    uint32_t activeCycles;
};

// SW DPU profiling data payload
struct SwDpuData_t {
    uint64_t begin;
    uint64_t end;
};

// HWP DPU profiling data payload
struct HwpDpu27Mode0Data_t {
    uint64_t idu_wl_duration : 28;
    uint64_t idu_tstamp : 28;
    uint64_t sve_id : 5;
    uint64_t reserved3 : 3;
    uint64_t odu_wl_duration : 28;
    uint64_t odu_tstamp : 28;
    uint64_t reserved8 : 8;
};

struct HwpDma40Data_t {
    uint64_t desc_addr;
    uint64_t fetch_time;
    uint64_t ready_time;
    uint64_t start_time;
    uint64_t wdone_time;
    uint64_t finish_time;
    uint8_t la_id;
    uint8_t ch_id;
    uint16_t rsvd;
    uint16_t rstall_cnt;
    uint16_t wstall_cnt;
    uint32_t twbytes_cnt;
    uint32_t chcycle_cnt;
};

struct HwpDpuIduOduData_t {
    uint64_t idu_wl_duration : 32;
    uint64_t idu_wl_id : 16;
    uint64_t idu_dpu_id : 16;
    uint64_t idu_tstamp;
    uint64_t odu_wl_duration : 32;
    uint64_t odu_wl_id : 16;
    uint64_t odu_dpu_id : 16;
    uint64_t odu_tstamp;
};

struct DMA20Data_t {
    uint32_t startCycle;
    uint32_t endCycle;
};

struct DMA27Data_t {
    uint64_t startCycle;
    uint64_t endCycle;
};

struct WorkpointConfiguration_t {
    uint16_t pllMultiplier;
    uint16_t configId;
};

struct M2IData_t {
    uint64_t descAddr : 64;
    uint64_t fetchTime : 64;
    uint64_t readyTime : 64;
    uint64_t startTime : 64;
    uint64_t doneTime : 64;
    uint64_t finishTime : 64;
    uint64_t linkAgentID : 8;
    uint64_t parentID : 8;
    uint64_t RSVD : 16;
    uint64_t RStallCount : 16;
    uint64_t WStallCount : 16;
    uint64_t WRCycleCount : 32;
    uint64_t RDCycleCount : 32;
};

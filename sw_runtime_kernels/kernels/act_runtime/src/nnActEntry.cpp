/*
 * {% copyright %}
 */

#include <DrvRegUtils.h>
#include <DrvSvuL1Cache.h>
#include <cpuWhoAmI.h>
#include <nn_barrier.h>
#include <nn_fifo.h>
#include <nn_fifo_manager.h>
#include <nn_perf_manager.h>
#include <nn_counter.h>
#include <string.h>

#include <nnActRtUtils.h>
#include <nnActRtPerf.h>
#include <nnActRtDebug.h>

#include <sys/__moviconfig.h>

#define P_CFG_SETTING ~0b011110

#if defined(USE_SHAVE_NN_PRINT)
#include <stdio.h>
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

using namespace nn::act_runtime;
using namespace nn::util;
using namespace nn::common_runtime::fifo;

extern "C" void nnActEntry(void *config, void *scratch) {
//    uint32_t * tmp = (uint32_t *)0x2e014000;
//    uint32_t& debInd = *tmp;
//    debInd = 1;
    const SHVFifoConfig fifoCfg = unpackSHVConfig(reinterpret_cast<uint32_t>(config));
    const uint32_t wlFifoAddr = computeFifoRecieveAddress(fifoCfg.work.fifo, fifoCfg.work.index);
    const uint32_t ctFifoAddr = computeFifoRecieveAddress(fifoCfg.ctrx.fifo, fifoCfg.ctrx.index);
    const uint32_t prFifoAddr = computeFifoRecieveAddress(fifoCfg.perf.fifo, fifoCfg.perf.index);

    // TODO: double check that this is working now with latest tools
    const unsigned int shaveIndex = cpuWhoAmI() - PROCESS_ACT_SHAVE0;
    // const unsigned int shaveIndex = __builtin_shave_getcpuid();
    UNUSED(prFifoAddr);

    ActKernelInvocation *ki{nullptr};
    ActKernelRange *kr{nullptr};

    ActPerfReport pr;
    char packedPr[sizeof(ActPerfReport)];
    uint32_t perfPackedSize{0};
    uint32_t perfMetricMask{0};

    auto handleCtrl = [&](uint32_t fifo_val) {
        const ASCtrlMessage ctrl = unpackASCtrlMessage(reinterpret_cast<uint32_t>(fifo_val));

        switch (ctrl.message) {
            case SHVCtrlMessage::HWStatsEnable:
                break;
            case SHVCtrlMessage::PreemptHaltAndAck:
                break;
            case SHVCtrlMessage::EnablePerfStream:
                perfMetricMask = ctrl.payload;
                perfPackedSize = actPRPackedSize(perfMetricMask);
                configCounters(perfMetricMask);
                break;
            case SHVCtrlMessage::DisablePerfStream:
                perfMetricMask = 0;
                perfPackedSize = 0;
                break;

            default:
                break;
        }
    };

    auto handleKRChange = [&]() {
        // do something with the previous kRange
        // TODO: maybe do perf roll-up and push to perf FIFO?

        /*
         * TODO: we also need to prefetch the .text to L2
         * Note that a-shvs will share the same iL2 partition (per tile), so we may be spamming the prefetch here.
         *   Use a free HW mutex?
         */
        kr = ki->range_;

//        tmp[debInd++] = 222222;

        setShaveWindow(1, kr->textWindowBase_);

        // sDrvPfetchDl1LineL();
        // sDrvPfetchDl2(ki->data_);
    };

    auto writeDbgVal = [scratch](uint64_t val) {
        static volatile uint64_t *dbg = reinterpret_cast<uint64_t *>(scratch);
        *(dbg++) = val;
    };

    auto writeFRC = [writeDbgVal](void) {
#ifdef CONFIG_VALIDATION_APP_ENABLED
        writeDbgVal(nn::util::sampleFRC());
#else
        UNUSED(writeDbgVal);
#endif
    };

#ifdef ACT_RT_DEBUG
    /*
     * WARNING: This debug helper will almost certainly corrupt an inference and is _not_ safe to call by multiple
     * shaves from any tile > 0. Use at your own risk. It's only intended as a fast debugging tool to avoid MoviDebug's
     * complicated debuging features.
     */
    auto cmxDebugStride = [](uint32_t value) {
        // NOTE!: .data* sections are windowed to same window as .text for the ActRT.
        //        That means all shaves share the same .data!
        static uint32_t *debug{(uint32_t *)(0x2E000000 + 1024 * 1024 - 1024)};
        static uint32_t next{0};

        if (next < 1024) {
            *reinterpret_cast<uint32_t *>((reinterpret_cast<uint32_t>(debug) + next)) = value;
            next += 4;
        }
    };

    auto waitWL = [&]() {
        uint32_t ct;

        do {
            ki = reinterpret_cast<ActKernelInvocation *>(GET_REG_WORD_VAL(wlFifoAddr));
            ct = GET_REG_WORD_VAL(ctFifoAddr);

            if (ct != 0) {
                handleCtrl(ct);
            }
        } while (ki == 0);
    };
#else
    auto waitWL = [&]() {
        if (fifoWaitGpioWithCtrl(fifoCfg.work.fifo, fifoCfg.ctrx.fifo)) {
            ki = reinterpret_cast<ActKernelInvocation *>(GET_REG_WORD_VAL(wlFifoAddr));
        } else {
            ki = nullptr;
            handleCtrl(GET_REG_WORD_VAL(ctFifoAddr));
        }
        writeFRC();
    };
#endif

    auto execWL = [&]() {
        const auto &barriers = ki->barriers_;
        const auto &barriers_gpio = ki->barriersGpio_;

//        tmp[debInd++] = 333330;
        setShaveWindow(2, ki->dataWindowBase_);

//        tmp[debInd++] = 333331;
//        tmp[debInd++] = HglBarrierGetProdConsCounts(0);
//        tmp[debInd++] = HglBarrierGetProdConsCounts(1);
        waitBarrier(barriers, barriers_gpio, shaveIndex);
//        HglBarrierWait(barriers.wait_mask_);
//        tmp[debInd++] = 333332;
//        tmp[debInd++] = HglBarrierGetProdConsCounts(0);
//        tmp[debInd++] = HglBarrierGetProdConsCounts(1);
        HglBarrierConsume(barriers.wait_mask_);
//        tmp[debInd++] = 333333;

        if (perfMetricMask) {
            resetCounters(pr);

//            tmp[debInd++] = 333334;
            (kr->kernelEntry_)(ki->kernelArgs_);
//            tmp[debInd++] = 333335;

            recordCounters(pr);
            packActPerfReport(perfMetricMask, pr, reinterpret_cast<void *>(packedPr));

            if (ki->perfPacketOut_) {
                memcpy_s(ki->perfPacketOut_, sizeof(ActPerfReport), reinterpret_cast<const void *>(packedPr),
                         perfPackedSize);
            } else {
                // TODO: stream it out
            }
        } else {
            writeFRC();
//            tmp[debInd++] = 333336;
            (kr->kernelEntry_)(ki->kernelArgs_);
//            tmp[debInd++] = 333337;
        }
//#ifdef JTAG_LOW_LEVEL
//        if (kr->type_ == ActWLType::WL_KERNEL_LRT_SYNC && kr->LRTSynch_ == ActKernelRange::LRT_WAIT) {
//            kr->LRTSynch_ = ActKernelRange::KERNEL_DONE;
//        }
//#endif
//        tmp[debInd++] = 333338;
        HglBarrierProduce(barriers.post_mask_);
//        tmp[debInd++] = 333339;
    };

    setFPRound(P_CFG_SETTING);

    do {
        waitWL();

        if (ki) {
            if (ki->range_ != kr)
                handleKRChange();

            switch (kr->type_) {
//#ifdef JTAG_LOW_LEVEL
//                case ActWLType::WL_KERNEL_LRT_SYNC:
//#endif
                case ActWLType::WL_KERNEL: {
                    execWL();
                    break;
                }
#ifdef NN_ENABLE_CONTEXT_DEBUGGING
                case ActWLType::WL_DEBUG: {
                    execDebug(kr, shaveIndex, fifoCfg);
                    break;
                }
#endif
                case ActWLType::WL_UNKNOWN: {
                    break;
                }
                default:
                    break;
            }
        }
    } while (true);
}

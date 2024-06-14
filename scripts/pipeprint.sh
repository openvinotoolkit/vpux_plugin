#!/bin/bash
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

cat << EOF_PIPE_PRINT_SRC | g++ -xc++ -o pipeprint - && chmod +x pipeprint && ./pipeprint $1
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

#include <iostream>
#include <sstream>

typedef uint64_t u64;
typedef uint32_t u32;
typedef int32_t i32;
typedef uint8_t u8;

#define SET_RED_COLOR         printf("\033[0;31m")
#define SET_YELLOW_COLOR      printf("\033[0;33m")
#define SET_WHITE_COLOR       printf("\033[0;37m")
#define SET_BLUE_COLOR        printf("\033[0;34m")
#define RESET_COLOR           printf("\033[0m")

 typedef struct  {
     volatile u32  canaryStart;
     volatile u32  in;
     volatile u32  out;
     volatile i32  queueSize;
     volatile u32  queuePtr;
     volatile u32  canaryEnd;
} tyMvConsoleQueue;

#define PIPEPRINT_CANARY_START (0x22334455)
#define PIPEPRINT_CANARY_END (0xBBCCDDEE)

#define VPU_CACHE_LINE_SIZE 64L
#define ALIGN_DOWN(a, size) ((a)&(~(size-1)))

#define PAGE_SIZE (4096)

#define PAGE_ALIGN_UP(a) (((a + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE)
#define PAGE_ALIGN_DOWN(phy_addr) ALIGN_DOWN(phy_addr, PAGE_SIZE)

#define isPageAligned(a) (!((a) & PAGE_SIZE - 1))

// physical pointer in another process, points to certain data size
template <class T>
class  PhysPtr {
protected:
    u8  * mapped_ptr = nullptr;
    size_t mapsize = 0;
    size_t page_offset = 0;
    int fd;
public:
    /**
     * @ptr to physically stable adress - in any process you need to map
     * @sz - size in bytes of the object
     */
    PhysPtr(u64 phy_addr, size_t sz = sizeof(T)) {
        fd = open("/dev/mem", O_RDONLY | O_SYNC);
        if (fd < 0) {
            throw std::runtime_error("/dev/mem Open failed");
        }
        // phy_addr - might be not page aligned - so map from page boundary
        auto phy_addr_page_aligned = PAGE_ALIGN_DOWN(phy_addr);
        page_offset = phy_addr - phy_addr_page_aligned;

        if (!isPageAligned(phy_addr)) {
            std::cout  << "phy_addr 0x"<<std::hex << phy_addr << " is not page-aligned!" <<
                " memory mapping to lest page aligned addr of: 0x" << std::hex << phy_addr_page_aligned << "\n";
        }

        mapsize = PAGE_ALIGN_UP(page_offset + sz);
        mapped_ptr = reinterpret_cast<uint8_t*>(mmap(NULL, mapsize, PROT_READ, MAP_SHARED, fd, phy_addr_page_aligned));
        if(mapped_ptr == MAP_FAILED) {
            std::stringstream err;
            err << "failed to map header at : mmap(offset=0x" << std::hex << phy_addr_page_aligned << " failed";
            close(fd);
            throw std::runtime_error(err.str());
        }
    }
    T* operator *() const {
        return  reinterpret_cast<T*>(mapped_ptr + page_offset);
    }
    T* operator ->() const {
        return  reinterpret_cast<T*>(mapped_ptr + page_offset);
    }
    virtual ~PhysPtr() {
        if (mapped_ptr != nullptr) {
            munmap(mapped_ptr, mapsize);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

void (*oldHandler)(int) = nullptr;

void intHandler(int sig) {
    RESET_COLOR;
    printf("\n");
    if (oldHandler != nullptr) {
      oldHandler(sig);
    } else {
      exit(0);
    }
}

int main(int argc, const char * argv[]) {

    u64	phy_addr = 0x94400040;
    oldHandler = signal(SIGINT, intHandler);

    if (argc > 1)
        phy_addr = strtoll(argv[1], NULL, 0);

    try {
        PhysPtr<tyMvConsoleQueue> header(phy_addr);

        // if buffer pointer outside mapped page
        SET_YELLOW_COLOR;
        printf("\nThe pipeprint header content:\n");
        printf("\theader->canaryStart=0x%08X\n", header->canaryStart);
        printf("\t        header->in =0x%08X\n", header->in);
        printf("\t        header->out=0x%08X\n", header->out);
        printf("\t  header->queueSize=0x%08X\n", header->queueSize);
        printf("\t  header->queuePtr =0x%08X\n", header->queuePtr);
        printf("\t  header->canaryEnd=0x%08X\n", header->canaryEnd);

        if (PIPEPRINT_CANARY_START != header->canaryStart) {
            throw std::runtime_error("Invalid start Canary at given adress: expected to be "
                                     + std::to_string(PIPEPRINT_CANARY_START));
        }
        if (PIPEPRINT_CANARY_END != header->canaryEnd) {
            throw std::runtime_error("Invalid end Canary at given adress: expected to be "
                                     + std::to_string(PIPEPRINT_CANARY_END));
        }

        PhysPtr<u8> queBuffer(static_cast<u64>(header->queuePtr), header->queueSize);
        PhysPtr<volatile u32> inputCounterPtr(static_cast<u64>(header->in));

        // 1ms sleep when no logs are presented.
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 1 * 1000000;
        u32 no_data_ticks = 0;

        auto in = 0;
        auto queueSize = header->queueSize;
        auto queuePtr = header->queuePtr;

        while(1) {
            u32 next_in = *(*inputCounterPtr);
            auto cnt = (next_in - in) % queueSize;

            if (cnt > 0) {
                //printf("queueSize=%d, next_in=%d, in=%d, cnt=%d\n",queueSize, next_in, in, cnt);fflush(stdout);

                // only 64bit aligned part is flushed from cache to RAM in time
                // the rest part will be flushed later by subsequent logs or forcely with timeout
                auto cnt_caligned = (ALIGN_DOWN((queuePtr + next_in), VPU_CACHE_LINE_SIZE) -
                                                ((queuePtr + in))) % queueSize;

                if (cnt_caligned)
                    cnt = cnt_caligned;
                else if (no_data_ticks < 10000)
                    cnt = 0;

                if (cnt) {
                    write(1, *queBuffer + in, cnt);
                    in = (in + cnt) % queueSize;
                    no_data_ticks = 0;
                    continue;
                }
            }
            nanosleep(&ts, &ts);
            no_data_ticks ++;
        }

    } catch(const std::exception &ex) {
        RESET_COLOR;
        std::cout << "\nERROR: " << ex.what();
        return 1;
    }

    return 0;
}

EOF_PIPE_PRINT_SRC

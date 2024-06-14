//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <stdio.h>
#include <stdlib.h>

#include "vpux_driver_compiler.h"

int readFile(const char* fileName, char** buffer, size_t* size) {
    FILE* file = fopen(fileName, "rb");
    if (!file) {
        perror("Can't open blob file");
        return EXIT_FAILURE;
    }

    fseek(file, 0L, SEEK_END);
    long fileSize = ftell(file);
    if (fileSize < 0) {
        printf("Ftell method returns failure.");
        fclose(file);
        return VCL_RESULT_ERROR_IO;
    }
    uint64_t unsignedFileSize = (uint64_t)fileSize;
    fseek(file, 0L, SEEK_SET);

    char* binaryBuffer = (char*)malloc(unsignedFileSize);
    if (!binaryBuffer) {
        fprintf(stderr, "Can't allocate %zu bytes to read %s.", unsignedFileSize, fileName);
        fclose(file);
        return EXIT_FAILURE;
    }

    // Use fgetc to read from the file into the buffer
    int ch;
    size_t bytesRead = 0;

    while ((ch = fgetc(file)) != EOF && bytesRead < unsignedFileSize - 1) {
        if (ch >= 32 && ch <= 126) {
            binaryBuffer[bytesRead++] = (char)ch;
        } else {
            free(binaryBuffer);
            fprintf(stderr, "Can't character is a printable ASCII character.");
            fclose(file);
            return EXIT_FAILURE;
        }
    }

    // Null-terminate the buffer
    buffer[bytesRead] = '\0';

    if (bytesRead <= 0) {
        free(binaryBuffer);
        fprintf(stderr, "Binary buffer is empty");
        fclose(file);
        return EXIT_FAILURE;
    }

    fclose(file);

    *buffer = binaryBuffer;
    *size = unsignedFileSize;

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage:\n"
               "\tprofilingTest network.blob profiling_output.bin\n"
               "where\n"
               "\tnetwork.blob - blob with profiling enabled ('PERF_COUNT YES' parameter in the compiler)"
               "\tprofiling_output.bin - raw profiling output acquired from InferenceManagerDemo according to "
               "guides/how-to-use-profiling.md\n");
        return EXIT_FAILURE;
    }

    const char* blobFileName = argv[1];
    const char* profFileName = argv[2];

    char* blobBuffer = NULL;
    size_t blobSize = 0;
    int result = readFile(blobFileName, &blobBuffer, &blobSize);
    if (result != EXIT_SUCCESS) {
        return result;
    }

    char* profBuffer = NULL;
    size_t profSize = 0;
    result = readFile(profFileName, &profBuffer, &profSize);
    if (result != EXIT_SUCCESS) {
        free(blobBuffer);
        return result;
    }

    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_profiling_input_t profilingApiInput = {.blobData = blobBuffer,
                                               .blobSize = blobSize,
                                               .profData = profBuffer,
                                               .profSize = profSize};
    vcl_profiling_handle_t profHandle = NULL;
    ret = vclProfilingCreate(&profilingApiInput, &profHandle, NULL);
    if (ret != VCL_RESULT_SUCCESS) {
        result = EXIT_FAILURE;
        goto exit;
    }

    vcl_profiling_properties_t profProperties;
    ret = vclProfilingGetProperties(profHandle, &profProperties);
    if (ret != VCL_RESULT_SUCCESS) {
        result = EXIT_FAILURE;
        goto exit_destroy;
    }
    printf("Using profiling version %hu.%hu\n", profProperties.version.major, profProperties.version.minor);

    vcl_profiling_output_t profOutput;
    profOutput.data = NULL;
    ret = vclGetDecodedProfilingBuffer(profHandle, VCL_PROFILING_LAYER_LEVEL, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
        result = EXIT_FAILURE;
        goto exit_destroy;
    }

    profOutput.data = NULL;
    ret = vclGetDecodedProfilingBuffer(profHandle, VCL_PROFILING_TASK_LEVEL, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
        result = EXIT_FAILURE;
        goto exit_destroy;
    }

    profOutput.data = NULL;
    ret = vclGetDecodedProfilingBuffer(profHandle, VCL_PROFILING_RAW, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
        result = EXIT_FAILURE;
        goto exit_destroy;
    }

    printf("Test passed. Profiling API works! Great success!\n");

exit_destroy:
    vclProfilingDestroy(profHandle);

exit:
    free(blobBuffer);
    free(profBuffer);

    return result;
}

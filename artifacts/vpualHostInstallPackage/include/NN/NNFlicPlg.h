///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      GraphManagerPlg.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for GraphManagerPlg Host FLIC plugin stub using VPUAL.
///
#ifndef __NNFLIC_PLG_H__
#define __NNFLIC_PLG_H__

#include "Flic.h"
#include "Message.h"
#include "NN_Types.h"
#include <vector>

#include "Pool.h"

typedef enum {
    SUCCESS = MVNCI_SUCCESS,
    INPUT_FORMAT = MVNCI_WRONG_INPUT_FORMAT,
    NETWORK_ELEMENT = MVNCI_UNSUPPORTED_NETWORK_ELEMENT,
    INVALID_HANDLE = MVNCI_INVALID_HANDLE,
    OUT_OF_RESOURCES = MVNCI_OUT_OF_RESOURCES,
    NOT_IMPLEMENTED = MVNCI_NOT_IMPLEMENTED,
    INTERNAL_ERROR = MVNCI_INTERNAL_ERROR,
} NNPlgState;

class NNFlicPlg : public PluginStub{

  public:

  SReceiver<TensorMsgPtr> tensorInput;
  MReceiver<TensorMsgPtr> resultInput;
  MSender<TensorMsgPtr> output;

  SReceiver <InferenceMsgPtr> inferenceInput;
  MReceiver <InferenceMsgPtr> inferenceResult;
  MSender <InferenceMsgPtr> inferenceOutput;

  public:

    /** Constructor. */
    NNFlicPlg(uint32_t device_id = 0) : PluginStub("NNFlicDecoder", device_id), output{device_id}, inferenceOutput{device_id} {}

    void Create(BlobHandle_t * Blhdl);
    // void Delete();
    // void Stop();

    void SetNumberOfThreads(int32_t threadNum);
    void SetNumberOfShaves(int32_t shaves);
    void SetScratchBuffer(const std::vector<void *> &physAddrs) const;
    void SetPrefetchBuffer(void* physAddrs, unsigned int size, bool skipInit = false) const;

    NNPlgState GetLatestState();

    unsigned int GetNumberOfInputs() const;
    unsigned int GetNumberOfOutputs() const;
    unsigned int GetMaxNumberOfThreads() const;
    unsigned int GetMaxNumberOfShaves() const;
    unsigned int GetMemoryRequirements() const;

    unsigned int GetNumberOfStages() const;

    uint32_t *GetBlobVersion() const;
    flicTensorDescriptor_t GetInputTensorDescriptor(unsigned int index) const;
    flicTensorDescriptor_t GetOutputTensorDescriptor(unsigned int index) const;

};
#endif // __NNFLIC_PLG_H__

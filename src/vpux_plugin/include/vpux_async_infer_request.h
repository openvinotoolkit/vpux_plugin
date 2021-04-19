//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "vpux_infer_request.h"

namespace vpux {

class AsyncInferRequest final : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AsyncInferRequest>;

    explicit AsyncInferRequest(const InferRequest::Ptr& inferRequest,
                               const InferenceEngine::ITaskExecutor::Ptr& requestExecutor,
                               const InferenceEngine::ITaskExecutor::Ptr& getResultExecutor,
                               const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~AsyncInferRequest();

private:
    InferRequest::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _getResultExecutor;
};

}  // namespace vpux

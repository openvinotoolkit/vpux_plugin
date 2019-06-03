//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <ie_blob.h>

namespace vpu {

namespace ie = InferenceEngine;

ie::Blob::Ptr getBlobFP16(const ie::Blob::Ptr& in);

ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in);
ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in, ie::Layout outLayout);
void copyBlob(const ie::Blob::Ptr& in, const ie::Blob::Ptr& out);

}  // namespace vpu

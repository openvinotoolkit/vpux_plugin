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

#include <blob_factory.hpp>
#include <blob_transform.hpp>
#include <cpp/ie_cnn_network.h>
#include <cpp_interfaces/exception2status.hpp>
#include <details/ie_exception.hpp>
#include <ie_blob.h>
#include <ie_common.h>
#include <ie_core.hpp>
#include <ie_data.h>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_layouts.h>
#include <ie_metric_helpers.hpp>
#include <ie_parallel.hpp>
#include <ie_parameter.hpp>
#include <ie_plugin_config.hpp>
#include <ie_precision.hpp>
#include <precision_utils.h>

#include <ngraph/function.hpp>
#include <ngraph/node_output.hpp>
#include <ngraph/ops.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

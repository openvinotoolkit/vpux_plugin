//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/layers_reader.hpp"
#include "utils/error.hpp"

OpenVINOLayersReader& getOVReader() {
    static OpenVINOLayersReader reader;
    return reader;
}

InOutLayers LayersReader::readLayers(const InferenceParams& params) {
    if (std::holds_alternative<OpenVINOParams>(params)) {
        const auto& ov = std::get<OpenVINOParams>(params);
        return getOVReader().readLayers(ov);
    }
    ASSERT(std::holds_alternative<ONNXRTParams>(params));
    const auto& ort = std::get<ONNXRTParams>(params);
    // NB: Using OpenVINO to read the i/o layers information for *.onnx model
    OpenVINOParams ov;
    ov.path = OpenVINOParams::ModelPath{ort.model_path, ""};
    return getOVReader().readLayers(ov);
}

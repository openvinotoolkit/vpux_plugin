// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef _WIN32
#ifdef ENABLE_DX12

#include <initguid.h>  // it has to be placed before dxcore

#ifndef NOMINMAX
#define NOMINMAX
#define NOMINMAX_DEFINED_CTX_UT
#endif

#include <combaseapi.h>
#include <d3d12.h>
#include <d3dcommon.h>
#include <d3dx12_core.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <wrl.h>
#include <wrl/client.h>

#ifdef NOMINMAX_DEFINED_CTX_UT
#undef NOMINMAX
#undef NOMINMAX_DEFINED_CTX_UT
#endif

#include <vector>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <common_test_utils/test_constants.hpp>
#include <openvino/core/any.hpp>
#include <openvino/core/type/element_iterator.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <overload/overload_test_utils_npu.hpp>

#include "base/ov_behavior_test_utils.hpp"
#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"

namespace ov {
namespace test {
namespace behavior {
class NPUInferRequestDynamicRemoteTensorsTests_NPU3720 : public OVInferRequestDynamicTests {
protected:
    void checkOutputFP16(const ov::Tensor& in, const ov::Tensor& actual) {
        auto net = ie->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
        ov::InferRequest req;
        req = net.create_infer_request();
        auto tensor = req.get_tensor(function->inputs().back().get_any_name());
        tensor.set_shape(in.get_shape());
        for (size_t i = 0; i < in.get_size(); i++) {
            tensor.data<ov::element_type_traits<ov::element::f32>::value_type>()[i] =
                    in.data<ov::element_type_traits<ov::element::f32>::value_type>()[i];
        }
        req.infer();
        OVInferRequestDynamicTests::checkOutput(actual, req.get_output_tensor(0));
    }

    Microsoft::WRL::ComPtr<IDXCoreAdapter> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device9> device;
    Microsoft::WRL::ComPtr<ID3D12Heap> heap = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> placed_resources = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> comitted_resource;

    HANDLE shared_mem = nullptr;
    static const std::size_t alignment = 4096;

    void createAdapter() {
        Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> factory;

        auto res = DXCoreCreateAdapterFactory(IID_PPV_ARGS(factory.GetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "DXCoreCreateAdapterFactory failed.";

        const auto regex = std::regex("^\\bIntel\\b.*?\\bGraphics\\b.*?");
        const GUID guids[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE};

        // create the adapter list
        Microsoft::WRL::ComPtr<IDXCoreAdapterList> adapter_list;
        res = factory->CreateAdapterList(ARRAYSIZE(guids), guids, IID_PPV_ARGS(adapter_list.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateAdapterList failed.";

        // find our adapter
        for (uint32_t iter = 0; iter < adapter_list->GetAdapterCount(); iter++) {
            Microsoft::WRL::ComPtr<IDXCoreAdapter> local_adapter;
            res = adapter_list->GetAdapter(iter, IID_PPV_ARGS(local_adapter.ReleaseAndGetAddressOf()));
            ASSERT_FALSE(res != S_OK) << "GetAdapter failed.";

            size_t driver_desc_size = 0;
            res = local_adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &driver_desc_size);
            ASSERT_FALSE(res != S_OK) << "GetPropertySize failed.";

            std::vector<char> driver_desc(driver_desc_size);
            res = local_adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, driver_desc_size,
                                             &driver_desc[0]);
            ASSERT_FALSE(res != S_OK) << "GetProperty failed.";

            if (std::regex_match(std::string(driver_desc.data()), regex)) {
                adapter = local_adapter;
                break;
            }
        }

        auto check_adapter = adapter->IsValid();
        if (!check_adapter) {
            OPENVINO_THROW("GPU adapter is not valid");
        }
    }

    void creeateDevice() {
        auto res = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_1_0_CORE,
                                     IID_PPV_ARGS(device.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "D3D12CreateDevice failed.";
    }

    void createHeap(const size_t byte_size) {
        const size_t size = (byte_size + (static_cast<size_t>(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) - 1)) &
                            ~(static_cast<size_t>(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) - 1);

        D3D12_HEAP_DESC desc_heap{};
        desc_heap.SizeInBytes = size;
        desc_heap.Properties.Type = D3D12_HEAP_TYPE_CUSTOM;
        desc_heap.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
        desc_heap.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        desc_heap.Properties.CreationNodeMask = 1;
        desc_heap.Properties.VisibleNodeMask = 1;
        desc_heap.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc_heap.Flags = D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER | D3D12_HEAP_FLAG_SHARED;
        auto res = device->CreateHeap(&desc_heap, IID_PPV_ARGS(heap.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateHeap failed.";

        res = device->CreateSharedHandle(heap.Get(), nullptr, GENERIC_ALL, nullptr, &shared_mem);
        ASSERT_FALSE(res != S_OK) << "CreateSharedHandle failed.";
    }

    void createPlacedResources(const size_t byte_size) {
        D3D12_RESOURCE_DESC desc_resource{};
        desc_resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc_resource.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc_resource.Width = byte_size;
        desc_resource.Height = 1;
        desc_resource.DepthOrArraySize = 1;
        desc_resource.MipLevels = 1;
        desc_resource.Format = DXGI_FORMAT_UNKNOWN;
        desc_resource.SampleDesc.Count = 1;
        desc_resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc_resource.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        auto res = device->CreatePlacedResource(heap.Get(), 0, &desc_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                nullptr, IID_PPV_ARGS(placed_resources.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreatePlacedResource failed.";
    }

    void createComittedResources(const size_t byte_size) {
        auto res = device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                                                   D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(byte_size),
                                                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                   IID_PPV_ARGS(comitted_resource.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateCommittedResource failed.";
    }

    void createResources(const size_t byte_size) {
        createHeap(byte_size);
        createPlacedResources(byte_size);
        createComittedResources(byte_size);
    }

    void copyResources(const size_t byte_size) {
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> command_list;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        uint32_t fence_value = 0;

        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.NodeMask = 0;
        auto res = device->CreateCommandQueue(&desc, IID_PPV_ARGS(command_queue.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateCommandQueue failed.";

        res = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateFence failed.";

        res = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateCommandAllocator failed.";

        res = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, command_allocator.Get(), nullptr,
                                        IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateCommandList failed.";

        command_list->CopyBufferRegion(placed_resources.Get(), 0, comitted_resource.Get(), 0, byte_size);
        res = command_list->Close();
        ASSERT_FALSE(res != S_OK) << "Close command list failed.";

        ID3D12CommandList* command_lists[] = {command_list.Get()};
        command_queue->ExecuteCommandLists(ARRAYSIZE(command_lists), command_lists);
        res = command_queue->Signal(fence.Get(), ++fence_value);
        ASSERT_FALSE(res != S_OK) << "Signal command queue failed.";

        volatile auto event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        res = fence->SetEventOnCompletion(fence_value, event);
        ASSERT_FALSE(res != S_OK) << "SetEventOnCompletion failed.";
        WaitForSingleObject(event, INFINITE);
    }
};

TEST_P(NPUInferRequestDynamicRemoteTensorsTests_NPU3720, InferDynamicNetworkRemoteTensorWindows) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::vector<ov::Shape> vector_shapes{inOutShapes[0].first, inOutShapes[0].first};
    const std::string inputName = "Parameter_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(1, inOutShapes[1].first[0]), ov::Dimension(1, inOutShapes[1].first[1]),
                         ov::Dimension(1, inOutShapes[1].first[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto context = ie->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto inference_request = ie->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    const std::string outputName = "Relu_2";
    for (auto& shape : vector_shapes) {
        ov::Tensor in_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);

        const auto byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(in_tensor.get_shape()));

        createAdapter();
        creeateDevice();
        createResources(byte_size);
        void* mem;
        comitted_resource.Get()->Map(0, nullptr, &mem);
        memcpy(mem, in_tensor.data(), byte_size);
        comitted_resource.Get()->Unmap(0, nullptr);
        copyResources(byte_size);

        auto remote_tensor = context.create_tensor(ov::element::f32, in_tensor.get_shape(), shared_mem);

        OV_ASSERT_NO_THROW(req = inference_request.create_infer_request());
        OV_ASSERT_NO_THROW(req.set_tensor(inputName, remote_tensor));
        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(in_tensor, req.get_tensor(outputName)));
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

#endif
#endif

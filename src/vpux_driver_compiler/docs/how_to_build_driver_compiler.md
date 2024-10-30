# How to build Driver Compiler

## Dependencies

Please refer
* [System requirements](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-linux.html)
* [OpenVINO software requirements](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md)
* [How to build](../../../guides/how-to-build.md) in [NPU-Plugin Project]


## Using Cmake command

Driver Compiler is built with OpenVINO static runtime. To build the library and related tests (npu_driver_compiler, npu_elf, compilerTest, profilingTest, loaderTest) using following commands :

1. Clone repos.

    Clone [OpenVINO Project] repo and [NPU-Plugin Project] repo to special location. **Or** just unpack OPENVINO and NPU-Plugin source code to special location.

    <details>
    <summary> clone command</summary>

    ```sh
        # set the proxy, if required.
        # export/set  http_proxy=xxxx
        # export/set  https_proxy=xxxx

        cd /path/to/workspace
        git clone https://github.com/openvinotoolkit/openvino.git 
        cd openvino
        git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
        git submodule update --init --recursive

        cd /path/to/workspace
        git clone https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin
        cd applications.ai.vpu-accelerators.vpux-plugin
        git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
        git submodule update --init --recursive
    ```
    <details>
    

2. Set `NPU_PLUGIN_HOME` environment variable.

    On Ubuntu:
    ```sh
        export NPU_PLUGIN_HOME=/path/to/applications.ai.vpu-accelerators.vpux-plugin
    ```

    On `x64 Native Tools Command Prompt for VS XXXX` on Windows:
    ```bat
        set NPU_PLUGIN_HOME=/path/to/applications.ai.vpu-accelerators.vpux-plugin
    ```

3. Create build folder and run build commands.

    <details>
    <summary> on Ubuntu</summary>

    ```sh
        cd /path/to/openvino/
        mkdir build-x86_64
        cd build-x86_64
        cmake \
            -D CMAKE_BUILD_TYPE=Release \
            -D BUILD_SHARED_LIBS=OFF \
            -D OPENVINO_EXTRA_MODULES=$NPU_PLUGIN_HOME \
            -D ENABLE_TESTS=OFF \
            -D ENABLE_BLOB_DUMP=OFF \
            -D ENABLE_HETERO=OFF \
            -D ENABLE_MULTI=OFF \
            -D ENABLE_AUTO=OFF \
            -D ENABLE_AUTO_BATCH=OFF \
            -D ENABLE_TEMPLATE=OFF \
            -D ENABLE_OV_ONNX_FRONTEND=OFF \
            -D ENABLE_OV_PYTORCH_FRONTEND=OFF \
            -D ENABLE_OV_PADDLE_FRONTEND=OFF \
            -D ENABLE_OV_TF_FRONTEND=OFF \
            -D ENABLE_OV_TF_LITE_FRONTEND=OFF \
            -D ENABLE_INTEL_CPU=OFF \
            -D ENABLE_INTEL_GPU=OFF \
            -D ENABLE_PROXY=OFF \
            -D ENABLE_OV_IR_FRONTEND=ON \
            -D THREADING=TBB \
            -D ENABLE_TBBBIND_2_5=OFF \
            -D ENABLE_SYSTEM_TBB=OFF \
            -D ENABLE_TBB_RELEASE_ONLY=OFF \
            -D ENABLE_JS=OFF \
            -D BUILD_COMPILER_FOR_DRIVER=ON \
            -D ENABLE_NPU_PROTOPIPE=OFF \
            ..

        cmake --build . --config Release --target gtest_main gtest ov_dev_targets compilerTest profilingTest vpuxCompilerL0Test loaderTest -j8
        # or just use
        cmake --build . --config Release --target compilerTest profilingTest vpuxCompilerL0Test loaderTest -j8
    ```
    > Notice: If you encounter the following error during building `c++: internal compiler error: Killed (program cc1plus)`. You should consider decreasing the number of threads during compiling. For example, you could consider using `-j4` decrease the  thread number to 4 or a smaller value. 
    </details>

    <details>
    <summary> x64 Native Tools Command Prompt for VS XXXX on Winodws</summary>
    
    ```bat
        cd \path\to\openvino\
        md build-x86_64
        cd build-x86_64
        cmake ^
            -D CMAKE_BUILD_TYPE=Release ^
            -D BUILD_SHARED_LIBS=OFF ^
            -D OPENVINO_EXTRA_MODULES=%NPU_PLUGIN_HOME% ^
            -D ENABLE_TESTS=OFF ^
            -D ENABLE_BLOB_DUMP=OFF ^
            -D ENABLE_HETERO=OFF ^
            -D ENABLE_MULTI=OFF ^
            -D ENABLE_AUTO=OFF ^
            -D ENABLE_AUTO_BATCH=OFF ^
            -D ENABLE_TEMPLATE=OFF ^
            -D ENABLE_OV_ONNX_FRONTEND=OFF ^
            -D ENABLE_OV_PYTORCH_FRONTEND=OFF ^
            -D ENABLE_OV_PADDLE_FRONTEND=OFF ^
            -D ENABLE_OV_TF_FRONTEND=OFF ^
            -D ENABLE_OV_TF_LITE_FRONTEND=OFF ^
            -D ENABLE_INTEL_CPU=OFF ^
            -D ENABLE_INTEL_GPU=OFF ^
            -D ENABLE_PROXY=OFF ^
            -D ENABLE_OV_IR_FRONTEND=ON ^
            -D THREADING=TBB ^
            -D ENABLE_TBBBIND_2_5=OFF ^
            -D ENABLE_SYSTEM_TBB=OFF ^
            -D ENABLE_TBB_RELEASE_ONLY=OFF ^
            -D ENABLE_JS=OFF ^
            -D BUILD_COMPILER_FOR_DRIVER=ON ^
            -D ENABLE_NPU_PROTOPIPE=OFF ^
            ..

        cmake --build . --config Release --target gtest_main gtest ov_dev_targets compilerTest profilingTest vpuxCompilerL0Test loaderTest -j8
        # or just use
        cmake --build . --config Release --target compilerTest profilingTest vpuxCompilerL0Test loaderTest -j8
    ```
    </details>
    
### See also

Follow the blow guide to build the Driver Compiler library and test targets with Ninja:
 * `Using ninja` section of [how-to-build.md](../../../guides/how-to-build.md) of [NPU-Plugin Project].

Driver compiler build is a static build, to get a static build of [NPU-Plugin Project] repo, please see
 * [how to build static](../../../guides/how-to-build-static.md).

To learn more about the Driver Compiler package, please see
 * [introduction of Driver Compiler package](https://github.com/intel-innersource/applications.ai.vpu-accelerators.flex-cid-tools/blob/develop/docs/introductio-of-driver_compiler_package.md).
 * [how to Release Driver Compiler package.md](https://github.com/intel-innersource/applications.ai.vpu-accelerators.flex-cid-tools/blob/develop/docs/how-to-release-package.md).


## Using CMakePresets

### Using CMakePresets as an extra module of OpenVINO
Here provides a default pre-configured CMake presets for users named: "NpuCidRelease". The setting is to build [NPU-Plugin Project] as an extra module of [OpenVINO Project]. In this case, `NPU_PLUGIN_HOME` environment variable must be set.

1. Clone repos
    <details>
    <summary> clone command</summary>

    ```sh
        # set the proxy, if required.
        # export/set  http_proxy=xxxx
        # export/set  https_proxy=xxxx

        cd /path/to/workspace
        git clone https://github.com/openvinotoolkit/openvino.git 
        cd openvino
        git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
        git submodule update --init --recursive

        cd /path/to/workspace
        git clone https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin
        cd applications.ai.vpu-accelerators.vpux-plugin
        git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
        git submodule update --init --recursive
    ```
    <details>

2. Set environment variables and create a symlink for a preset file in OpenVINO Project root:
    <details>
    <summary>on Ubuntu</summary>
    
    ```sh
        export OPENVINO_HOME=/path/to/openvino
        export NPU_PLUGIN_HOME=/path/to/applications.ai.vpu-accelerators.vpux-plugin
        cd $OPENVINO_HOME
        ln -s $NPU_PLUGIN_HOME/CMakePresets.json ./CMakePresets.json
    ```
    </details>

    <details>
    <summary>x64 Native Tools Command Prompt for VS XXXX on Windows(run as administrator)</summary>
    
        ```bat
        set OPENVINO_HOME=\path\to\openvino
        set NPU_PLUGIN_HOME=\path\to\applications.ai.vpu-accelerators.vpux-plugin
        cd %OPENVINO_HOME%
        mklink .\CMakePresets.json %NPU_PLUGIN_HOME%\CMakePresets.json
        ```
    </details>

3. Build with the following commands:

    <details>
    <summary>on Ubuntu</summary>
    
    ```sh
        cmake --preset NpuCidRelease
        cd build-x86_64/Release/
        cmake --build ./ --target compilerTest profilingTest vpuxCompilerL0Test loaderTest
    ```
    </details>

    <details>
    <summary>x64 Native Tools Command Prompt for VS XXXX on Windows</summary>
    
    ```bat
        cmake --preset NpuCidRelease
        cd build-x86_64\Release\
        cmake --build .\ --target compilerTest profilingTest vpuxCompilerL0Test loaderTest
    ```
    </details>


### Using CMakePresets in a single tree with OpenVINO

Here provides a default pre-configured CMake presets for users named: "ovNpuCidRelease". The setting is to build [NPU-Plugin Project] as part of [OpenVINO Project]. In this case, NPU-plugin is considered as an OpenVINO module.

1. Clone repos
    <details>
    <summary> clone command</summary>

    ```sh
        # set the proxy, if required.
        # export/set  http_proxy=xxxx
        # export/set  https_proxy=xxxx

        cd /path/to/workspace
        git clone https://github.com/openvinotoolkit/openvino.git 
        cd openvino
        git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
        git submodule update --init --recursive

        cd /path/to/workspace
        git clone https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin
        cd applications.ai.vpu-accelerators.vpux-plugin
        git checkout -b master origin/master (Just an example, you could use your own branch/tag/commit.)
        git submodule update --init --recursive
    ```
    <details>

2. Set environment variables, put NPU Plugin Project into `$OPENVINO_HOME/modules/vpux`and create a symlink for a preset file in OpenVINO Project root.
    <details>
    <summary>on Ubuntu</summary>
    
    ```sh
        export OPENVINO_HOME=/path/to/openvino
        export NPU_PLUGIN_HOME=/path/to/applications.ai.vpu-accelerators.vpux-plugin

        mkdir $OPENVINO_HOME/modules
        mv $NPU_PLUGIN_HOME $OPENVINO_HOME/modules/vpux

        cd $OPENVINO_HOME
        ln -s ./modules/vpux/CMakePresets.json ./CMakePresets.json
    ```
    </details>

    <details>
    <summary>x64 Native Tools Command Prompt for VS XXXX on Windows</summary>
    
    ```bat
        set OPENVINO_HOME=\path\to\openvino
        set NPU_PLUGIN_HOME=\path\to\applications.ai.vpu-accelerators.vpux-plugin

        md %OPENVINO_HOME%\modules
        move %NPU_PLUGIN_HOME% %OPENVINO_HOME%\modules\vpux

        cd %OPENVINO_HOME%
        mklink  .\CMakePresets.json .\modules\vpux\CMakePresets.json
    ```
    </details>

3. Build with the following commands:
    <details>
    <summary>on Ubuntu</summary>
    
    ```sh
        cmake --preset ovNpuCidRelease
        cd build-x86_64/Release/
        cmake --build ./ --target compilerTest profilingTest vpuxCompilerL0Test loaderTest
    ```
    </details>

    <details>
    <summary>x64 Native Tools Command Prompt for VS XXXX on Windows</summary>
    
    ```bat
        cmake --preset ovNpuCidRelease
        cd build-x86_64\Release\
        cmake --build .\ --target compilerTest profilingTest vpuxCompilerL0Test loaderTest
    ```
    </details>

### Note

1. Presets step for "NpuCidRelease" and "ovNpuCidRelease" need to be built in [OpenVINO Project] folder.
2. The presets for "NpuCidRelease" and "ovNpuCidRelease" define the build directory build-x86_64/Release.
3. The presets are configured to use Ninja as default generator, so installing Ninja package is an extra requirement.
4. Currently Presets for "NpuCidRelease" and "ovNpuCidRelease" will build the smallest size targrts of Driver Compiler. If the user wishes to build Driver Compiler and other targets, can driectly inherit "Cid" preset and enable the needed build option to self configuration presets.

[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin
[CiD Project]: https://github.com/intel-innersource/applications.ai.vpu-accelerators.flex-cid-tools

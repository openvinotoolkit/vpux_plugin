# How to build VPUXCompilerL0 and related tests

# Windows

1. Set `%OPENVINO_HOME%` as the path of OpenVINO. Clone OpenVINO to `%OPENVINO_HOME%` directory. Check out the branch to a specific commit, if required.
   After that, clone OpenVINO submodules.

   ```sh
    set OPENVINO_HOME=/path/to/openvino
    git clone https://github.com/openvinotoolkit/openvino.git %OPENVINO_HOME%
    cd %OPENVINO_HOME%
    git checkout <fixed-commit-id>
    git submodule update --init --recursive
   ```
2. Set `%VPUX_PLUGIN_HOME%` as the path of VPUX Plugin. Clone VPUXPlugin to `%VPUX_PLUGIN_HOME%` directory and clone VPUX PLUGIN submodules.**OR** unpack VPUXPlugin source package to `%VPUX_PLUGIN_HOME%`.

3. Go to a build directory in `%OPENVINO_HOME%`

    ```sh
    mkdir %OPENVINO_HOME%\build-x86_64
    cd %OPENVINO_HOME%\build-x86_64
    ```

4. Build VPUXCompilerL0 library and related tests using the commands in `x64 Native Tools Command Prompt for VS 20xx`:

    ```sh
    cmake ^
        -D CMAKE_BUILD_TYPE=Release ^
        -D BUILD_SHARED_LIBS=OFF ^
        -D OPENVINO_EXTRA_MODULES=%VPUX_PLUGIN_HOME% ^
        -D ENABLE_TESTS=ON ^
        -D ENABLE_BLOB_DUMP=OFF ^
        -D ENABLE_HETERO=OFF ^
        -D ENABLE_MULTI=OFF ^
        -D ENABLE_AUTO_BATCH=OFF ^
        -D ENABLE_TEMPLATE=OFF ^
        -D ENABLE_OV_ONNX_FRONTEND=OFF ^
        -D ENABLE_OV_PYTORCH_FRONTEND=OFF ^
        -D ENABLE_OV_PADDLE_FRONTEND=OFF ^
        -D ENABLE_OV_TF_FRONTEND=OFF ^
        -D ENABLE_OV_TF_LITE_FRONTEND=OFF ^
        -D ENABLE_INTEL_CPU=OFF ^
        -D ENABLE_INTEL_GPU=OFF ^
        -D ENABLE_OV_IR_FRONTEND=ON ^
        -D BUILD_COMPILER_FOR_DRIVER=ON ^
        -D ENABLE_TBBBIND_2_5=OFF ^
        -D THREADING=TBB ^
        ..

    cmake --build . --config Release --target ov_dev_targets compilerTest profilingTest vpuxCompilerL0Test loaderTest -j 8
    ```

# Linux

1. Set `$OPENVINO_HOME` as the path of OpenVINO. Clone OpenVINO to `$OPENVINO_HOME` directory. Check out the branch to a specific commit, if required.
   After that, clone OpenVINO submodules.

   ```sh
    export OPENVINO_HOME=/path/to/openvino
    git clone https://github.com/openvinotoolkit/openvino.git $OPENVINO_HOME
    cd $OPENVINO_HOME
    git checkout <fixed-commit-id>
    git submodule update --init --recursive
   ```

2. Set `$VPUX_PLUGIN_HOME` as the path of VPUX Plugin. Clone VPUXPlugin to `$VPUX_PLUGIN_HOME` directory and clone VPUX PLUGIN submodules. **OR** unpack VPUXPlugin source package to `$VPUX_PLUGIN_HOME`.

3. Go to a build directory in `$OPENVINO_HOME`

    ```sh
    mkdir $OPENVINO_HOME/build-x86_64
    cd $OPENVINO_HOME/build-x86_64
    ```

4. Build VPUXCompilerL0 library and related tests using following commands:

    ```sh
    cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D BUILD_SHARED_LIBS=OFF \
        -D OPENVINO_EXTRA_MODULES=$VPUX_PLUGIN_HOME \
        -D ENABLE_TESTS=ON \
        -D ENABLE_BLOB_DUMP=OFF \
        -D ENABLE_HETERO=OFF \
        -D ENABLE_MULTI=OFF \
        -D ENABLE_AUTO_BATCH=OFF \
        -D ENABLE_TEMPLATE=OFF \
        -D ENABLE_OV_ONNX_FRONTEND=OFF \
        -D ENABLE_OV_PYTORCH_FRONTEND=OFF \
        -D ENABLE_OV_PADDLE_FRONTEND=OFF \
        -D ENABLE_OV_TF_FRONTEND=OFF \
        -D ENABLE_OV_TF_LITE_FRONTEND=OFF \
        -D ENABLE_INTEL_CPU=OFF \
        -D ENABLE_INTEL_GPU=OFF \
        -D ENABLE_OV_IR_FRONTEND=ON \
        -D BUILD_COMPILER_FOR_DRIVER=ON \
        -D ENABLE_TBBBIND_2_5=OFF \
        -D THREADING=TBB \
        ..

    cmake --build . --config Release --target ov_dev_targets compilerTest profilingTest vpuxCompilerL0Test loaderTest -j 8
    ```

> Notice: If you want to build static CiD package or onecore version of windows library, please follow commands in applications.ai.vpu-accelerators.flex-cid-tools repo.

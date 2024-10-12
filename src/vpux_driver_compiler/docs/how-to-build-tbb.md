# How to build TBB

Here are the steps for generating the TBB library on both Windows and Linux. To successfully pass some binary scanning tools, some patches are applied prior to building the TBB library.

## Content
* [Build Linux TBB library](#Build_Linux_TBB_library)
* [Build Windows TBB library](#Build_Windows_TBB_library)

## <span id="Build_Linux_TBB_library">Build Linux TBB library</span>

1. First, build hwloc:

    ```sh
    export WORKDIR=$(pwd)
    wget https://github.com/open-mpi/hwloc/archive/refs/tags/hwloc-2.9.2.tar.gz
    tar zxvf hwloc-2.9.2.tar.gz
    cd hwloc-hwloc-2.9.2/
    ./autogen.sh
    ./configure --disable-io --disable-libudev --disable-libxml2 --disable-cairo CFLAGS="-fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2"
    make -j8
    make install prefix=$(pwd)/install
    ```

2. build static TBB

    ```sh
    cd ${WORKDIR}
    wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.2.4.tar.gz
    tar zxvf v2021.2.4.tar.gz
    cd oneTBB-2021.2.4
    cp $WORKDIR/npu_splugin/src/vpux_driver_compiler/docs/patch/linux/hwloc_2_9_2.diff ./
    git apply hwloc_2_9_2.diff
    mkdir build_release
    cd build_release
    cmake -D BUILD_SHARED_LIBS=OFF \
          -D CMAKE_HWLOC_2_9_LIBRARY_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/lib/libhwloc.so \
          -D CMAKE_HWLOC_2_9_INCLUDE_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/include \
          -D CMAKE_HWLOC_2_9_DLL_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/lib \
            -D TBB_STRICT=OFF \
            -D TBB_TEST=OFF \
            -D CMAKE_CXX_FLAGS=-fPIC \
            -D TBB_COMMON_COMPILE_FLAGS="-fstack-protector-all -D_FORTIFY_SOURCE=2" \
            -D TBB_LIB_LINK_FLAGS="-Wl,-z,now" \
            -D CMAKE_BUILD_TYPE=Release \
            ..
    cmake --build . --config Release
    cmake --install . --prefix ../install --config Release
    cp ../LICENSE.txt ../install/LICENSE
    cd ..
    mkdir build_debug
    cd build_debug
    cmake -D BUILD_SHARED_LIBS=OFF \
          -D CMAKE_HWLOC_2_9_LIBRARY_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/lib/libhwloc.so \
          -D CMAKE_HWLOC_2_9_INCLUDE_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/include \
          -D CMAKE_HWLOC_2_9_DLL_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/lib \
            -D TBB_STRICT=OFF \
            -D TBB_TEST=OFF \
            -D CMAKE_CXX_FLAGS=-fPIC \
	        -D TBB_COMMON_COMPILE_FLAGS="-fstack-protector-all" \
	        -D TBB_LIB_LINK_FLAGS="-Wl,-z,now" \
            -D CMAKE_BUILD_TYPE=Debug \
            ..
    cmake --build . --config Debug
    cmake --install . --prefix ../install --config Debug
    ```

3. build dynamic TBB

    ```sh
    cd ${WORKDIR}
    wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.2.4.tar.gz
    tar zxvf v2021.2.4.tar.gz
    cd oneTBB-2021.2.4
    cp $WORKDIR/npu_splugin/src/vpux_driver_compiler/docs/patch/linux/hwloc_2_9_2.diff ./
    git apply hwloc_2_9_2.diff
    mkdir build_release
    cd build_release
    cmake -D BUILD_SHARED_LIBS=ON \
          -D CMAKE_HWLOC_2_9_LIBRARY_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/lib/libhwloc.so \
          -D CMAKE_HWLOC_2_9_INCLUDE_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/include \
          -D CMAKE_HWLOC_2_9_DLL_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/lib \
            -D TBB_STRICT=OFF \
            -D TBB_TEST=OFF \
            -D CMAKE_CXX_FLAGS=-fPIC \
            -D TBB_COMMON_COMPILE_FLAGS="-fstack-protector-all -D_FORTIFY_SOURCE=2" \
            -D TBB_LIB_LINK_FLAGS="-Wl,-z,now" \
            -D CMAKE_BUILD_TYPE=Release \
            ..
    cmake --build . --config Release
    cmake --install . --prefix ../install --config Release
    cp ../LICENSE.txt ../install/LICENSE
    cd ..
    mkdir build_debug
    cd build_debug
    cmake -D BUILD_SHARED_LIBS=ON \
          -D CMAKE_HWLOC_2_9_LIBRARY_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/lib/libhwloc.so \
          -D CMAKE_HWLOC_2_9_INCLUDE_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/include \
          -D CMAKE_HWLOC_2_9_DLL_PATH=${WORKDIR}/hwloc-hwloc-2.9.2/install/lib \
            -D TBB_STRICT=OFF \
            -D TBB_TEST=OFF \
            -D CMAKE_CXX_FLAGS=-fPIC \
            -D TBB_COMMON_COMPILE_FLAGS="-fstack-protector-all" \
            -D TBB_LIB_LINK_FLAGS="-Wl,-z,now" \
            -D CMAKE_BUILD_TYPE=Debug \
            ..
    cmake --build . --config Debug
    cmake --install . --prefix ../install --config Debug
    ```

Finally, you need to set the environment variable TBBROOT to this oneTBB install folder `export TBBROOT="${WORKDIR}/oneTBB-2021.2.4/install"` to perform Driver compiler build.


## <span id="Build_Windows_TBB_library">Build Windows oneTBB library</span>

Currently, Windows TBB code is from https://github.com/oneapi-src/oneTBB/releases/tag/v2021.2.5, align with OpenVINO prebuilt TBB library

1. First, determine the hwloc version, find the ```tbbbind_2_5_debug.lib``` file in the temp directory of openvino, and then use the ```dumpbin /ALL tbbbind_2_5_debug.lib|grep -i hwl |tee logfile``` command to get the hwloc version information, here we get ```hwloc-2.8.0```. download hwloc from https://github.com/open-mpi/hwloc/archive/refs/tags/hwloc-2.8.0.zip and then build hwloc library

```bat
set WORKDIR=%cd%
cd hwloc-hwloc-2.8.0\contrib\windows-cmake
cmake -A X64 --install-prefix=%cd%\install -DHWLOC_SKIP_TOOLS=ON -DHWLOC_WITH_LIBXML2=OFF -DBUILD_SHARED_LIBS=ON -B build
cmake --build build --parallel --config Release
cmake --install build --config Release
set hwloc_install_dir=%WORKDIR%\hwloc-hwloc-2.8.0\contrib\windows-cmake\install
```

2. Clone oneTBB and build static library. Please make sure that change ALL 2_4 to 2_5 in source code

```bat
::set TBB_HOME to appropriate location
git clone https://github.com/oneapi-src/oneTBB.git %TBB_HOME%
cd %TBB_HOME%
git checkout v20xx.x
```

To generate pdb file for TBB release, need to add following code to CMakeLists.txt. Or you can use [this  patch](./patch/winodws/1.static_patch.diff) in oneTBB repo.

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
```

Then, build static onecore TBB

```bat
mkdir build
cd build
cmake ^
    -G "Visual Studio 16 2019" -A X64 ^
    -D CMAKE_INSTALL_PREFIX=%cd%\install ^
    -D BUILD_SHARED_LIBS=OFF ^
    -D CMAKE_HWLOC_2_5_LIBRARY_PATH=%hwloc_install_dir%\lib\hwloc.lib ^
    -D CMAKE_HWLOC_2_5_INCLUDE_PATH=%hwloc_install_dir%\include ^
    -D CMAKE_HWLOC_2_5_DLL_PATH=%hwloc_install_dir%\bin\hwloc.dll ^
    -D TBB_TEST=OFF ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" ^
    ..
        
cmake --build . --config Release
cmake --install . --config Release
cmake --build . --config Debug
cmake --install . --config Debug
copy ..\LICENSE.txt install\LICENSE
copy %hwloc_install_dir%\bin\hwloc.dll install\lib
```

The install folder contains all the static TBB library, which is named as release-tools\TBBPackage\win-release-static.

**Do NOT forget to copy pdb files into final TBB package.**

3. Build dynamic onecore TBB library. Please make sure that change ALL 2_4 to 2_5 in source code

To avoid BinSkim issue for TBB, need to add following code to CMakeLists.txt

```cmake
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /FS /Zf /ZH:SHA_256 /guard:cf /Qspectre /sdl")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi /FS /Zf /ZH:SHA_256 /guard:cf /Qspectre /sdl")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf /sdl  /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf /sdl /LTCG")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf /sdl /LTCG")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /FS /Zf /ZH:SHA_256 /guard:cf /Qspectre /sdl")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Zi /FS /Zf /ZH:SHA_256 /guard:cf /Qspectre /sdl")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf /sdl /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf /sdl /LTCG")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf /sdl /LTCG")
```

And need to change tbbmalloc warning level from ```set(TBB_WARNING_SUPPRESS ${TBB_WARNING_SUPPRESS} /wd4267 /wd4244 /wd4245 /wd4018 /wd4458)``` to ```set(TBB_WARNING_SUPPRESS ${TBB_WARNING_SUPPRESS} /W3 /wd4245 /wd4458)``` on [oneTBB/src/tbbmalloc/CMakeLists.txt](https://github.com/oneapi-src/oneTBB/blob/47061f128b14f758d1a9d5e90d0cfc5fa9793b89/src/tbbmalloc/CMakeLists.txt#L50)

Or to fix the above issue, you can use [this patch](./patch/winodws/2.2tbb_dynmaic.diff) in oneTBB repo.

Then, build dynamic onecore TBB

```bat
mkdir build-dynamic
cd build-dynamic
cmake ^
    -G "Visual Studio 16 2019" -A X64 ^
    -D CMAKE_INSTALL_PREFIX=%cd%\install ^
    -D BUILD_SHARED_LIBS=ON ^
    -D CMAKE_HWLOC_2_5_LIBRARY_PATH=%hwloc_install_dir%\lib\hwloc.lib ^
    -D CMAKE_HWLOC_2_5_INCLUDE_PATH=%hwloc_install_dir%\include ^
    -D CMAKE_HWLOC_2_5_DLL_PATH=%hwloc_install_dir%\bin\hwloc.dll ^
    -D TBB_TEST=OFF ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" ^
    ..

cmake --build . --config Release
cmake --install . --config Release
cmake --build . --config Debug
cmake --install . --config Debug
copy ..\LICENSE.txt install\LICENSE
copy %hwloc_install_dir%\bin\hwloc.dll install\bin
```

The install folder contains all the dynamic TBB library, which is named as release-tools\TBBPackage\win-release-dynamic.

**Do NOT forget to copy pdb files into final TBB package.**


[oneTBB Project]: https://github.com/oneapi-src/oneTBB
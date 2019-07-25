# kmb-plugin

KMBPlugin for Inference Engine


## How to build
There are two variants to build KMBPlugin: build-script and manual.
But for both variants you must first of all to build Inference Engine in dldt with 
script "dldt/inference-engine/build-after-clone.sh" or see instructions in "dldt/inference-engine/CONTRIBUTING.md".

## Build with help of script:
1. Clone kmb-plugin from repository: git clone git@gitlab-icv.inn.intel.com:inference-engine/kmb-plugin.git
2. Find bash-script "build_after_clone.sh" in the base directory of KMBPlugin and run it. 
3. When build finishes its work check output for possible errors.
4. Then run script "run_tests_after_build.sh" to check that you have built KMBPlugin correctly.

## Manual build:
1. Create variables with path to base directories of kmb-plugin and dldt:
You could use such commands:
- go to base dldt directory and make DLDT_HOME variable with command: 

* export DLDT_HOME=$(pwd)

- go to base kmb-plugin directory and make KMB_PLUGIN_HOME variable with command:

* export KMB_PLUGIN_HOME=$(pwd)


2. Install additional packages for KMBPlugin:

Install Swig with command: 

* sudo apt install swig

Install python3-dev with command: 

* sudo apt install python3-dev

Install python-numpy with command: 

* sudo apt install python-numpy

Install metis with command: 

* sudo apt install libmetis-dev libmetis5 metis

3. Move to dldt base directory and make some building with commands. Note that if you miss -DCMAKE_BUILD_TYPE=Debug then you will not be able to debug your code in kmb-plugin:

* cd $(DLDT_HOME)
* mkdir $(DLDT_HOME)/inference-engine/build
* cd $(DLDT_HOME)/inference-engine/build
* cmake -DENABLE_TESTS=ON -DENABLE_BEH_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DENABLE_PLUGIN_RPATH=ON -DCMAKE_BUILD_TYPE=Debug ..
* make -j8


4. There is a little bug in current version of inference-engine so you have to eliminate it with commands:

* rm $(DLDT_HOME)/inference-engine/build/targets_developer.cmake
* cd $(DLDT_HOME)/inference-engine/build/
* cmake ..

5. Move to base directory of KMBPlugin and build it with commands:

* cd $(KMB_PLUGIN_HOME)
* export MCM_HOME=$(KMB_PLUGIN_HOME)/thirdparty/movidius/mcmCompiler
* git submodule update --init --recursive
* mkdir $(KMB_PLUGIN_HOME)/build
* cd $(KMB_PLUGIN_HOME)/build
* cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/inference-engine/build ..
* make -j8


6. To check results of previous steps it is recommended to execute tests with the following commands:

If you built Inference Engine with parameter "-DENABLE_PLUGIN_RPATH=ON" then go to command beginning with "export MCM_HOME..", otherwise enter these commands:
* export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(DLDT_HOME)/inference-engine/bin/intel64/Release/lib
* export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(DLDT_HOME)/inference-engine/temp/opencv_4.1.0_ubuntu18/lib
* export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(DLDT_HOME)/inference-engine/temp/tbb/lib
* export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(KMB_PLUGIN_HOME)/thirdparty/vsi_cmodel/vpusmm/x86_64
* export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(KMB_PLUGIN_HOME)/thirdparty/movidius/mcmCompiler/build/lib

export MCM_HOME=$(KMB_PLUGIN_HOME)/thirdparty/movidius/mcmCompiler
* cd $(DLDT_HOME)/inference-engine/bin/intel64/Release/
* ./KmbBehaviorTests --gtest_filter=\*Behavior\*orrectLib\*kmb\*
* ./KmbFunctionalTests

If you see that all enabled tests are passed then you may congratulate yourself with successful build of KMBPlugin.

## Cross build for Yocto

Cross build use Yocto SDK. You can install it with:

``` sh
wget -q http://nnt-srv01.inn.intel.com/dl_score_engine/thirdparty/linux/keembay/stable/ww28.5/oecore-x86_64-aarch64-toolchain-1.0.sh && \
        chmod +x oecore-x86_64-aarch64-toolchain-1.0.sh && \
        ./oecore-x86_64-aarch64-toolchain-1.0.sh -y -d /usr/local/oecore-x86_64 && \
        rm oecore-x86_64-aarch64-toolchain-1.0.sh
```
1. Clone dldt:

```sh
(cd .. && git clone git@gitlab-icv.inn.intel.com:inference-engine/dldt.git)
```

2. Configure and build inference engine:

Run following command from temporary build folder (e.g. `dldt\inference-engine\build`):

```sh
(\
  mkdir -p ../dldt/inference-engine/build && \
  cd ../dldt/inference-engine/build && \
  source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux && \
  rm targets*.cmake ; cmake -DENABLE_TESTS=ON .. && \
  cmake --build . --parallel $(nproc) \
)
```

Note: the command attempt to clean the auto-generated `targets*.cmake`. Due to
bug in in inference engine you need to clean `targets*.cmake` before cmake
execution.

3. Build plugin:

Run following command from temporary build folder (e.g. `kmb-plugin\build`):

```sh
(\
  source /usr/local/oecore-x86_64/environment-setup-aarch64-ese-linux && \
  cmake -DInferenceEngineDeveloperPackage_DIR=$(realpath ../../dldt/inference-engine/build) .. &&\
  cmake --build . --parallel $(nproc) \
)
```
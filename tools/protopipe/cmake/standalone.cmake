set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(gflags REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(Threads REQUIRED)
find_package(OpenCV 4.9.0 REQUIRED COMPONENTS gapi)

set(DEPENDENCIES
        Threads::Threads
        gflags
        yaml-cpp
        openvino::runtime
        opencv_gapi
)

if (WIN32)
    list(APPEND DEPENDENCIES "winmm.lib")
endif()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(APPEND SOURCES main.cpp)

add_executable(${TARGET_NAME} ${SOURCES})
target_link_libraries(${TARGET_NAME} ${DEPENDENCIES})
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/src/")

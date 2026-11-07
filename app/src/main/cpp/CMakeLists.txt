cmake_minimum_required(VERSION 3.22.1)
project(qvac_chatterbox_android LANGUAGES C CXX)

if (NOT ANDROID)
    message(FATAL_ERROR "Build this prototype with the Android NDK.")
endif()

if (NOT DEFINED QVAC_TTS_SOURCE_DIR)
    message(FATAL_ERROR "QVAC_TTS_SOURCE_DIR is not configured.")
endif()

if (NOT EXISTS "${QVAC_TTS_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "QVAC_TTS_SOURCE_DIR must point to qvac-fabric-speech.cpp/engines/tts")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Only the QVAC TTS library is linked. Tests and command-line tools are disabled.
set(TTS_CPP_BUILD_LIBRARY ON CACHE BOOL "" FORCE)
set(TTS_CPP_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(TTS_CPP_BUILD_EXECUTABLES OFF CACHE BOOL "" FORCE)
set(TTS_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TTS_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(TTS_CPP_USE_SYSTEM_GGML OFF CACHE BOOL "" FORCE)
set(TTS_CPP_OPENMP OFF CACHE BOOL "" FORCE)

# First Android prototype: Vulkan when available, with QVAC's CPU fallback.
# OpenCL and Hexagon are excluded because they need vendor-specific Android
# OpenCL libraries or the separate Qualcomm SDK/toolchain.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
# CPU variants are runtime-loaded plugins and require GGML_BACKEND_DL=ON.
# This prototype statically links GGML backends, so keep the portable CPU backend.
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_VULKAN_DISABLE_COOPMAT ON CACHE BOOL "" FORCE)
set(GGML_VULKAN_DISABLE_COOPMAT2 ON CACHE BOOL "" FORCE)

add_subdirectory("${QVAC_TTS_SOURCE_DIR}" qvac-tts)

add_library(qvac_tts_jni SHARED qvac_tts_jni.cpp)
target_include_directories(qvac_tts_jni PRIVATE
    "${QVAC_TTS_SOURCE_DIR}/ggml/include"
)
target_compile_features(qvac_tts_jni PRIVATE cxx_std_17)
target_compile_options(qvac_tts_jni PRIVATE -fexceptions -frtti)
target_link_libraries(qvac_tts_jni PRIVATE tts-cpp android log)

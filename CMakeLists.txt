cmake_minimum_required(VERSION 3.12)
project(onnx_yolo VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit QUIET)  # CUDA support (if available)

# Import ONNXRuntime
set(ONNXRuntime_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime)
set(ONNXRuntime_INCLUDE_DIRS "${ONNXRuntime_ROOT}/include")
set(ONNXRuntime_LIB_DIR "${ONNXRuntime_ROOT}/lib")

# Detect OS and set ONNXRuntime library path accordingly
if(WIN32)
    set(ONNXRuntime_LIBS "${ONNXRuntime_LIB_DIR}/onnxruntime.lib")
    set(ONNXRuntime_DLL "${ONNXRuntime_LIB_DIR}/onnxruntime.dll")
elseif(UNIX AND NOT APPLE)  # Linux
    set(ONNXRuntime_LIBS "${ONNXRuntime_LIB_DIR}/libonnxruntime.so")
elseif(APPLE)  # macOS
    set(ONNXRuntime_LIBS "${ONNXRuntime_LIB_DIR}/libonnxruntime.dylib")
else()
    message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

# Detect CUDA support
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: Enabling CUDA support")
    add_definitions(-DHAS_CUDA)
endif()

# Detect CoreML support on macOS
if(APPLE)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${ONNXRuntime_INCLUDE_DIRS}")
    check_cxx_source_compiles("
        #include <coreml_provider_factory.h>
        int main() { return 0; }
    " HAS_COREML_HEADERS)

    if(HAS_COREML_HEADERS)
        message(STATUS "CoreML headers found: Enabling CoreML support")
        add_definitions(-DHAS_COREML)
    else()
        message(WARNING "CoreML headers not found, disabling CoreML support.")
    endif()
endif()

# Add sources
file(GLOB_RECURSE SOURCES src/*.cpp)

# Create library
add_library(onnx_yolo SHARED ${SOURCES})

target_include_directories(onnx_yolo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${ONNXRuntime_INCLUDE_DIRS}>
)

# Link dependencies
target_link_libraries(onnx_yolo PRIVATE
  ${OpenCV_LIBS}
  ${ONNXRuntime_LIBS}
)

# RPATH settings for shared libraries
set_target_properties(onnx_yolo PROPERTIES
    CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CMAKE_INSTALL_RPATH "\$ORIGIN/../lib"
)

# Installation rules
install(TARGETS onnx_yolo
    EXPORT onnx_yolo_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT onnx_yolo_targets
    FILE onnx_yoloConfig.cmake
    DESTINATION lib/cmake/onnx_yolo
)

# Copy ONNXRuntime DLL for Windows users
if(WIN32)
    install(FILES ${ONNXRuntime_DLL} DESTINATION bin)
endif()

# Add example programs
add_subdirectory(examples)
